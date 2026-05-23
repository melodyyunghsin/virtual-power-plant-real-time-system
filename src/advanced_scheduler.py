"""
VPP Real-Time Scheduling System — Event-Triggered Rolling-Horizon Scheduler.

Wraps the static Phase 1 ILP in a 72-hour simulation that:
  * reveals pv_actual / realtime prices one hour at a time;
  * re-optimises over a 12-hour window when a trigger fires
      - PV forecast deviation > 15 %
      - sporadic / aperiodic job arrival
      - periodic refresh every 6 hours;
  * locks day_ahead_commit[t] at the static value and applies a
    cancellation penalty (cancel_rate · day-ahead price · shortfall) when
    realised sell[t] falls below the commitment.

The replan ILP re-decides dispatch (P, k, sell, chg/dis/soc, generator
on/off) inside [t_now, t_end] while freezing periodic / accepted-sporadic
placements from the current plan and letting newly arrived sporadic /
aperiodic jobs be placed by the solver.

Output:
  output/schedule_result_advanced.json
  output/evaluation_results_advanced.json
"""

import copy
import json
import math
from pathlib import Path

import pulp

from scheduler import (
    H, ALPHA, EPS, RESERVE_PER_GEN,
    OUTPUT_DIR,
    load_inputs, parse_renewable_forecast, parse_renewable_actuals,
    parse_forecast_error_std, parse_price, parse_price_extended,
    expand_periodic, expand_aperiodic,
)


# ============================================================================
# Helpers
# ============================================================================

def _planned_hours(plan, jid):
    """Return list of hours where the plan has an allocation entry under jid.

    The static schedule keys k by task_id (e.g., "p1"), not by instance id —
    instances of the same task never overlap (d_j <= p_j), so each hour with
    that key belongs to exactly one instance, which we can resolve by
    matching against the instance window.
    """
    out = []
    for slot in plan:
        if jid in slot["k"]:
            out.append(slot["t"])
    return out


def _instance_hours(plan, periodic_instance):
    """Hours in the plan that belong to a specific periodic instance.

    The plan keys k entries by task_id; multiple instances of the same task
    share that key. We split them by checking against each instance's
    [release, deadline] window.
    """
    task_id = periodic_instance["task_id"]
    lo = periodic_instance["release"]
    hi = periodic_instance["deadline"]
    return [slot["t"] for slot in plan
            if task_id in slot["k"] and lo <= slot["t"] <= hi]


# ============================================================================
# AdvancedScheduler
# ============================================================================

class AdvancedScheduler:
    WINDOW = 12
    REPLAN_INTERVAL = 6
    PV_DEV_THRESH = 0.15
    REPLAN_TIME_LIMIT = 30

    # -- init ---------------------------------------------------------------
    def __init__(self):
        self.proc, price_data, task_set = load_inputs()
        self.pv_forecast = parse_renewable_forecast(self.proc)
        self.pv_actual = parse_renewable_actuals(self.proc)
        self.err_std = parse_forecast_error_std(self.proc)
        self.price_arr = parse_price(price_data)
        self.cancel_rate, self.rt_factors = parse_price_extended(price_data)

        self.gen_by_id = {g["generator_id"]: g for g in self.proc["generator"]}
        self.pv_by_id = {r["renewable_id"]: r for r in self.proc["renewable_capacity"]}
        self.bat_by_id = {b["storage_id"]: b for b in self.proc["storage"]}
        self.gen_ids = list(self.gen_by_id)
        self.pv_ids = list(self.pv_by_id)
        self.bat_ids = list(self.bat_by_id)
        self.proc_ids = self.gen_ids + self.pv_ids + self.bat_ids
        self.chg_jobs = [{"id": f"{b}_chg", "battery": b} for b in self.bat_ids]

        self.periodic_jobs = expand_periodic(task_set.get("periodic", {}))
        self.aperiodic_jobs = expand_aperiodic(task_set.get("aperiodic", []))
        # Normalize sporadic to a list of dicts each carrying its id.
        # New input format is dict-of-dict ({"s1": {...}}); older was list.
        sp_raw = task_set.get("sporadic", [])
        if isinstance(sp_raw, dict):
            self.sporadic_input = [{"id": sid, **s} for sid, s in sp_raw.items()]
        else:
            self.sporadic_input = [{"id": s.get("id", f"s{i}"), **s}
                                   for i, s in enumerate(sp_raw)]

        # Initial plan from static schedule
        static_path = OUTPUT_DIR / "schedule_result.json"
        static_data = json.loads(static_path.read_text(encoding="utf-8"))
        self.plan = copy.deepcopy(static_data["schedule_result"])
        # Day-ahead commitment locked from static
        self.locked_commit = {
            slot["t"]: float(slot.get("day_ahead_commit", slot.get("sell", 0.0)))
            for slot in self.plan
        }

        # Simulation state (updated each executed hour)
        self.soc = {b: float(self.bat_by_id[b]["soc_init"]) for b in self.bat_ids}
        self.gen_state = {}
        for g in self.gen_ids:
            gd = self.gen_by_id[g]
            ion = int(gd.get("initial_on_time", 0))
            self.gen_state[g] = {
                "on": ion > 0,
                "on_h": ion,
                "off_h": int(gd.get("initial_off_time", 0)),
                "last_P": float(gd.get("initial_energy", 0)),
            }

        # Tracking
        self.replans = 0
        self.replan_triggers = {"pv_deviation": 0, "sporadic_arrival": 0,
                                "periodic_6h": 0, "aperiodic_arrival": 0}
        self.last_periodic_replan = 0
        self.total_penalty = 0.0
        self.acceptance_log = []        # sporadic acceptance decisions
        self.aperiodic_queue = []       # arrived, not yet placed
        self.aperiodic_placements = {}  # jid -> {slots, decision, ...}
        self.replan_failures = 0
        self.actual_schedule = []       # what really executed each hour

    # -- main loop ----------------------------------------------------------
    def run(self):
        for t in range(1, H + 1):
            arriving_sp = [s for s in self.sporadic_input
                           if int(s.get("r", s.get("release", -1))) == t]
            arriving_ap = [j for j in self.aperiodic_jobs if j["release"] == t]

            # newly-arrived aperiodic enter the queue
            for aj in arriving_ap:
                if aj["id"] not in self.aperiodic_placements:
                    self.aperiodic_queue.append(aj)

            # sporadic acceptance test (uses current plan + tentative replan)
            accepted_now = []
            for sj in arriving_sp:
                rec = self._process_sporadic(sj, t)
                self.acceptance_log.append(rec)
                if rec["decision"] == "accept":
                    accepted_now.append(rec)

            # triggers
            triggers = []
            if accepted_now:
                triggers.append("sporadic_arrival")
            if arriving_ap:
                triggers.append("aperiodic_arrival")
            if t - self.last_periodic_replan >= self.REPLAN_INTERVAL:
                triggers.append("periodic_6h")
            if self._pv_deviation(t) > self.PV_DEV_THRESH:
                triggers.append("pv_deviation")

            if triggers:
                ok = self._rolling_replan(t, accepted_now)
                if ok:
                    self.replans += 1
                    for tr in triggers:
                        self.replan_triggers[tr] += 1
                    if "periodic_6h" in triggers:
                        self.last_periodic_replan = t
                else:
                    self.replan_failures += 1

            self._execute_hour(t)

        # Any aperiodic that never got placed by a replan: try fitting them
        # via slack on the final realised schedule.
        self._final_aperiodic_sweep()

    # -- PV deviation -------------------------------------------------------
    def _pv_deviation(self, t):
        max_dev = 0.0
        for pv in self.pv_ids:
            fc = self.pv_forecast[pv][t]
            ac = self.pv_actual[pv][t]
            if fc > 0.01:
                max_dev = max(max_dev, abs(ac - fc) / fc)
        return max_dev

    # -- sporadic acceptance ------------------------------------------------
    def _process_sporadic(self, sj, t_now):
        """Cheap slack-based acceptance test on the current plan."""
        sid = str(sj["id"])
        # New format uses {"r","d",…}; older used {"release","hard_deadline"|"deadline",…}.
        r = int(sj.get("r", sj.get("release")))
        d = int(sj.get("d", sj.get("hard_deadline", sj.get("deadline", H))))
        e = int(sj["e"])
        w = float(sj["w"])
        preempt = int(sj.get("preempt", 1))

        # Free capacity at hour t in current plan = P[i,t]-used[i,t] + sell[t]
        # + on-generator headroom + PV headroom (capped at robust limit).
        def _slack(tt):
            slot = self.plan[tt - 1]
            s = slot.get("sell", 0.0)
            for g in self.gen_ids:
                p_curr = slot["P"].get(g, 0.0)
                if p_curr <= EPS:
                    continue
                gd = self.gen_by_id[g]
                p_prev = (self.plan[tt - 2]["P"].get(g, 0.0)
                          if tt > 1 else 0.0)
                p_next = (self.plan[tt]["P"].get(g, 0.0)
                          if tt < H else p_curr)
                max_new = min(float(gd["output_max"]),
                              p_prev + gd["ramp_up_rate"],
                              p_next + gd["ramp_down_rate"])
                s += max(0.0, max_new - p_curr)
            for pv in self.pv_ids:
                cap = self.pv_by_id[pv]["capacity"]
                avail = cap * self.pv_forecast[pv][tt] * (1 - self.err_std)
                s += max(0.0, avail - slot["P"].get(pv, 0.0))
            return s

        window = list(range(max(r, t_now), min(d, H) + 1))
        if preempt == 0:
            chosen = []
            for start in range(window[0], window[-1] - e + 2):
                blk = list(range(start, start + e))
                if all(_slack(tt) >= w - EPS for tt in blk):
                    chosen = blk
                    break
            if not chosen:
                return {"job_id": sid, "decision": "reject", "arrival": r,
                        "release": r, "deadline": d, "e": e, "w": w,
                        "reason": f"no contiguous {e}h slot of >={w}MW in [{r},{d}]",
                        "caused_violation": False}
        else:
            feas = [tt for tt in window if _slack(tt) >= w - EPS]
            if len(feas) < e:
                return {"job_id": sid, "decision": "reject", "arrival": r,
                        "release": r, "deadline": d, "e": e, "w": w,
                        "reason": f"only {len(feas)} feasible hours, need {e}",
                        "caused_violation": False}
            chosen = feas[:e]

        return {"job_id": sid, "decision": "accept", "arrival": r,
                "release": r, "deadline": d, "e": e, "w": w,
                "slots": chosen, "preempt": preempt,
                "caused_violation": False}

    # -- rolling replan -----------------------------------------------------
    def _rolling_replan(self, t_now, newly_accepted_sporadic):
        """Re-optimise dispatch over [t_now, t_end].

        Periodic / previously-accepted sporadic placements inside the window
        are frozen (their k allocations are FIXED). The ILP redecides:
          - thermal P, on/off, ramp;
          - battery chg / dis / soc;
          - renewable P (bounded by pv_actual for t_now, pv_forecast·0.92 else);
          - sell[t] and (penalty / over-revenue) decomposition;
          - placement (x, k) of newly-accepted sporadic + queued aperiodic.
        Returns True iff solve succeeded and plan was updated.
        """
        t_end = min(t_now + self.WINDOW - 1, H)
        T = list(range(t_now, t_end + 1))

        # ------- collect new jobs to place (sporadic just accepted +
        #          queued aperiodic) — built first so we know which jids
        #          to exclude from frozen demand.
        new_jobs = []
        for rec in newly_accepted_sporadic:
            new_jobs.append({
                "id":       rec["job_id"],
                "release":  max(rec["release"], t_now),
                "deadline": min(rec["deadline"], t_end),
                "e":        rec["e"],
                "w":        rec["w"],
                "preempt":  rec.get("preempt", 1),
                "kind":     "sporadic",
            })
        for aj in list(self.aperiodic_queue):
            r = aj["release"]
            d = aj["deadline"]
            if r > t_end or d < t_now:
                continue
            new_jobs.append({
                "id":       aj["id"],
                "release":  max(r, t_now),
                "deadline": min(d, t_end),
                "e":        aj["e"],
                "w":        aj["w"],
                "preempt":  aj.get("preempt", 1),
                "kind":     "aperiodic",
            })
        new_job_ids = {j["id"] for j in new_jobs}
        queued_ap_ids = {aj["id"] for aj in self.aperiodic_queue}

        # ------- collect frozen demand (periodic + already-placed jobs) -----
        # Skip any jid that the replan is going to re-decide (queued aperiodic
        # or newly-accepted sporadic). That avoids double-counting the same
        # demand both as frozen and as a new placement.
        frozen_demand = []     # list of (jid, t, w)
        for slot in self.plan:
            if slot["t"] not in T:
                continue
            for jid, alloc in slot["k"].items():
                if jid.endswith("_chg"):
                    continue
                if jid in new_job_ids or jid in queued_ap_ids:
                    continue
                total_w = sum(alloc.values())
                if total_w > EPS:
                    frozen_demand.append((jid, slot["t"], total_w))

        # ------- build ILP -------------------------------------------------
        prob = pulp.LpProblem(f"VPP_Replan_t{t_now}", pulp.LpMinimize)

        P = pulp.LpVariable.dicts("P", (self.proc_ids, T), lowBound=0)
        u = pulp.LpVariable.dicts("u", (self.gen_ids, T), cat="Binary")
        z_on = pulp.LpVariable.dicts("zon", (self.gen_ids, T), cat="Binary")
        z_off = pulp.LpVariable.dicts("zoff", (self.gen_ids, T), cat="Binary")
        chg = pulp.LpVariable.dicts("chg", (self.bat_ids, T), lowBound=0)
        dis = pulp.LpVariable.dicts("dis", (self.bat_ids, T), lowBound=0)
        soc = pulp.LpVariable.dicts("soc", (self.bat_ids, T), lowBound=0)
        v_chg = pulp.LpVariable.dicts("vchg", (self.bat_ids, T), cat="Binary")
        sfrac = pulp.LpVariable.dicts("sfrac", (self.bat_ids, T),
                                      lowBound=0, upBound=1)
        sell = pulp.LpVariable.dicts("sell", T, lowBound=0)
        sell_share = pulp.LpVariable.dicts("ss", (self.proc_ids, T), lowBound=0)
        s_under = pulp.LpVariable.dicts("sund", T, lowBound=0)
        s_over = pulp.LpVariable.dicts("sov", T, lowBound=0)

        # K[jid][i][t] for frozen demand allocations
        K_frozen = {}
        for jid, t, w in frozen_demand:
            K_frozen.setdefault(jid, {}).setdefault(t, {})
            for i in self.proc_ids:
                K_frozen[jid][t][i] = pulp.LpVariable(
                    f"kf_{jid}_{i}_{t}", lowBound=0)

        # K + x for new jobs (sporadic + aperiodic)
        x = {}
        y = {}
        miss = {}
        K_new = {}
        for j in new_jobs:
            jid = j["id"]
            if j["release"] > j["deadline"]:
                # No feasible hour in window; if aperiodic, force miss
                if j["kind"] == "aperiodic":
                    miss[jid] = pulp.LpVariable(f"miss_{jid}", cat="Binary")
                    prob += miss[jid] == 1
                continue
            x[jid] = {t: pulp.LpVariable(f"x_{jid}_{t}", cat="Binary")
                      for t in range(j["release"], j["deadline"] + 1)}
            if j["preempt"] == 0:
                starts = list(range(j["release"],
                                    j["deadline"] - j["e"] + 2))
                y[jid] = {s: pulp.LpVariable(f"y_{jid}_{s}", cat="Binary")
                          for s in starts}
            if j["kind"] == "aperiodic":
                miss[jid] = pulp.LpVariable(f"miss_{jid}", cat="Binary")
            K_new[jid] = {}
            for i in self.proc_ids:
                K_new[jid][i] = {
                    t: pulp.LpVariable(f"kn_{jid}_{i}_{t}", lowBound=0)
                    for t in x[jid]
                }

        # Kchg[bat_chg][i][t]
        Kchg = {}
        for cj in self.chg_jobs:
            cid = cj["id"]
            Kchg[cid] = {}
            for i in self.gen_ids + self.pv_ids:
                Kchg[cid][i] = {t: pulp.LpVariable(
                    f"kc_{cid}_{i}_{t}", lowBound=0) for t in T}

        # ------- objective -------------------------------------------------
        f1 = pulp.lpSum(miss.values()) if miss else 0
        f2 = pulp.lpSum(
            self.gen_by_id[g]["cost_fixed"] * u[g][t]
            + self.gen_by_id[g]["cost_variable"] * P[g][t]
            for g in self.gen_ids for t in T
        ) + pulp.lpSum(
            float(self.bat_by_id[b].get("aging_cost", 0.0)) * P[b][t]
            for b in self.bat_ids for t in T
        )
        # Revenue model: full sell at day-ahead price + bonus at realtime
        # price for the portion above commit, less penalty for shortfall.
        # We cap rt_factor at 1.0 in the planner's coefficient — otherwise
        # the LP is unbounded when rt > p_da (s_over has no upper bound).
        # Realised revenue computed at execution uses the true rt_factor.
        f3 = pulp.lpSum(
            -self.price_arr[t] * sell[t]
            + self.price_arr[t] * (1.0 - min(self.rt_factors[t], 1.0))
              * s_over[t]
            + self.cancel_rate * self.price_arr[t] * s_under[t]
            for t in T
        )
        prob += ALPHA * f1 + f2 + f3, "TotalCost"

        # ------- frozen-demand satisfaction --------------------------------
        for jid, t, w in frozen_demand:
            prob += pulp.lpSum(K_frozen[jid][t][i] for i in self.proc_ids) == w, \
                    f"fdem_{jid}_{t}"

        # ------- new-job execution / demand --------------------------------
        for j in new_jobs:
            jid = j["id"]
            if jid not in x:
                continue
            e = j["e"]
            if j["kind"] == "aperiodic":
                prob += pulp.lpSum(x[jid].values()) == e * (1 - miss[jid]), \
                        f"jE_{jid}"
            else:
                prob += pulp.lpSum(x[jid].values()) == e, f"jE_{jid}"
            if j["preempt"] == 0:
                if j["kind"] == "aperiodic":
                    prob += pulp.lpSum(y[jid].values()) == 1 - miss[jid], \
                            f"oneSt_{jid}"
                else:
                    prob += pulp.lpSum(y[jid].values()) == 1, f"oneSt_{jid}"
                for t in x[jid]:
                    cov = [s for s in y[jid] if s <= t <= s + e - 1]
                    prob += x[jid][t] == pulp.lpSum(y[jid][s] for s in cov), \
                            f"lnk_{jid}_{t}"
            for t in x[jid]:
                prob += pulp.lpSum(K_new[jid][i][t] for i in self.proc_ids) \
                        == j["w"] * x[jid][t], f"dem_{jid}_{t}"

        # ------- battery charging demand ----------------------------------
        for cj in self.chg_jobs:
            cid = cj["id"]
            b = cj["battery"]
            for t in T:
                prob += pulp.lpSum(Kchg[cid][i][t] for i in Kchg[cid]) \
                        == chg[b][t], f"cdem_{cid}_{t}"

        # ------- generators ------------------------------------------------
        for g in self.gen_ids:
            gd = self.gen_by_id[g]
            Pmin, Pmax = gd["output_min"], gd["output_max"]
            ru, rd = gd["ramp_up_rate"], gd["ramp_down_rate"]
            UT, DT = gd["min_up_time"], gd["min_down_time"]
            u_init = 1 if self.gen_state[g]["on"] else 0
            P_init = self.gen_state[g]["last_P"]
            on_done = self.gen_state[g]["on_h"]
            off_done = self.gen_state[g]["off_h"]

            for idx, t in enumerate(T):
                prob += P[g][t] >= Pmin * u[g][t], f"pmin_{g}_{t}"
                prob += P[g][t] <= (Pmax - RESERVE_PER_GEN) * u[g][t], \
                        f"pmax_{g}_{t}"
                u_prev = u[g][T[idx - 1]] if idx > 0 else u_init
                P_prev = P[g][T[idx - 1]] if idx > 0 else P_init
                prob += z_on[g][t] - z_off[g][t] == u[g][t] - u_prev, \
                        f"sw_{g}_{t}"
                prob += z_on[g][t] + z_off[g][t] <= 1, f"swex_{g}_{t}"
                prob += P[g][t] - P_prev <= ru, f"ru_{g}_{t}"
                prob += P_prev - P[g][t] <= rd, f"rd_{g}_{t}"

            # min up / down time (within window)
            for idx, t in enumerate(T):
                for s in T[idx: idx + UT]:
                    prob += u[g][s] >= z_on[g][t], f"ut_{g}_{t}_{s}"
                for s in T[idx: idx + DT]:
                    prob += 1 - u[g][s] >= z_off[g][t], f"dt_{g}_{t}_{s}"

            # Honor remaining initial-state restrictions
            if u_init == 1 and on_done < UT:
                rem = UT - on_done
                for t in T[: rem]:
                    prob += u[g][t] == 1, f"finit_on_{g}_{t}"
            if u_init == 0 and off_done < DT:
                rem = DT - off_done
                for t in T[: rem]:
                    prob += u[g][t] == 0, f"finit_off_{g}_{t}"

        # ------- renewables ------------------------------------------------
        # First hour: pv_actual is revealed; remaining hours: pv_forecast·0.92
        for pv in self.pv_ids:
            cap = self.pv_by_id[pv]["capacity"]
            for t in T:
                if t == t_now:
                    bound = cap * self.pv_actual[pv][t]
                else:
                    bound = cap * self.pv_forecast[pv][t] * (1 - self.err_std)
                prob += P[pv][t] <= bound, f"pvmx_{pv}_{t}"

        # ------- batteries -------------------------------------------------
        for b in self.bat_ids:
            bd = self.bat_by_id[b]
            soc_min_b, soc_max_b = bd["soc_min"], bd["soc_max"]
            chg_max, dis_max = bd["charge_max"], bd["discharge_max"]
            eta_c = float(bd.get("charge_efficiency", 1.0))
            eta_d = float(bd.get("discharge_efficiency", 1.0))
            sigma = float(bd.get("self_discharge_rate", 0.0))
            soc_now = self.soc[b]
            sfrac_now = min(1.0, soc_now / (0.3 * soc_max_b))

            for idx, t in enumerate(T):
                prob += chg[b][t] <= chg_max * v_chg[b][t], f"cmx_{b}_{t}"
                prob += dis[b][t] <= dis_max * (1 - v_chg[b][t]), \
                        f"dmx_{b}_{t}"
                prob += soc[b][t] >= soc_min_b, f"smn_{b}_{t}"
                prob += soc[b][t] <= soc_max_b, f"smx_{b}_{t}"
                prev = soc[b][T[idx - 1]] if idx > 0 else soc_now
                prob += (soc[b][t] == prev * (1 - sigma)
                         + chg[b][t] * eta_c - dis[b][t] / eta_d), \
                        f"sdyn_{b}_{t}"
                prob += sfrac[b][t] * (0.3 * soc_max_b) <= soc[b][t], \
                        f"sfub_{b}_{t}"
                prev_sf = sfrac[b][T[idx - 1]] if idx > 0 else sfrac_now
                prob += dis[b][t] <= dis_max * prev_sf, f"sdep_{b}_{t}"
                prob += P[b][t] == dis[b][t], f"pdis_{b}_{t}"

        # ------- energy balance per processor ------------------------------
        for i in self.proc_ids:
            for t in T:
                to_fjobs = pulp.lpSum(
                    K_frozen[jid][t][i]
                    for jid in K_frozen if t in K_frozen[jid]
                )
                to_njobs = pulp.lpSum(
                    K_new[jid][i][t] for jid in K_new if t in K_new[jid][i]
                )
                to_chg = pulp.lpSum(
                    Kchg[cj["id"]][i][t]
                    for cj in self.chg_jobs if i in Kchg[cj["id"]]
                )
                prob += P[i][t] == to_fjobs + to_njobs + to_chg \
                        + sell_share[i][t], f"bal_{i}_{t}"

        # ------- sell composition + over/under linearisation ---------------
        for t in T:
            prob += sell[t] == pulp.lpSum(sell_share[i][t]
                                          for i in self.proc_ids), \
                    f"sells_{t}"
            c_t = self.locked_commit.get(t, 0.0)
            prob += s_over[t] >= sell[t] - c_t, f"sov_{t}"
            prob += s_under[t] >= c_t - sell[t], f"sund_{t}"

        # ------- solve ----------------------------------------------------
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=self.REPLAN_TIME_LIMIT)
        prob.solve(solver)
        status = pulp.LpStatus[prob.status]
        if prob.status != pulp.LpStatusOptimal:
            print(f"  [replan t={t_now}] WARNING: status={status} — keeping current plan")
            return False

        def _v(expr, default=0.0):
            if isinstance(expr, (int, float)):
                return float(expr)
            x = pulp.value(expr)
            return float(x) if x is not None else default

        # ------- write plan in window -------------------------------------
        sporadic_placed = set()
        aperiodic_resolved = set()
        for t in T:
            slot = self.plan[t - 1]
            new_P = {}
            for i in self.proc_ids:
                pv_ = _v(P[i][t])
                if pv_ > EPS:
                    new_P[i] = round(pv_, 4)
            slot["P"] = new_P

            new_k = {}
            for jid in K_frozen:
                if t in K_frozen[jid]:
                    entry = {}
                    for i in self.proc_ids:
                        kv = _v(K_frozen[jid][t][i])
                        if kv > EPS:
                            entry[i] = round(kv, 4)
                    if entry:
                        new_k[jid] = entry
            for jid in K_new:
                if t not in K_new[jid][self.proc_ids[0]]:
                    continue
                if _v(x[jid][t]) <= 0.5:
                    continue
                entry = {}
                for i in self.proc_ids:
                    kv = _v(K_new[jid][i][t])
                    if kv > EPS:
                        entry[i] = round(kv, 4)
                if entry:
                    new_k[jid] = entry
                # Detect category
                if any(jid == r["job_id"] for r in newly_accepted_sporadic):
                    sporadic_placed.add(jid)
                else:
                    aperiodic_resolved.add(jid)
            for cj in self.chg_jobs:
                cid = cj["id"]
                if _v(chg[cj["battery"]][t]) > EPS:
                    entry = {}
                    for i in Kchg[cid]:
                        kv = _v(Kchg[cid][i][t])
                        if kv > EPS:
                            entry[i] = round(kv, 4)
                    if entry:
                        new_k[cid] = entry
            slot["k"] = new_k
            slot["sell"] = round(_v(sell[t]), 4)
            slot["soc"] = {b: round(_v(soc[b][t]), 4) for b in self.bat_ids}

        # Remove placed aperiodic from queue; record their placement
        for jid in aperiodic_resolved:
            self.aperiodic_queue = [aj for aj in self.aperiodic_queue
                                    if aj["id"] != jid]
            # Build placement record
            hours = []
            for t in T:
                if jid in self.plan[t - 1]["k"]:
                    hours.append(t)
            if hours:
                d_orig = next(j["deadline"] for j in self.aperiodic_jobs
                              if j["id"] == jid)
                e_orig = next(j["e"] for j in self.aperiodic_jobs
                              if j["id"] == jid)
                w_orig = next(j["w"] for j in self.aperiodic_jobs
                              if j["id"] == jid)
                completion = max(hours)
                tardiness = max(0, completion - d_orig)
                self.aperiodic_placements[jid] = {
                    "job_id": jid,
                    "decision": ("scheduled_on_time" if tardiness == 0
                                 else "scheduled_late"),
                    "release": next(j["release"] for j in self.aperiodic_jobs
                                    if j["id"] == jid),
                    "soft_deadline": d_orig,
                    "e": e_orig, "w": w_orig,
                    "slots": hours, "completion": completion,
                    "tardiness": tardiness,
                    "missed_soft_deadline": (tardiness > 0),
                    "via_sell_borrow": False,
                }

        # Aperiodic that were "missed" by the ILP — flag and pull from queue
        for jid, mvar in miss.items():
            if _v(mvar) > 0.5:
                # job not placed; only flag if it was an aperiodic in queue
                j_obj = next((j for j in self.aperiodic_jobs
                              if j["id"] == jid), None)
                if j_obj and jid not in self.aperiodic_placements:
                    # leave in queue — final sweep may rescue it
                    pass

        return True

    # -- execute one hour ---------------------------------------------------
    def _execute_hour(self, t):
        """Realise the plan for hour t, updating state and recording penalty."""
        slot = copy.deepcopy(self.plan[t - 1])

        # Cap PV at actual availability; any shortfall reduces sell first.
        shortfall = 0.0
        for pv in self.pv_ids:
            planned = slot["P"].get(pv, 0.0)
            cap = self.pv_by_id[pv]["capacity"]
            avail = cap * self.pv_actual[pv][t]
            if planned > avail + EPS:
                shortfall += planned - avail
                slot["P"][pv] = round(avail, 4)
            elif planned < EPS:
                slot["P"].pop(pv, None)
        if shortfall > EPS:
            actual_supply = sum(slot["P"].values())
            actual_demand = sum(sum(alloc.values()) for alloc in slot["k"].values())
            current_sell  = slot.get("sell", 0.0)
            imbalance = actual_demand + current_sell - actual_supply
            if imbalance > EPS:
                reduction = min(imbalance, current_sell)
                slot["sell"] = round(current_sell - reduction, 4)
                if imbalance - reduction > 0.01:
                    print(f"  WARNING t={t}: unresolved energy shortfall "
                          f"{imbalance - reduction:.4f} MWh after PV capping")

        # Cancellation penalty against the locked day-ahead commitment.
        commit_t = self.locked_commit.get(t, 0.0)
        realised_sell = slot.get("sell", 0.0)
        if commit_t > realised_sell + EPS:
            pen = self.cancel_rate * self.price_arr[t] * (commit_t - realised_sell)
        else:
            pen = 0.0
        slot["day_ahead_commit"] = round(commit_t, 4)
        slot["cancellation_penalty"] = round(pen, 4)
        self.total_penalty += pen

        # SOC update (L2 dynamics) — uses what was actually charged/discharged
        for b in self.bat_ids:
            bd = self.bat_by_id[b]
            eta_c = float(bd.get("charge_efficiency", 1.0))
            eta_d = float(bd.get("discharge_efficiency", 1.0))
            sigma = float(bd.get("self_discharge_rate", 0.0))
            dis_b = slot["P"].get(b, 0.0)
            chg_b = sum(slot["k"].get(f"{b}_chg", {}).values())
            new_soc = (self.soc[b] * (1 - sigma)
                       + chg_b * eta_c - dis_b / eta_d)
            self.soc[b] = max(bd["soc_min"], min(bd["soc_max"], new_soc))
            slot["soc"][b] = round(self.soc[b], 4)

        # Generator state update
        for g in self.gen_ids:
            p_val = slot["P"].get(g, 0.0)
            is_on = p_val > EPS
            st = self.gen_state[g]
            if is_on and not st["on"]:
                st["on_h"] = 1
                st["off_h"] = 0
            elif not is_on and st["on"]:
                st["off_h"] = 1
                st["on_h"] = 0
            elif is_on:
                st["on_h"] += 1
            else:
                st["off_h"] += 1
            st["on"] = is_on
            st["last_P"] = p_val

        slot["pv_forecast"] = {pv: round(self.pv_forecast[pv][t], 4)
                               for pv in self.pv_ids}
        slot["pv_actual"] = {pv: round(self.pv_actual[pv][t], 4)
                             for pv in self.pv_ids}
        self.actual_schedule.append(slot)

    # -- final aperiodic sweep ---------------------------------------------
    def _final_aperiodic_sweep(self):
        """Best-effort placement for aperiodic jobs that never got placed.

        Uses raw slack on the realised schedule (no replan). Mirrors the
        static phase3 logic at a smaller scale.
        """
        for aj in self.aperiodic_queue:
            jid = aj["id"]
            if jid in self.aperiodic_placements:
                continue
            r, d, e, w = aj["release"], aj["deadline"], aj["e"], aj["w"]

            def _slack(tt):
                slot = self.actual_schedule[tt - 1]
                s = slot.get("sell", 0.0)
                for g in self.gen_ids:
                    p_curr = slot["P"].get(g, 0.0)
                    if p_curr <= EPS:
                        continue
                    gd = self.gen_by_id[g]
                    p_prev = (self.actual_schedule[tt - 2]["P"].get(g, 0.0)
                              if tt > 1 else 0.0)
                    p_next = (self.actual_schedule[tt]["P"].get(g, 0.0)
                              if tt < len(self.actual_schedule) else p_curr)
                    max_new = min(float(gd["output_max"]),
                                  p_prev + gd["ramp_up_rate"],
                                  p_next + gd["ramp_down_rate"])
                    s += max(0.0, max_new - p_curr)
                for pv in self.pv_ids:
                    avail = (self.pv_by_id[pv]["capacity"]
                             * self.pv_actual[pv][tt])
                    s += max(0.0, avail - slot["P"].get(pv, 0.0))
                return s

            feasible = [tt for tt in range(max(r, 1), H + 1)
                        if _slack(tt) >= w - EPS]
            if len(feasible) >= e:
                slots = feasible[:e]
                # Apply: reduce sell, attribute to job
                for tt in slots:
                    slot = self.actual_schedule[tt - 1]
                    use = min(w, slot.get("sell", 0.0))
                    slot["sell"] = round(slot.get("sell", 0.0) - use, 4)
                    # any residual would mean slack came from gen/PV ramp;
                    # we don't reshape P here, treat as off-balance correction
                completion = slots[-1]
                tardy = max(0, completion - d)
                self.aperiodic_placements[jid] = {
                    "job_id": jid,
                    "decision": ("scheduled_on_time" if tardy == 0
                                 else "scheduled_late"),
                    "release": r, "soft_deadline": d, "e": e, "w": w,
                    "slots": slots, "completion": completion,
                    "tardiness": tardy,
                    "missed_soft_deadline": (tardy > 0),
                    "via_sell_borrow": True,
                }
            else:
                self.aperiodic_placements[jid] = {
                    "job_id": jid, "decision": "skipped",
                    "release": r, "soft_deadline": d, "e": e, "w": w,
                    "slots": [], "completion": None, "tardiness": None,
                    "missed_soft_deadline": True,
                    "via_sell_borrow": False,
                    "reason": f"no slack window of {e}h with >={w}MW in [{r},{H}]",
                }
        self.aperiodic_queue.clear()

    # -- export ------------------------------------------------------------
    def export_results(self, static_eval_path=None):
        # 1) schedule_result_advanced.json
        sched_path = OUTPUT_DIR / "schedule_result_advanced.json"
        with open(sched_path, "w", encoding="utf-8") as f:
            json.dump({"schedule_result": self.actual_schedule},
                      f, indent=2)
        print(f"  wrote {sched_path}")

        # 2) Build aperiodic log
        ap_log = []
        for j in self.aperiodic_jobs:
            jid = j["id"]
            if jid in self.aperiodic_placements:
                ap_log.append(self.aperiodic_placements[jid])
            else:
                ap_log.append({
                    "job_id": jid, "decision": "skipped",
                    "release": j["release"], "soft_deadline": j["deadline"],
                    "e": j["e"], "w": j["w"], "slots": [],
                    "completion": None, "tardiness": None,
                    "missed_soft_deadline": True,
                    "via_sell_borrow": False,
                    "reason": "never placed",
                })

        acc_path = OUTPUT_DIR / "acceptance_test_log_advanced.json"
        with open(acc_path, "w", encoding="utf-8") as f:
            json.dump({"acceptance_test_log": self.acceptance_log,
                       "aperiodic_log": ap_log}, f, indent=2)
        print(f"  wrote {acc_path}")

        # 3) Compute metrics
        gen_cost = 0.0
        market_rev = 0.0
        aging_cost = 0.0
        for slot in self.actual_schedule:
            t = slot["t"]
            for gid, power in slot["P"].items():
                if gid in self.gen_by_id and power > EPS:
                    g = self.gen_by_id[gid]
                    gen_cost += g["cost_fixed"] + g["cost_variable"] * power
            for bid in self.bat_ids:
                power = slot["P"].get(bid, 0.0)
                aging_cost += float(self.bat_by_id[bid]
                                    .get("aging_cost", 0.0)) * power
            # Realised market revenue (Assumption III):
            #   committed portion paid at day-ahead price,
            #   overage paid at realtime price = day-ahead * rt_factor.
            sell_t = slot.get("sell", 0.0)
            commit_t = self.locked_commit.get(t, 0.0)
            committed_sold = min(sell_t, commit_t)
            overage = max(0.0, sell_t - commit_t)
            p_da = self.price_arr[t]
            p_rt = p_da * self.rt_factors[t]
            market_rev += p_da * committed_sold + p_rt * overage

        # Periodic + sporadic hard-deadline tracking
        # Hours per task_id from actual_schedule:
        task_hours = {}
        for slot in self.actual_schedule:
            for jid in slot["k"]:
                if jid.endswith("_chg"):
                    continue
                task_hours.setdefault(jid, []).append(slot["t"])

        periodic_instances = []
        # Re-expand to split task_id hours by instance
        task_set_pj = {}
        for j in self.periodic_jobs:
            task_set_pj.setdefault(j["task_id"], []).append(j)
        for task_id, instances in task_set_pj.items():
            hrs = sorted(task_hours.get(task_id, []))
            i = 0
            for inst in sorted(instances, key=lambda x: x["release"]):
                lo, hi = inst["release"], inst["deadline"]
                # Greedily take contiguous hours in [lo, hi]
                chunk = [h for h in hrs[i:] if lo <= h <= hi][:inst["e"]]
                i += len(chunk)
                completion = max(chunk) if chunk else None
                periodic_instances.append({
                    "task_id": task_id, "release": lo,
                    "deadline": hi, "completion": completion,
                    "e": inst["e"], "executed": len(chunk),
                })

        hard_misses = sum(
            1 for inst in periodic_instances
            if inst["completion"] is None or inst["completion"] > inst["deadline"]
        )

        # Sporadic completed
        accepted = [r for r in self.acceptance_log if r["decision"] == "accept"]
        sporadic_completed = []
        for rec in accepted:
            hrs = sorted(task_hours.get(rec["job_id"], []))
            comp = max(hrs) if hrs else None
            sporadic_completed.append({**rec, "completion": comp})
        sp_total_e = sum(r["e"] for r in accepted)
        sp_ontime_e = sum(r["e"] for r in sporadic_completed
                          if r["completion"] is not None
                          and r["completion"] <= r["deadline"])
        sporadic_value_rate = (sp_ontime_e / sp_total_e
                               if sp_total_e > 0 else None)

        # Aperiodic
        missed_aperiodic = sum(1 for e in ap_log
                               if e.get("missed_soft_deadline"))
        soft_miss_rate = (missed_aperiodic / len(self.aperiodic_jobs)
                          if self.aperiodic_jobs else 0.0)

        # Objective (mirrors evaluator.py)
        obj_val = ALPHA * missed_aperiodic + gen_cost - market_rev

        # vs_static — pulled from static evaluation_results.json
        vs_static = {}
        static_eval_path = static_eval_path or (OUTPUT_DIR / "evaluation_results.json")
        try:
            with open(static_eval_path, encoding="utf-8") as f:
                static_eval = json.load(f)
            stat_obj = static_eval.get("objective_value")
            stat_gc = static_eval.get("generator_cost")
            stat_mr = static_eval.get("market_revenue")
            stat_smr = static_eval.get("soft_deadline_miss_rate")
            stat_svr = static_eval.get("sporadic_value_rate")
            vs_static = {
                "objective_value_static": stat_obj,
                "objective_value_dynamic": round(obj_val, 2),
                "generator_cost_static": stat_gc,
                "generator_cost_dynamic": round(gen_cost, 2),
                "generator_cost_change": round(gen_cost - (stat_gc or 0), 2),
                "market_revenue_static": stat_mr,
                "market_revenue_dynamic": round(market_rev, 2),
                "market_revenue_change": round(market_rev - (stat_mr or 0), 2),
                "cancellation_penalty_total": round(self.total_penalty, 2),
                "soft_miss_rate_static": stat_smr,
                "soft_miss_rate_dynamic": round(soft_miss_rate, 6),
                "sporadic_value_rate_static": stat_svr,
                "sporadic_value_rate_dynamic": sporadic_value_rate,
            }
        except Exception as exc:
            vs_static = {"error_reading_static": str(exc)}

        results = {
            "hard_deadline_miss_rate": round(
                hard_misses / max(1, len(periodic_instances) + len(accepted)), 6),
            "soft_deadline_miss_rate": round(soft_miss_rate, 6),
            "generator_cost": round(gen_cost, 2),
            "market_revenue": round(market_rev, 2),
            "cancellation_penalty": round(self.total_penalty, 2),
            "net_market_revenue": round(market_rev - self.total_penalty, 2),
            "objective_value": round(obj_val, 2),
            "acceptance_test": {
                "total": len(self.acceptance_log),
                "accepted": len(accepted),
                "rejected": len(self.acceptance_log) - len(accepted),
            },
            "sporadic_value_rate": sporadic_value_rate,
            "advanced_scheduler": {
                "method": "Event-Triggered Rolling Horizon",
                "window_size": self.WINDOW,
                "replan_interval": self.REPLAN_INTERVAL,
                "deviation_threshold": self.PV_DEV_THRESH,
                "total_replans": self.replans,
                "replan_failures": self.replan_failures,
                "replan_triggers": self.replan_triggers,
                "total_cancellation_penalty": round(self.total_penalty, 2),
                "vs_static": vs_static,
            },
        }
        eval_path = OUTPUT_DIR / "evaluation_results_advanced.json"
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"  wrote {eval_path}")
        return results


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 64)
    print("  VPP Advanced (Rolling-Horizon) Scheduler")
    print("=" * 64)
    sched = AdvancedScheduler()
    print(f"[Input] {len(sched.periodic_jobs)} periodic instances, "
          f"{len(sched.aperiodic_jobs)} aperiodic, "
          f"{len(sched.sporadic_input)} sporadic inbound")
    sched.run()
    results = sched.export_results()

    print()
    print("=" * 64)
    print("  Rolling-Horizon Simulation — summary")
    print("=" * 64)
    print(f"  Total replans                : {sched.replans}")
    print(f"  Replan triggers              : {sched.replan_triggers}")
    print(f"  Cancellation penalty         : ${sched.total_penalty:,.2f}")
    print(f"  Hard deadline miss rate      : {results['hard_deadline_miss_rate']*100:.2f}%")
    print(f"  Soft deadline miss rate      : {results['soft_deadline_miss_rate']*100:.2f}%")
    print(f"  Generator cost               : ${results['generator_cost']:,.2f}")
    print(f"  Market revenue               : ${results['market_revenue']:,.2f}")
    print(f"  Objective                    : ${results['objective_value']:,.2f}")
    vs = results["advanced_scheduler"]["vs_static"]
    if "objective_value_static" in vs:
        print()
        print("  -- vs static --")
        for k, v in vs.items():
            print(f"   {k:<35} {v}")
    print("=" * 64)


if __name__ == "__main__":
    main()
