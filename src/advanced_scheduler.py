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
    SlackAbsorber,
)

INF = float("inf")


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

        # Slack absorber operates on the rolling plan; same `self.plan`
        # reference as below, so its slack_at sees post-commit P values.
        self.absorber = SlackAbsorber(
            self.plan, self.proc, self.pv_forecast, err_std=self.err_std
        )

        # Tracking
        self.replans = 0
        self.replan_triggers = {"pv_deviation": 0, "sporadic_arrival": 0,
                                "periodic_6h": 0, "aperiodic_arrival": 0}
        self.last_periodic_replan = 0
        self.total_penalty = 0.0
        self.acceptance_log = []        # sporadic acceptance decisions
        self.aperiodic_placements = {}  # jid -> {slots, decision, ...} — populated on arrival
        self.replan_failures = 0
        self.actual_schedule = []       # what really executed each hour

    # -- main loop ----------------------------------------------------------
    def run(self):
        """Hour-by-hour rolling simulation.

        At each hour, arrivals are placed immediately (Level-1 online_phase
        semantics): sporadic via acceptance test (may reject), aperiodic via
        force-placement (must execute, per C4). The rolling-horizon replan
        then re-optimises dispatch given the now-frozen placements, fired by
        PV-actual deviation, periodic refresh interval, or fresh arrivals.
        """
        for t in range(1, H + 1):
            arriving_sp = [s for s in self.sporadic_input
                           if int(s.get("r", s.get("release", -1))) == t]
            arriving_ap = [j for j in self.aperiodic_jobs
                           if j["release"] == t
                           and j["id"] not in self.aperiodic_placements]

            # Sporadic first (hard deadline takes priority at ties).
            sp_placed = False
            for sj in arriving_sp:
                rec = self._process_sporadic(sj, t)
                self.acceptance_log.append(rec)
                if rec["decision"] == "accept":
                    sp_placed = True

            # Aperiodic — always force-placed.
            ap_placed = False
            for aj in arriving_ap:
                rec = self._process_aperiodic(aj, t)
                self.aperiodic_placements[aj["id"]] = rec
                ap_placed = True

            # Triggers for dispatch replan (placements are already committed).
            triggers = []
            if sp_placed:
                triggers.append("sporadic_arrival")
            if ap_placed:
                triggers.append("aperiodic_arrival")
            if t - self.last_periodic_replan >= self.REPLAN_INTERVAL:
                triggers.append("periodic_6h")
            if self._pv_deviation(t) > self.PV_DEV_THRESH:
                triggers.append("pv_deviation")

            if triggers:
                ok = self._rolling_replan(t)
                if ok:
                    self.replans += 1
                    for tr in triggers:
                        self.replan_triggers[tr] += 1
                    if "periodic_6h" in triggers:
                        self.last_periodic_replan = t
                else:
                    self.replan_failures += 1

            self._execute_hour(t)

    # -- PV deviation -------------------------------------------------------
    def _pv_deviation(self, t):
        max_dev = 0.0
        for pv in self.pv_ids:
            fc = self.pv_forecast[pv][t]
            ac = self.pv_actual[pv][t]
            if fc > 0.01:
                max_dev = max(max_dev, abs(ac - fc) / fc)
        return max_dev

    # -- slack / cost helpers (mirror scheduler.online_phase) ---------------
    def _normal_slack(self, t):
        """Slack from gen ramp + PV underuse only — sell intentionally
        excluded so we minimise sell borrow."""
        return self.absorber.slack_at(t) - self.plan[t - 1].get("sell", 0.0)

    def _hour_cost(self, t, w):
        """Estimated $ of sell borrow at hour t to satisfy w MWh. Returns INF
        when even 100 % sell + gen + PV ramp cannot satisfy w."""
        free = self._normal_slack(t)
        avail = free + self.plan[t - 1].get("sell", 0.0)
        if avail < w - EPS:
            return INF
        sell_needed = max(0.0, w - free)
        return sell_needed * float(self.price_arr[t])

    def _find_min_cost(self, r, end, e_len, w_need, preempt):
        """Cheapest feasible placement of e_len hours of w_need MWh in [r, end].
        Returns (sorted_slots, total_cost) or (None, INF) if infeasible."""
        if r > end or end - r + 1 < e_len:
            return None, INF
        if preempt:
            ranked = sorted(range(r, end + 1),
                            key=lambda t: self._hour_cost(t, w_need))
            chosen = ranked[:e_len]
            total = sum(self._hour_cost(t, w_need) for t in chosen)
            if total >= INF:
                return None, INF
            return sorted(chosen), total
        best_window, best_cost = None, INF
        for start in range(r, end - e_len + 2):
            window = list(range(start, start + e_len))
            total = sum(self._hour_cost(t, w_need) for t in window)
            if total < best_cost:
                best_cost, best_window = total, window
        if best_cost >= INF:
            return None, INF
        return best_window, best_cost

    def _group_consecutive(self, slots):
        """Split sorted slot list into maximal runs of consecutive hours."""
        if not slots:
            return []
        ss = sorted(slots)
        runs = [[ss[0]]]
        for s in ss[1:]:
            if s == runs[-1][-1] + 1:
                runs[-1].append(s)
            else:
                runs.append([s])
        return runs

    def _commit_min_sell(self, t, w, jid):
        """Allocate w MWh at hour t into self.plan using gen+PV ramp first,
        sell only for the residual. Returns (residual, sell_taken)."""
        rec = self.plan[t - 1]
        saved_sell = rec.get("sell", 0.0)
        rec["sell"] = 0.0
        residual = self.absorber.commit_at(t, w, jid)
        rec["sell"] = saved_sell
        if residual <= EPS:
            return 0.0, 0.0
        sell_take = min(residual, rec.get("sell", 0.0))
        if sell_take <= EPS:
            return residual, 0.0
        rec["sell"] = round(rec.get("sell", 0.0) - sell_take, 4)
        if "day_ahead_commit" in rec:
            rec["day_ahead_commit"] = round(
                max(0.0, rec["day_ahead_commit"] - sell_take), 4)
        already = {i: 0.0 for i in rec.get("P", {})}
        for k_ent in rec.get("k", {}).values():
            for i, v in k_ent.items():
                if i in already:
                    already[i] += v
        to_dist = sell_take
        for i, p_val in sorted(rec.get("P", {}).items(),
                               key=lambda kv: -kv[1]):
            free = p_val - already.get(i, 0.0)
            give = min(to_dist, free)
            if give > EPS:
                rec.setdefault("k", {}).setdefault(jid, {})
                rec["k"][jid][i] = round(rec["k"][jid].get(i, 0.0) + give, 4)
                already[i] += give
                to_dist -= give
            if to_dist <= EPS:
                break
        committed_sell = sell_take - to_dist
        return residual - committed_sell, committed_sell

    def _commit_run_min_sell(self, run, w, jid):
        """Joint-LP commit over a consecutive run; falls back to single-hour
        commit for runs of length 1. Returns total sell MWh borrowed."""
        if len(run) == 1:
            _, taken = self._commit_min_sell(run[0], w, jid)
            return taken
        T = run
        prob = pulp.LpProblem(f"joint_{jid}_{T[0]}_{T[-1]}", pulp.LpMinimize)
        dx = {g: {t: pulp.LpVariable(f"dx_{g}_{t}", lowBound=0)
                  for t in T} for g in self.gen_ids}
        dp = {pv: {t: pulp.LpVariable(f"dp_{pv}_{t}", lowBound=0)
                   for t in T} for pv in self.pv_ids}
        s_t = {t: pulp.LpVariable(f"s_{t}", lowBound=0) for t in T}
        prob += pulp.lpSum(s_t[t] * float(self.price_arr[t]) for t in T)
        for t in T:
            prob += (pulp.lpSum(dx[g][t] for g in dx) +
                     pulp.lpSum(dp[pv][t] for pv in dp) +
                     s_t[t] == w), f"dem_{t}"
            prob += s_t[t] <= self.plan[t - 1].get("sell", 0.0), f"sell_cap_{t}"
        for g in self.gen_ids:
            gd = self.gen_by_id[g]
            ru, rd = float(gd["ramp_up_rate"]), float(gd["ramp_down_rate"])
            out_max = float(gd["output_max"])
            for idx, t in enumerate(T):
                rec = self.plan[t - 1]
                p_curr = rec["P"].get(g, 0.0)
                if p_curr <= EPS:
                    prob += dx[g][t] == 0, f"goff_{g}_{t}"
                    continue
                prob += p_curr + dx[g][t] <= out_max, f"omx_{g}_{t}"
                # ramp_up at t (from t-1)
                if idx == 0:
                    p_prev = (self.plan[t - 2]["P"].get(g, 0.0) if t > 1
                              else float(gd.get("initial_energy", 0)))
                    prob += p_curr + dx[g][t] - p_prev <= ru, f"ru_{g}_{t}"
                else:
                    tp = T[idx - 1]
                    p_prev_curr = self.plan[tp - 1]["P"].get(g, 0.0)
                    prob += ((p_curr + dx[g][t]) - (p_prev_curr + dx[g][tp])
                             <= ru), f"ru_{g}_{t}"
                # ramp_down at t (toward t+1)
                if idx == len(T) - 1:
                    p_next = (self.plan[t]["P"].get(g, 0.0)
                              if t < len(self.plan) else p_curr)
                    prob += p_curr + dx[g][t] - p_next <= rd, f"rd_{g}_{t}"
                else:
                    tn = T[idx + 1]
                    p_next_curr = self.plan[tn - 1]["P"].get(g, 0.0)
                    prob += ((p_curr + dx[g][t]) - (p_next_curr + dx[g][tn])
                             <= rd), f"rd_{g}_{t}"
        for pv in self.pv_ids:
            cap = self.pv_by_id[pv]["capacity"]
            for t in T:
                rec = self.plan[t - 1]
                p_curr = rec["P"].get(pv, 0.0)
                max_pv = cap * self.pv_forecast[pv][t] * self.absorber.robust
                prob += p_curr + dp[pv][t] <= max_pv, f"pvmx_{pv}_{t}"
        solver = pulp.PULP_CBC_CMD(msg=False)
        prob.solve(solver)
        if prob.status != pulp.LpStatusOptimal:
            total = 0.0
            for tt in T:
                _, taken = self._commit_min_sell(tt, w, jid)
                total += taken
            return total

        def _val(x):
            v = pulp.value(x)
            return float(v) if v is not None else 0.0

        total_sell_borrowed = 0.0
        for t in T:
            rec = self.plan[t - 1]
            for g in self.gen_ids:
                inc = _val(dx[g][t])
                if inc <= EPS:
                    continue
                rec["P"][g] = round(rec["P"].get(g, 0.0) + inc, 4)
                rec.setdefault("k", {}).setdefault(jid, {})
                rec["k"][jid][g] = round(rec["k"][jid].get(g, 0.0) + inc, 4)
            for pv in self.pv_ids:
                inc = _val(dp[pv][t])
                if inc <= EPS:
                    continue
                rec["P"][pv] = round(rec["P"].get(pv, 0.0) + inc, 4)
                rec.setdefault("k", {}).setdefault(jid, {})
                rec["k"][jid][pv] = round(rec["k"][jid].get(pv, 0.0) + inc, 4)
            sell_take = _val(s_t[t])
            if sell_take <= EPS:
                continue
            total_sell_borrowed += sell_take
            rec["sell"] = round(rec.get("sell", 0.0) - sell_take, 4)
            if "day_ahead_commit" in rec:
                rec["day_ahead_commit"] = round(
                    max(0.0, rec["day_ahead_commit"] - sell_take), 4)
            already = {i: 0.0 for i in rec.get("P", {})}
            for k_ent in rec.get("k", {}).values():
                for i, v in k_ent.items():
                    if i in already:
                        already[i] += v
            to_dist = sell_take
            for i, p_val in sorted(rec.get("P", {}).items(),
                                   key=lambda kv: -kv[1]):
                free = p_val - already.get(i, 0.0)
                give = min(to_dist, free)
                if give > EPS:
                    rec.setdefault("k", {}).setdefault(jid, {})
                    rec["k"][jid][i] = round(
                        rec["k"][jid].get(i, 0.0) + give, 4)
                    already[i] += give
                    to_dist -= give
                if to_dist <= EPS:
                    break
        return total_sell_borrowed

    # -- sporadic / aperiodic placement (mirrors Level 1 online_phase) ------
    def _process_sporadic(self, sj, t_now):
        """Online acceptance test: cheapest e-hour placement in
        [max(release, t_now), hard_deadline] using gen+PV+sell. Accept and
        commit immediately if feasible; otherwise reject."""
        sid = str(sj["id"])
        r = int(sj.get("r", sj.get("release")))
        if "d" in sj:
            d = min(r + int(sj["d"]) - 1, H)
        else:
            d = int(sj.get("hard_deadline", sj.get("deadline", H)))
        e = int(sj["e"])
        w = float(sj["w"])
        preempt = int(sj.get("preempt", 1))

        slots, cost = self._find_min_cost(max(r, t_now), min(d, H),
                                          e, w, preempt)
        if slots is None:
            return {"job_id": sid, "decision": "reject",
                    "arrival": r, "release": r, "deadline": d,
                    "e": e, "w": w,
                    "reason": (f"no feasible {e}h placement of {w}MW "
                               f"in [{max(r, t_now)},{d}] even at 100% sell"),
                    "caused_violation": False}

        sell_borrowed = 0.0
        for run in self._group_consecutive(slots):
            sell_borrowed += self._commit_run_min_sell(run, w, sid)
        return {"job_id": sid, "decision": "accept",
                "arrival": r, "release": r, "deadline": d,
                "e": e, "w": w,
                "slots": slots, "preempt": preempt,
                "sell_borrowed": round(sell_borrowed, 4),
                "cost_estimate": round(cost, 2),
                "caused_violation": False}

    def _process_aperiodic(self, aj, t_now):
        """Force-place an aperiodic job per spec C4 (must execute e hours by
        H). Prefer cheapest on-time placement; fall back to cheapest late
        placement in [release, H] only if on-time infeasible."""
        jid = aj["id"]
        r = max(aj["release"], t_now)
        d_soft = aj["deadline"]
        e = aj["e"]
        w = aj["w"]
        preempt = aj.get("preempt", 1)

        slots, cost = self._find_min_cost(r, min(d_soft, H), e, w, preempt)
        on_time = slots is not None
        if not on_time:
            slots, cost = self._find_min_cost(r, H, e, w, preempt)

        if slots is None:
            return {"job_id": jid, "decision": "infeasible",
                    "release": aj["release"], "soft_deadline": d_soft,
                    "e": e, "w": w, "slots": [],
                    "completion": None, "tardiness": None,
                    "missed_soft_deadline": True,
                    "via_sell_borrow": False,
                    "reason": (f"no feasible {e}h placement of {w}MW "
                               f"in [{r},{H}] even at 100% sell")}

        sell_borrowed = 0.0
        for run in self._group_consecutive(slots):
            sell_borrowed += self._commit_run_min_sell(run, w, jid)
        completion = max(slots)
        tardiness = max(0, completion - d_soft)
        decision = "scheduled_on_time" if on_time else "scheduled_late"
        return {"job_id": jid, "decision": decision,
                "release": aj["release"], "soft_deadline": d_soft,
                "e": e, "w": w, "slots": slots,
                "completion": completion, "tardiness": tardiness,
                "missed_soft_deadline": (not on_time),
                "via_sell_borrow": (sell_borrowed > EPS),
                "sell_borrowed": round(sell_borrowed, 4),
                "cost_estimate": round(cost, 2)}

    # -- rolling replan -----------------------------------------------------
    def _rolling_replan(self, t_now):
        """Re-optimise dispatch over [t_now, t_end].

        All sporadic / aperiodic placements are committed at arrival time
        (Level-1 semantics), so this replan only redecides:
          - thermal P, on/off, ramp;
          - battery chg / dis / soc;
          - renewable P (bounded by pv_actual for t_now, pv_forecast·robust else);
          - sell[t] and (penalty / over-revenue) decomposition;
          - per-(job, hour) processor split K[jid][i][t] (under the FROZEN
            demand w[jid][t] established at commit time).
        Returns True iff solve succeeded and the plan was updated.
        """
        t_end = min(t_now + self.WINDOW - 1, H)
        T = list(range(t_now, t_end + 1))

        # ------- collect frozen demand (every committed job in the window) --
        # All non-charging jobs in the current plan are treated as frozen —
        # no new-job placement happens inside the replan anymore.
        frozen_demand = []     # list of (jid, t, w)
        for slot in self.plan:
            if slot["t"] not in T:
                continue
            for jid, alloc in slot["k"].items():
                if jid.endswith("_chg"):
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

        # Kchg[bat_chg][i][t]
        Kchg = {}
        for cj in self.chg_jobs:
            cid = cj["id"]
            Kchg[cid] = {}
            for i in self.gen_ids + self.pv_ids:
                Kchg[cid][i] = {t: pulp.LpVariable(
                    f"kc_{cid}_{i}_{t}", lowBound=0) for t in T}

        # ------- objective -------------------------------------------------
        # Sporadic/aperiodic placements are frozen — no miss variables here,
        # so f1 = 0 in the replan. f2 and f3 still drive dispatch optimisation.
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
        # rt_factor coefficient capped at 1.0 to keep the LP bounded; realised
        # revenue at execution uses the true rt_factor.
        f3 = pulp.lpSum(
            -self.price_arr[t] * sell[t]
            + self.price_arr[t] * (1.0 - min(self.rt_factors[t], 1.0))
              * s_over[t]
            + self.cancel_rate * self.price_arr[t] * s_under[t]
            for t in T
        )
        prob += f2 + f3, "TotalCost"

        # ------- frozen-demand satisfaction --------------------------------
        for jid, t, w in frozen_demand:
            prob += pulp.lpSum(K_frozen[jid][t][i] for i in self.proc_ids) == w, \
                    f"fdem_{jid}_{t}"

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
                to_chg = pulp.lpSum(
                    Kchg[cj["id"]][i][t]
                    for cj in self.chg_jobs if i in Kchg[cj["id"]]
                )
                prob += P[i][t] == to_fjobs + to_chg \
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
        # Job placements are frozen — only dispatch numbers (P, k splits,
        # sell, soc, battery k) are refreshed from the LP solution.
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
        sp_total_e = sum(r["e"] for r in self.acceptance_log)
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
