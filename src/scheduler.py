"""
VPP Real-Time Scheduling System — 72-hour ILP-based Scheduler.

Phase 1 — Day-ahead static schedule (PuLP ILP)
  decision vars:
    P[i,t]        continuous   total output of processor i at hour t
    k[j,i,t]      continuous   power supplied by processor i to job j at t
    u[g,t]        binary       generator g on/off
    z_on/z_off    binary       generator startup/shutdown indicators
    chg/dis[b,t]  continuous   battery charge / discharge rate
    soc[b,t]      continuous   battery state of charge
    v_chg[b,t]    binary       battery charging-mode flag (mutex)
    x[j,t]        binary       job j executing at t
    y[j,s]        binary       non-preemptive start time
    sell[t]       continuous   amount sold to market
    miss[j]       binary       aperiodic miss flag

Phase 2 — Acceptance test for sporadic jobs (online, no ILP rebuild)
Phase 3 — Aperiodic queue post-processing
"""

import json
import sys
from pathlib import Path

import pulp


H = 72              # planning horizon (hours)
ALPHA = 10000       # aperiodic miss penalty ($ per miss) 巨大懲罰
EPS = 1e-6          # numerical zero-threshold for output filtering

# Day-ahead reservation strategy: each on-generator must leave this much
# headroom (MW) below its output_max, providing spinning reserve that
# Phase 2 can use to absorb sporadic arrivals without violating ramp limits.
RESERVE_PER_GEN = 5

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"


# =============================================================================
# Input loading
# =============================================================================

def load_inputs():
    proc = json.loads((INPUT_DIR / "processor_settings.json").read_text(encoding="utf-8"))
    price = json.loads((INPUT_DIR / "price_72hr.json").read_text(encoding="utf-8"))
    tasks = json.loads((OUTPUT_DIR / "task_set.json").read_text(encoding="utf-8"))

    # Demo-time override: sporadic and aperiodic jobs are provided at the
    # demo (per spec §1.1 item 3). They live in this optional file so they
    # don't get tangled with the team-generated periodic task set.
    demo_path = INPUT_DIR / "aperiodic_n_sporadic.json"
    if demo_path.exists():
        demo = json.loads(demo_path.read_text(encoding="utf-8"))
        tasks["sporadic"]  = demo.get("sporadic",  [])
        tasks["aperiodic"] = demo.get("aperiodic", [])
    return proc, price, tasks


def parse_renewable_forecast(proc):
    """Returns {pv_id: list[H+1] of fractions}, 1-indexed (index 0 unused)."""
    forecasts = {}
    for entry in proc["renewable_forecast"]:
        for pv_id, points in entry.items():
            arr = [0.0] * (H + 1)
            for v in points:
                arr[int(v["hour"])] = float(v["pv_forecast"])
            forecasts[pv_id] = arr
    return forecasts


def parse_renewable_actuals(proc):
    """Returns {pv_id: list[H+1] of actual fractions}, 1-indexed.
    Falls back to pv_forecast when pv_actual is absent (Level 1 compatibility)."""
    actuals = {}
    for entry in proc["renewable_forecast"]:
        for pv_id, points in entry.items():
            arr = [0.0] * (H + 1)
            for v in points:
                arr[int(v["hour"])] = float(v.get("pv_actual", v["pv_forecast"]))
            actuals[pv_id] = arr
    return actuals


def parse_forecast_error_std(proc):
    """Returns the forecast_error_std scalar (same for all renewables/hours)."""
    for entry in proc["renewable_forecast"]:
        for pv_id, points in entry.items():
            for v in points:
                return float(v.get("forecast_error_std", 0.0))
    return 0.0


def parse_price(price):
    """Returns list[H+1] of $/MWh, 1-indexed."""
    arr = [0.0] * (H + 1)
    for v in price["price"]:
        arr[int(v["hour"])] = float(v["market_price"])
    return arr


def parse_price_extended(price):
    """Returns (cancel_rate, rt_factors list[H+1]) from price file.
    Defaults to cancel_rate=0.0 and factors=1.0 if fields are absent."""
    cancel_rate = 0.0
    rt_factors = [1.0] * (H + 1)
    for v in price["price"]:
        t = int(v["hour"])
        rt_factors[t] = float(v.get("realtime_price_factor", 1.0))
        if cancel_rate == 0.0:
            cancel_rate = float(v.get("cancellation_penalty_rate", 0.0))
    return cancel_rate, rt_factors


# =============================================================================
# Job expansion
# =============================================================================

def expand_periodic(periodic_set):
    """Expand each periodic task into individual instances over [1..H].

    Each instance:
        id           unique instance id (e.g. "p1_k0")
        task_id      original task id ("p1") — used in output (instances of
                     the same task never overlap because d <= p)
        release      absolute release hour
        deadline     absolute deadline hour (last allowed execution hour)
        e, w, preempt, kind="periodic"
    """
    jobs = []
    for tid, t in periodic_set.items():
        r, p, e, d, w, preempt = (
            int(t["r"]), int(t["p"]), int(t["e"]),
            int(t["d"]), float(t["w"]), int(t["preempt"]),
        )
        k = 0
        while True:
            release = r + k * p
            if release > H:
                break
            deadline = min(release + d - 1, H)
            if deadline - release + 1 >= e:
                jobs.append({
                    "id":       f"{tid}_k{k}",
                    "task_id":  tid,
                    "release":  release,
                    "deadline": deadline,
                    "e":        e,
                    "w":        w,
                    "preempt":  preempt,
                    "kind":     "periodic",
                })
            k += 1
    return jobs


def expand_aperiodic(aperiodic_set):
    """Aperiodic jobs participate in ILP with a miss-penalty term."""
    jobs = []
    if not aperiodic_set:
        return jobs
    items = aperiodic_set.items() if isinstance(aperiodic_set, dict) else \
            [(t.get("id", f"a{i}"), t) for i, t in enumerate(aperiodic_set)]
    for tid, t in items:
        # New format uses {"r","d",…} (consistent with periodic);
        # older format used {"release","soft_deadline"|"deadline",…}.
        release  = int(t.get("r",  t.get("release")))
        if "d" in t:
            deadline = min(release + int(t["d"]) - 1, H)
        else:
            deadline = int(t.get("soft_deadline", t.get("deadline", H)))
        jobs.append({
            "id":       str(tid),
            "task_id":  str(tid),
            "release":  release,
            "deadline": deadline,
            "e":        int(t["e"]),
            "w":        float(t["w"]),
            "preempt":  int(t.get("preempt", 1)),
            "kind":     "aperiodic",
        })
    return jobs


# =============================================================================
# Phase 1 — Day-ahead static schedule (ILP)
# =============================================================================

def phase1_static_schedule(proc, pv_forecast, price_arr, real_jobs,
                           cancel_rate=0.0, pv_actual=None, err_std=0.0):
    generators = proc["generator"]
    renewables = proc["renewable_capacity"]
    batteries  = proc["storage"]

    gen_ids = [g["generator_id"] for g in generators]
    pv_ids  = [r["renewable_id"] for r in renewables]
    bat_ids = [b["storage_id"]   for b in batteries]
    proc_ids = gen_ids + pv_ids + bat_ids

    gen_by_id = {g["generator_id"]: g for g in generators}
    pv_by_id  = {r["renewable_id"]: r for r in renewables}
    bat_by_id = {b["storage_id"]:   b for b in batteries}

    T = list(range(1, H + 1))

    # Battery-charging "jobs" (demand sinks for battery_b_chg)
    chg_jobs = [{"id": f"{b}_chg", "battery": b} for b in bat_ids]

    prob = pulp.LpProblem("VPP_DayAhead", pulp.LpMinimize)

    # ------------------------------------------------------------------ vars
    P = pulp.LpVariable.dicts("P", (proc_ids, T), lowBound=0) # 各設備之輸出功率

    u     = pulp.LpVariable.dicts("u",     (gen_ids, T), cat="Binary") # 發電機 On/ Off
    z_on  = pulp.LpVariable.dicts("zon",   (gen_ids, T), cat="Binary") # 是否正在開機
    z_off = pulp.LpVariable.dicts("zoff",  (gen_ids, T), cat="Binary") # 是否正在關機

    chg   = pulp.LpVariable.dicts("chg",   (bat_ids, T), lowBound=0) # 電池充電率
    dis   = pulp.LpVariable.dicts("dis",   (bat_ids, T), lowBound=0) # 電池放電率
    soc   = pulp.LpVariable.dicts("soc",   (bat_ids, T), lowBound=0) # 電池電量
    v_chg   = pulp.LpVariable.dicts("vchg",  (bat_ids, T), cat="Binary") # 1: 充電模式/ 0: 放電模式或閒置
    soc_frac = pulp.LpVariable.dicts("sfrac", (bat_ids, T), lowBound=0, upBound=1) # 電池電量之佔比

    sell        = pulp.LpVariable.dicts("sell",       T, lowBound=0) # 賣給市場的總功率
    sell_share  = pulp.LpVariable.dicts("sellshare",  (proc_ids, T), lowBound=0) # 各個設備賣給市場的功率
    commit      = pulp.LpVariable.dicts("commit",     T, lowBound=0) # Phase 1 承諾賣給市場的功率
    pen         = pulp.LpVariable.dicts("pen",        T, lowBound=0) # commit - sell

    # job execution variables (only inside [release, deadline] window)
    x = {}   # x[jid][t] 任務 j 是否在時間 t 執行
    y = {}   # y[jid][s] — only for non-preemptive, 任務是否在時間 s 「開始」執行
    for j in real_jobs:
        jid = j["id"]
        x[jid] = {
            t: pulp.LpVariable(f"x_{jid}_{t}", cat="Binary")
            for t in range(j["release"], j["deadline"] + 1)
        }
        if j["preempt"] == 0:
            starts = list(range(j["release"], j["deadline"] - j["e"] + 2))
            y[jid] = {
                s: pulp.LpVariable(f"y_{jid}_{s}", cat="Binary")
                for s in starts
            }

    # aperiodic miss flags
    miss = {
        j["id"]: pulp.LpVariable(f"miss_{j['id']}", cat="Binary")
        for j in real_jobs if j["kind"] == "aperiodic"
    }

    # K[jid][i][t] — per-job allocation (only for t in window)
    K = {} # 任務 j 在時間 t 從哪一台發電機或電池 i 拿了多少電
    for j in real_jobs:
        jid = j["id"]
        K[jid] = {i: {} for i in proc_ids}
        for i in proc_ids:
            for t in x[jid]:
                K[jid][i][t] = pulp.LpVariable(
                    f"k_{jid}_{i}_{t}", lowBound=0
                )

    # Kchg[chg_id][i][t] — battery-charging allocation.
    # Constraint C21: charging energy can ONLY come from generators or
    # renewables (I_g ∪ I_r). Batteries cannot supply other batteries.
    Kchg = {}
    for cj in chg_jobs:
        cid = cj["id"]
        Kchg[cid] = {}
        for i in gen_ids + pv_ids:
            Kchg[cid][i] = {
                t: pulp.LpVariable(f"kchg_{cid}_{i}_{t}", lowBound=0)
                for t in T
            }

    # ----------------------------------------------------------- objective
    f1 = pulp.lpSum(miss.values()) if miss else 0 
    f2 = pulp.lpSum(
        gen_by_id[g]["cost_fixed"]    * u[g][t]
        + gen_by_id[g]["cost_variable"] * P[g][t]
        for g in gen_ids for t in T
    ) + pulp.lpSum(
        float(bat_by_id[b].get("aging_cost", 0.0)) * P[b][t] # Level 2
        for b in bat_ids for t in T
    )
    f3 = (-pulp.lpSum(price_arr[t] * sell[t] for t in T)
          + pulp.lpSum(cancel_rate * price_arr[t] * pen[t] for t in T)) # Level 2
    prob += ALPHA * f1 + f2 + f3, "TotalCost" 

    # ----------------------------------------------------------- job execution
    for j in real_jobs:
        jid = j["id"]
        e   = j["e"]
        total_x = pulp.lpSum(x[jid].values())
        if j["kind"] == "aperiodic":
            # missed → no execution; otherwise exactly e hours
            prob += total_x == e * (1 - miss[jid]), f"jobE_{jid}" # C4
        else:
            prob += total_x == e, f"jobE_{jid}" # C3

        # non-preemptive: pick exactly one start, x = sum of covering starts
        if j["preempt"] == 0:
            sum_y = pulp.lpSum(y[jid].values())
            if j["kind"] == "aperiodic":
                prob += sum_y == 1 - miss[jid], f"oneStart_{jid}" # C5
            else:
                prob += sum_y == 1, f"oneStart_{jid}"
            for t in x[jid]:
                covering = [s for s in y[jid] if s <= t <= s + e - 1]
                prob += x[jid][t] == pulp.lpSum(y[jid][s] for s in covering), \
                        f"link_{jid}_{t}"

        # demand: sum_i K[j,i,t] = w * x[j,t]
        for t in x[jid]:
            prob += pulp.lpSum(K[jid][i][t] for i in proc_ids) == \
                    j["w"] * x[jid][t], f"dem_{jid}_{t}"

    # battery charging demand: sum_i Kchg = chg
    for cj in chg_jobs:
        cid = cj["id"]
        b   = cj["battery"]
        for t in T:
            prob += pulp.lpSum(Kchg[cid][i][t] for i in Kchg[cid]) \
                    == chg[b][t], f"cdem_{cid}_{t}"

    # ----------------------------------------------------------- generators
    for g in gen_ids:
        gd = gen_by_id[g]
        Pmin, Pmax = gd["output_min"], gd["output_max"]
        ru,   rd   = gd["ramp_up_rate"], gd["ramp_down_rate"]
        UT,   DT   = gd["min_up_time"], gd["min_down_time"]
        u_init     = 1 if gd.get("initial_on_time", 0) > 0 else 0
        P_init     = float(gd.get("initial_energy", 0))

        for t in T:
            prob += P[g][t] >= Pmin * u[g][t], f"pmin_{g}_{t}"
            # C6 + reservation: when on, leave RESERVE_PER_GEN MW of headroom
            # for Phase 2 sporadic absorption (spinning reserve).
            prob += P[g][t] <= (Pmax - RESERVE_PER_GEN) * u[g][t], \
                    f"pmax_{g}_{t}"

            u_prev = u[g][t - 1] if t > 1 else u_init
            P_prev = P[g][t - 1] if t > 1 else P_init

            # startup / shutdown linkage
            prob += z_on[g][t] - z_off[g][t] == u[g][t] - u_prev, \
                    f"sw_{g}_{t}"
            prob += z_on[g][t] + z_off[g][t] <= 1, f"swex_{g}_{t}"

            # ramp limits apply unconditionally, including shutdown:
            # when u[t]=0 → P[t]=0, so rd constraint forces P[t-1] <= rd.
            prob += P[g][t] - P_prev <= ru, f"ru_{g}_{t}"
            prob += P_prev - P[g][t] <= rd, f"rd_{g}_{t}"

        # min up / down time
        for t in T:
            for s in range(t, min(t + UT - 1, H) + 1):
                prob += u[g][s] >= z_on[g][t], f"ut_{g}_{t}_{s}"
            for s in range(t, min(t + DT - 1, H) + 1):
                prob += 1 - u[g][s] >= z_off[g][t], f"dt_{g}_{t}_{s}"

        # C11: if generator was on at t=0 with TN < UT, force on for the
        # remaining UT - TN hours so the initial up-period completes.
        TN = int(gd.get("initial_on_time", 0))
        if u_init == 1 and TN < UT:
            for t in range(1, min(UT - TN, H) + 1):
                prob += u[g][t] == 1, f"c11_{g}_{t}"

        # C12: symmetric for off-state with TF < DT.
        TF = int(gd.get("initial_off_time", 0))
        if u_init == 0 and TF < DT:
            for t in range(1, min(DT - TF, H) + 1):
                prob += u[g][t] == 0, f"c12_{g}_{t}"

    # ----------------------------------------------------------- renewables
    # C13 with robust tightening: reduce available PV by forecast_error_std
    # as a safety margin against forecast over-estimation (Assumption I).
    robust_factor = 1.0 - err_std
    for pv in pv_ids:
        cap = pv_by_id[pv]["capacity"]
        fc  = pv_forecast[pv]
        for t in T:
            prob += P[pv][t] <= cap * fc[t] * robust_factor, f"pvmax_{pv}_{t}"

    # ----------------------------------------------------------- batteries
    for b in bat_ids:
        bd      = bat_by_id[b]
        soc_min, soc_max = bd["soc_min"],    bd["soc_max"]
        chg_max, dis_max = bd["charge_max"], bd["discharge_max"]
        soc_init         = bd["soc_init"]
        eta_c  = float(bd.get("charge_efficiency",    1.0))
        eta_d  = float(bd.get("discharge_efficiency", 1.0))
        sigma  = float(bd.get("self_discharge_rate",  0.0))
        sfrac_init = min(1.0, soc_init / (0.3 * soc_max))

        for t in T:
            prob += chg[b][t] <= chg_max * v_chg[b][t],       f"cmx_{b}_{t}"
            prob += dis[b][t] <= dis_max * (1 - v_chg[b][t]), f"dmx_{b}_{t}"
            prob += soc[b][t] >= soc_min, f"smin_{b}_{t}"
            prob += soc[b][t] <= soc_max, f"smax_{b}_{t}"
            # C16 L2: SOC dynamics with round-trip efficiency and self-discharge
            prev = soc[b][t - 1] if t > 1 else soc_init
            prob += (soc[b][t] ==
                     prev * (1 - sigma)
                     + chg[b][t] * eta_c
                     - dis[b][t] / eta_d), f"sdyn_{b}_{t}"
            # soc_frac ∈ [0,1]: upper-bounded by SOC/(0.3*soc_max)
            prob += soc_frac[b][t] * (0.3 * soc_max) <= soc[b][t], f"sfrac_ub_{b}_{t}"
            # SOC-dependent discharge limit uses previous-hour soc_frac
            prev_sfrac = soc_frac[b][t - 1] if t > 1 else sfrac_init
            prob += dis[b][t] <= dis_max * prev_sfrac, f"sdeplim_{b}_{t}"
            prob += P[b][t] == dis[b][t], f"pdis_{b}_{t}"

    # ----------------------------------------------------------- balance
    for i in proc_ids:
        for t in T:
            to_jobs = pulp.lpSum(
                K[j["id"]][i][t]
                for j in real_jobs
                if t in K[j["id"]][i]
            )
            to_chg = pulp.lpSum(
                Kchg[cj["id"]][i][t]
                for cj in chg_jobs if i in Kchg[cj["id"]]
            )
            prob += P[i][t] == to_jobs + to_chg + sell_share[i][t], \
                    f"bal_{i}_{t}"

    for t in T:
        prob += sell[t] == pulp.lpSum(sell_share[i][t] for i in proc_ids), \
                f"sellsum_{t}"

    # ---- commitment constraints (Assumption III: flexible market mechanism) --
    for t in T:
        prob += commit[t] <= sell[t],               f"commit_le_sell_{t}"
        prob += commit[t] == sell[t],               f"commit_eq_sell_{t}"  # static schedule
        prob += pen[t] >= commit[t] - sell[t],      f"pen_lb_{t}"

    # ------------------------------------------------------------ solve
    n_vars = len(prob.variables())
    n_cons = len(prob.constraints)
    print(f"[Phase 1] ILP: {n_vars} vars, {n_cons} constraints, "
          f"{len(real_jobs)} jobs", flush=True)

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=300)
    prob.solve(solver)
    status = pulp.LpStatus[prob.status]
    obj    = pulp.value(prob.objective)
    print(f"[Phase 1] status={status}  objective={obj:.2f}", flush=True)

    if prob.status not in (pulp.LpStatusOptimal,):
        print(f"WARNING: solver returned non-optimal status={status}",
              file=sys.stderr)

    # ------------------------------------------------------- extract
    def val(v, default=0.0):
        if isinstance(v, (int, float)):
            return float(v)
        x = pulp.value(v)
        return float(x) if x is not None else default

    schedule = []
    for t in T:
        _pv_act = pv_actual if pv_actual is not None else pv_forecast
        rec = {
            "t":                 t,
            "P":                 {},
            "k":                 {},
            "sell":              round(val(sell[t]), 4),
            "day_ahead_commit":  round(val(commit[t]), 4),
            "soc":               {b: round(val(soc[b][t]), 4) for b in bat_ids},
            "pv_forecast":       {pv: round(pv_forecast[pv][t], 4) for pv in pv_ids},
            "pv_actual":         {pv: round(_pv_act[pv][t], 4) for pv in pv_ids},
            "missed_aperiodic":  [],
            "rejected_sporadic": [],
        }
        for i in proc_ids:
            v = val(P[i][t])
            if v > EPS:
                rec["P"][i] = round(v, 4)

        for j in real_jobs:
            jid = j["id"]
            if t not in x[jid]:
                continue
            if val(x[jid][t]) <= 0.5:
                continue
            key = j["task_id"]
            entry = {}
            for i in proc_ids:
                kv = val(K[jid][i][t])
                if kv > EPS:
                    entry[i] = round(kv, 4)
            rec["k"][key] = entry

        for cj in chg_jobs:
            cid = cj["id"]
            if val(chg[cj["battery"]][t]) <= EPS:
                continue
            entry = {}
            for i in Kchg[cid]:
                kv = val(Kchg[cid][i][t])
                if kv > EPS:
                    entry[i] = round(kv, 4)
            rec["k"][cid] = entry

        schedule.append(rec)

    # missed aperiodic — record at each job's deadline hour
    for j in real_jobs:
        if j["kind"] != "aperiodic":
            continue
        if val(miss[j["id"]]) > 0.5:
            t_mark = min(j["deadline"], H)
            schedule[t_mark - 1]["missed_aperiodic"].append(j["task_id"])

    return schedule, obj


# =============================================================================
# Slack Absorber — shared helper for Phase 2 and Phase 3
# =============================================================================

class SlackAbsorber:
    """Computes available slack at any hour and commits demand into the
    schedule. Used by Phase 2 (sporadic acceptance) and Phase 3 (aperiodic
    queue) so both apply the same physics and bookkeeping.

    Slack sources at hour t:
      a. Current `sell` value (already-generated power going to market)
      b. On-generator spinning reserve, respecting ramp limits to t-1, t+1
         and the generator's output_max
      c. PV underutilization (capacity*forecast - current output)

    Commit precedence: peel from sell first (no P change), then ramp up
    generators, then ramp up PVs.
    """

    def __init__(self, schedule, proc, pv_forecast):
        self.schedule    = schedule
        self.pv_forecast = pv_forecast
        self.gens        = {g["generator_id"]: g for g in proc["generator"]}
        self.pvs         = {r["renewable_id"]: r for r in proc["renewable_capacity"]}
        self.gen_ids     = list(self.gens)
        self.pv_ids      = list(self.pvs)

    def slack_at(self, t):
        rec = self.schedule[t - 1]
        s = rec["sell"]
        for g in self.gen_ids:
            gd = self.gens[g]
            p_curr = rec["P"].get(g, 0.0)
            if p_curr <= EPS:
                continue
            p_prev = (self.schedule[t - 2]["P"].get(g, 0.0) if t > 1
                      else float(gd.get("initial_energy", 0)))
            p_next = (self.schedule[t]["P"].get(g, 0.0)
                      if t < len(self.schedule) else p_curr)
            max_new = min(
                float(gd["output_max"]),
                p_prev + gd["ramp_up_rate"],
                p_next + gd["ramp_down_rate"],
            )
            s += max(0.0, max_new - p_curr)
        for pv in self.pv_ids:
            cap_avail = self.pvs[pv]["capacity"] * self.pv_forecast[pv][t]
            curr = rec["P"].get(pv, 0.0)
            s += max(0.0, cap_avail - curr)
        return s

    def commit_at(self, t, w, target_jid):
        """Allocate w MW at hour t to target_jid. Returns 0 if fully committed
        or the residual (>0) if slack was overestimated."""
        rec = self.schedule[t - 1]
        remaining = w
        allocation = {}

        # Step 1: peel from sell
        if remaining > EPS and rec["sell"] >= EPS:
            from_sell = min(remaining, rec["sell"])
            rec["sell"] = round(rec["sell"] - from_sell, 4)
            # keep day_ahead_commit in sync (static mode: commit tracks actual delivery)
            if "day_ahead_commit" in rec:
                rec["day_ahead_commit"] = round(
                    max(0.0, rec["day_ahead_commit"] - from_sell), 4
                )
            already = {i: 0.0 for i in rec["P"]}
            for k_ent in rec["k"].values():
                for i, v in k_ent.items():
                    if i in already:
                        already[i] += v
            to_distribute = from_sell
            for i, p_val in sorted(rec["P"].items(), key=lambda kv: -kv[1]):
                free = p_val - already.get(i, 0)
                take = min(to_distribute, free)
                if take > EPS:
                    allocation[i] = allocation.get(i, 0) + take
                    to_distribute -= take
                if to_distribute <= EPS:
                    break
            remaining -= from_sell

        # Step 2: ramp on-generators
        if remaining > EPS:
            for g in self.gen_ids:
                if remaining <= EPS:
                    break
                gd = self.gens[g]
                p_curr = rec["P"].get(g, 0.0)
                if p_curr <= EPS:
                    continue
                p_prev = (self.schedule[t - 2]["P"].get(g, 0.0) if t > 1
                          else float(gd.get("initial_energy", 0)))
                p_next = (self.schedule[t]["P"].get(g, 0.0)
                          if t < len(self.schedule) else p_curr)
                max_new = min(
                    float(gd["output_max"]),
                    p_prev + gd["ramp_up_rate"],
                    p_next + gd["ramp_down_rate"],
                )
                avail = max_new - p_curr
                if avail <= EPS:
                    continue
                take = min(remaining, avail)
                rec["P"][g] = round(p_curr + take, 4)
                allocation[g] = round(allocation.get(g, 0) + take, 4)
                remaining -= take

        # Step 3: ramp PVs
        if remaining > EPS:
            for pv in self.pv_ids:
                if remaining <= EPS:
                    break
                cap_avail = self.pvs[pv]["capacity"] * self.pv_forecast[pv][t]
                curr = rec["P"].get(pv, 0.0)
                avail = cap_avail - curr
                if avail <= EPS:
                    continue
                take = min(remaining, avail)
                rec["P"][pv] = round(curr + take, 4)
                allocation[pv] = round(allocation.get(pv, 0) + take, 4)
                remaining -= take

        clean = {i: round(v, 4) for i, v in allocation.items() if v > EPS}
        # Merge into any existing entry for this job at this hour
        if target_jid in rec["k"]:
            for i, v in clean.items():
                rec["k"][target_jid][i] = round(
                    rec["k"][target_jid].get(i, 0) + v, 4
                )
        else:
            rec["k"][target_jid] = clean
        return remaining


# =============================================================================
# Phase 3 — Aperiodic queue
# =============================================================================

def phase3_aperiodic(schedule, aperiodic_jobs, proc, pv_forecast):
    """Soft-deadline scheduling for aperiodic jobs (EDF order).

    Two-attempt strategy per job:
      Attempt 1 — normal slack
        slack[t] = sell[t] + on-generator spinning reserve + PV under-use.
        Try to fit e hours within [release, soft_deadline]; if not possible,
        try within [release, H] (late placement).

      Attempt 2 — sell borrow (only if attempt 1 found no on-time slots)
        Effective slack[t] += min(extra_borrow[t], original_sell[t] * 0.5)
        where extra_borrow[t] = max(0, sell_before_commit - orig_sell[t] * 0.5).
        Commit pre-diverts the extra portion of sell to the job before calling
        the normal commit path for the remainder.
        Log entries set via_sell_borrow=True when this path is taken.

    Mutates `schedule` in place. Returns list of log decisions.
    """
    log = []
    if not aperiodic_jobs:
        return log

    absorber = SlackAbsorber(schedule, proc, pv_forecast)

    # Snapshot of sell values produced by Phase 1+2 (before any Phase 3 commits).
    # Used as the reference for the 50%-of-original sell-borrow cap.
    original_sell: dict[int, float] = {
        t: schedule[t - 1]["sell"] for t in range(1, H + 1)
    }

    sorted_jobs = sorted(aperiodic_jobs,
                         key=lambda j: (j["deadline"], j["release"]))

    # ------------------------------------------------------------------ helpers

    def _extra_borrow(t: int) -> float:
        """Extra MW Phase 3 may still divert from sell at hour t.

        Phase 3 is allowed to reduce sell[t] by at most 50 % of original_sell[t].
        Already borrowed = original_sell[t] - current_sell[t].
        Remaining = max(0, 50 % × original - already borrowed)
                  = max(0, current_sell[t] - original_sell[t] × 0.5).
        """
        current = schedule[t - 1]["sell"]
        orig    = original_sell[t]
        return max(0.0, current - orig * 0.5)

    def _aggressive_slack(t):
        """Replace the unrestricted sell portion with the capped portion."""
        other = absorber.slack_at(t) - schedule[t - 1]["sell"]
        return other + _extra_borrow(t)


    def _commit_aggressive(t: int, w: float, target_jid: str):
        """Commit w MW at hour t using sell borrow then normal commit_at.

        Pre-diverts min(extra_borrow, w) MW from sell to the job (maintaining
        energy balance), then delegates the remainder to commit_at.

        Returns (residual, used_borrow):
            residual   – MW not committed (> 0 only if slack was overestimated).
            used_borrow – True when the extra sell-borrow path was invoked.
        """
        rec  = schedule[t - 1]
        orig = original_sell[t]

        extra_avail = _extra_borrow(t)      # MW still borrowable from sell

        used_borrow = False
        w_remaining = w

        if extra_avail > EPS:
            # Pre-commit up to extra_avail from sell before normal commit_at.
            take_extra = min(extra_avail, w_remaining)
            sell_now   = rec["sell"]
            take_extra = min(take_extra, sell_now)  # sell cannot go below 0

            if take_extra > EPS:
                rec["sell"] = round(sell_now - take_extra, 4)
                if "day_ahead_commit" in rec:
                    rec["day_ahead_commit"] = round(
                        max(0.0, rec["day_ahead_commit"] - take_extra), 4
                    )
                used_borrow = True

                # Attribute take_extra to the job from whichever processors
                # have free capacity (P[i] − Σk[j][i] > 0 at this hour).
                already: dict[str, float] = {i: 0.0 for i in rec["P"]}
                for k_ent in rec["k"].values():
                    for i, v in k_ent.items():
                        if i in already:
                            already[i] += v

                to_dist = take_extra
                for i, p_val in sorted(rec["P"].items(), key=lambda kv: -kv[1]):
                    free = p_val - already.get(i, 0.0)
                    give = min(to_dist, free)
                    if give > EPS:
                        if target_jid in rec["k"]:
                            rec["k"][target_jid][i] = round(
                                rec["k"][target_jid].get(i, 0.0) + give, 4
                            )
                        else:
                            rec["k"][target_jid] = {i: round(give, 4)}
                        already[i] = already.get(i, 0.0) + give
                        to_dist   -= give
                    if to_dist <= EPS:
                        break

                committed_extra = take_extra - to_dist
                w_remaining    -= committed_extra

        if w_remaining <= EPS:
            return 0.0, used_borrow

        residual = absorber.commit_at(t, w_remaining, target_jid)
        return residual, used_borrow

    def _find_contiguous(r_lo, r_hi, e_len, w_need, slack_fn=None):
        """First contiguous block of e_len hours in [r_lo, r_hi] with slack ≥ w."""
        if slack_fn is None:
            slack_fn = absorber.slack_at
        for start in range(r_lo, r_hi - e_len + 2):
            block = list(range(start, start + e_len))
            if all(slack_fn(tt) >= w_need - EPS for tt in block):
                return block
        return []

    # ------------------------------------------------------------------ main loop

    for j in sorted_jobs:
        jid     = j["id"]
        r       = int(j["release"])
        d       = int(j["deadline"])    # soft deadline (absolute hour)
        e       = int(j["e"])
        w       = float(j["w"])
        preempt = int(j.get("preempt", 1))

        slots           = []
        decision        = None
        via_sell_borrow = False

        # ── Attempt 1: normal slack (sell + spinning reserve + PV) ──────────

        if preempt == 0:
            slots = _find_contiguous(r, min(d, H), e, w)
            if slots:
                decision = "scheduled_on_time"
            else:
                slots = _find_contiguous(r, H, e, w)
                if slots:
                    decision = "scheduled_late"
        else:
            on_window   = list(range(r, min(d, H) + 1))
            feasible_on = [tt for tt in on_window
                           if absorber.slack_at(tt) >= w - EPS]
            if len(feasible_on) >= e:
                slots    = feasible_on[:e]
                decision = "scheduled_on_time"
            else:
                late_window   = list(range(r, H + 1))
                feasible_late = [tt for tt in late_window
                                 if absorber.slack_at(tt) >= w - EPS]
                if len(feasible_late) >= e:
                    slots    = feasible_late[:e]
                    decision = "scheduled_late"

        # ── Attempt 2: sell borrow (up to 50 % of original sell per hour) ───
        # Only triggered when attempt 1 found no on-time placement.

        if not slots:
            if preempt == 0:
                slots = _find_contiguous(r, min(d, H), e, w, _aggressive_slack)
                if slots:
                    decision        = "scheduled_on_time"
                    via_sell_borrow = True
                else:
                    slots = _find_contiguous(r, H, e, w, _aggressive_slack)
                    if slots:
                        decision        = "scheduled_late"
                        via_sell_borrow = True
            else:
                on_window = list(range(r, min(d, H) + 1))
                feasible  = [tt for tt in on_window
                             if _aggressive_slack(tt) >= w - EPS]
                if len(feasible) >= e:
                    slots           = feasible[:e]
                    decision        = "scheduled_on_time"
                    via_sell_borrow = True
                else:
                    late_window   = list(range(r, H + 1))
                    feasible_late = [tt for tt in late_window
                                     if _aggressive_slack(tt) >= w - EPS]
                    if len(feasible_late) >= e:
                        slots           = feasible_late[:e]
                        decision        = "scheduled_late"
                        via_sell_borrow = True

        # ── Commit ──────────────────────────────────────────────────────────

        if slots:
            ok           = True
            actual_borrow = False
            for tt in slots:
                if via_sell_borrow:
                    leftover, borrowed = _commit_aggressive(tt, w, jid)
                    if borrowed:
                        actual_borrow = True
                else:
                    leftover = absorber.commit_at(tt, w, jid)
                if leftover > EPS:
                    ok = False
                    break

            if ok:
                completion = slots[-1]
                tardiness  = max(0, completion - d)
                missed     = (decision == "scheduled_late")
                log.append({
                    "job_id":               jid,
                    "decision":             decision,
                    "release":              r,
                    "soft_deadline":        d,
                    "e":                    e,
                    "w":                    w,
                    "slots":                slots,
                    "completion":           completion,
                    "tardiness":            tardiness,
                    "missed_soft_deadline": missed,
                    "via_sell_borrow":      actual_borrow,
                })
                if missed and 1 <= d <= H:
                    schedule[d - 1]["missed_aperiodic"].append(jid)
                continue

        # ── No feasible placement ────────────────────────────────────────────

        log.append({
            "job_id":               jid,
            "decision":             "skipped",
            "release":              r,
            "soft_deadline":        d,
            "e":                    e,
            "w":                    w,
            "slots":                [],
            "completion":           None,
            "tardiness":            None,
            "missed_soft_deadline": True,
            "via_sell_borrow":      False,
            "reason":               f"no slack window of {e}h with >={w}MW in [{r},{H}]",
        })
        if 1 <= d <= H:
            schedule[d - 1]["missed_aperiodic"].append(jid)

    return log


# =============================================================================
# Phase 2 — Sporadic acceptance test
# =============================================================================

def phase2_acceptance(schedule, sporadic_input, proc, pv_forecast):
    """Online acceptance test for sporadic jobs (hard deadline).

    For each arriving sporadic, compute total slack at each candidate hour;
    accept only if e feasible slots exist in [release, hard_deadline]
    (contiguous for non-preemptive). If accepted, commit via SlackAbsorber.

    Mutates `schedule` in place. Returns a list of log entries.
    """
    log = []
    if not sporadic_input:
        return log

    absorber = SlackAbsorber(schedule, proc, pv_forecast)

    items = (sporadic_input.items() if isinstance(sporadic_input, dict)
             else [(t.get("id", f"s{i}"), t) for i, t in enumerate(sporadic_input)])

    for sid, sj in items:
        sid     = str(sid)
        # New format uses {"r","d",…}; older used {"release","hard_deadline"|"deadline",…}.
        r       = int(sj.get("r", sj.get("release")))
        if "d" in sj:
            d_abs = min(r + int(sj["d"]) - 1, H)
        else:
            d_abs = int(sj.get("hard_deadline", sj.get("deadline", H)))
        e       = int(sj["e"])
        w       = float(sj["w"])
        preempt = int(sj.get("preempt", 1))

        window = list(range(r, min(d_abs, H) + 1))

        chosen = []
        reason = None
        if preempt == 0:
            for start in range(r, min(d_abs, H) - e + 2):
                block = list(range(start, start + e))
                if all(absorber.slack_at(tt) >= w - EPS for tt in block):
                    chosen = block
                    break
            if not chosen:
                reason = (f"no contiguous block of {e}h with >={w}MW slack "
                          f"in [{r},{d_abs}]")
        else:
            feasible = [tt for tt in window if absorber.slack_at(tt) >= w - EPS]
            if len(feasible) >= e:
                chosen = feasible[:e]
            else:
                reason = (f"only {len(feasible)} feasible hours, need {e} "
                          f"(window [{r},{d_abs}], w={w})")

        if chosen:
            committed = []
            for tt in chosen:
                leftover = absorber.commit_at(tt, w, sid)
                if leftover > EPS:
                    reason = f"commit failed at t={tt}, residual={leftover:.3f}"
                    committed = []
                    break
                committed.append(tt)
            if committed:
                log.append({
                    "job_id":           sid,
                    "decision":         "accept",
                    "arrival":          r,
                    "release":          r,
                    "deadline":         d_abs,
                    "e":                e,
                    "w":                w,
                    "slots":            committed,
                    "caused_violation": False,
                })
                continue

        log.append({
            "job_id":           sid,
            "decision":         "reject",
            "arrival":          r,
            "release":          r,
            "deadline":         d_abs,
            "e":                e,
            "w":                w,
            "reason":           reason or "insufficient slack",
            "caused_violation": False,
        })
        if 1 <= r <= H:
            schedule[r - 1]["rejected_sporadic"].append(sid)

    return log


# =============================================================================
# Output writing
# =============================================================================

def write_outputs(schedule, acceptance_log, aperiodic_log):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sched_path = OUTPUT_DIR / "schedule_result.json"
    with open(sched_path, "w", encoding="utf-8") as f:
        json.dump({"schedule_result": schedule}, f, indent=2)
    print(f"  wrote {sched_path}")

    log_path = OUTPUT_DIR / "acceptance_test_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "acceptance_test_log": acceptance_log,
            "aperiodic_log":       aperiodic_log,
        }, f, indent=2)
    print(f"  wrote {log_path}")


def print_summary(schedule, obj, periodic_jobs, aperiodic_jobs, sporadic_input):
    total_sell = sum(r["sell"] for r in schedule)
    on_hours = {i: 0 for r in schedule for i in r["P"]}
    for r in schedule:
        for i in r["P"]:
            on_hours[i] += 1
    print("\n" + "=" * 60)
    print("  VPP Day-Ahead Schedule — summary")
    print("=" * 60)
    print(f"  Total objective       : {obj:.2f}")
    print(f"  Total energy sold     : {total_sell:.2f} MWh")
    print(f"  Periodic instances    : {len(periodic_jobs)}")
    print(f"  Aperiodic in queue    : {len(aperiodic_jobs)}")
    print(f"  Sporadic inbound      : {len(sporadic_input)}")
    print(f"  Active hours per proc :")
    for i, h in sorted(on_hours.items(), key=lambda kv: -kv[1]):
        print(f"    {i:<14}  {h:>3} h")
    print("=" * 60)


# =============================================================================
# Main
# =============================================================================

def main():
    proc, price_data, task_set = load_inputs()
    pv_forecast              = parse_renewable_forecast(proc)
    pv_actual                = parse_renewable_actuals(proc)
    err_std                  = parse_forecast_error_std(proc)
    price_arr                = parse_price(price_data)
    cancel_rate, rt_factors  = parse_price_extended(price_data)

    periodic_jobs  = expand_periodic(task_set.get("periodic", {}))
    aperiodic_jobs = expand_aperiodic(task_set.get("aperiodic", []))
    sporadic_input = task_set.get("sporadic", [])

    # Phase 1 schedules only periodic jobs (the day-ahead static schedule).
    # Per spec assumption 6, aperiodic and sporadic jobs arrive during
    # execution; they are handled in Phase 2 (sporadic acceptance) and
    # Phase 3 (aperiodic queue) respectively.
    print(f"[Input] {len(periodic_jobs)} periodic instances, "
          f"{len(aperiodic_jobs)} aperiodic, "
          f"{len(sporadic_input)} sporadic inbound")

    schedule, obj = phase1_static_schedule(
        proc, pv_forecast, price_arr, periodic_jobs,
        cancel_rate=cancel_rate, pv_actual=pv_actual, err_std=err_std
    )

    acceptance_log = phase2_acceptance(
        schedule, sporadic_input, proc, pv_forecast
    )

    aperiodic_log = phase3_aperiodic(
        schedule, aperiodic_jobs, proc, pv_forecast
    )

    write_outputs(schedule, acceptance_log, aperiodic_log)
    print_summary(schedule, obj, periodic_jobs, aperiodic_jobs, sporadic_input)


if __name__ == "__main__":
    main()