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
    soc_frac = pulp.LpVariable.dicts("sfrac", (bat_ids, T), lowBound=0, upBound=1) # soc_init / (0.3 * soc_max)

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

        # non-preemptive: pick exactly one start, x = sum of covering starts (C5)
        if j["preempt"] == 0:
            sum_y = pulp.lpSum(y[jid].values())
            if j["kind"] == "aperiodic":
                prob += sum_y == 1 - miss[jid], f"oneStart_{jid}" 
            else:
                prob += sum_y == 1, f"oneStart_{jid}"
            for t in x[jid]: # s: 啟動時間, t: 當下時間, e: 執行時間
                covering = [s for s in y[jid] if s <= t <= s + e - 1]
                prob += x[jid][t] == pulp.lpSum(y[jid][s] for s in covering), \
                        f"link_{jid}_{t}"

        # demand: sum_i K[j,i,t] = w * x[j,t] (C1)
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
            prob += P[g][t] - P_prev <= ru, f"ru_{g}_{t}" # C7
            prob += P_prev - P[g][t] <= rd, f"rd_{g}_{t}" # C7

        # min up / down time
        for t in T:
            for s in range(t, min(t + UT - 1, H) + 1):
                prob += u[g][s] >= z_on[g][t], f"ut_{g}_{t}_{s}" # C9
            for s in range(t, min(t + DT - 1, H) + 1):
                prob += 1 - u[g][s] >= z_off[g][t], f"dt_{g}_{t}_{s}" # C10

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
            prob += chg[b][t] <= chg_max * v_chg[b][t],       f"cmx_{b}_{t}" # C15
            prob += dis[b][t] <= dis_max * (1 - v_chg[b][t]), f"dmx_{b}_{t}" # C14
            prob += soc[b][t] >= soc_min, f"smin_{b}_{t}" # C17
            prob += soc[b][t] <= soc_max, f"smax_{b}_{t}" # C17
            # C16 L2: SOC dynamics with round-trip efficiency and self-discharge
            prev = soc[b][t - 1] if t > 1 else soc_init
            prob += (soc[b][t] ==
                     prev * (1 - sigma) # self-discharge
                     + chg[b][t] * eta_c
                     - dis[b][t] / eta_d), f"sdyn_{b}_{t}"
            # soc_frac ∈ [0,1]: upper-bounded by SOC/(0.3*soc_max) 電池電量低於 30% 時，放電受限
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
                    f"bal_{i}_{t}" # C23

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

    def __init__(self, schedule, proc, pv_forecast, err_std=0.0):
        self.schedule    = schedule
        self.pv_forecast = pv_forecast
        self.robust      = 1.0 - float(err_std)   # robust PV bound coefficient
        self.gens        = {g["generator_id"]: g for g in proc["generator"]}
        self.pvs         = {r["renewable_id"]: r for r in proc["renewable_capacity"]}
        self.gen_ids     = list(self.gens)
        self.pv_ids      = list(self.pvs)

    def slack_at(self, t):
        rec = self.schedule[t - 1]
        s = rec["sell"] # 賣出的
        for g in self.gen_ids:
            gd = self.gens[g]
            p_curr = rec["P"].get(g, 0.0)
            if p_curr <= EPS: # 關機，跳過
                continue
            p_prev = (self.schedule[t - 2]["P"].get(g, 0.0) if t > 1
                      else float(gd.get("initial_energy", 0))) # 前一小時的輸出
            # ramp_down boundary removed in slack_at() intentionally: this
            # function is used for feasibility screening only. The actual
            # commit is done by commit_at() (which retains ramp_down guard)
            # or _commit_run_min_sell LP (which enforces ramp across the run).
            # Removing it here allows _find_min_cost to consider hours that
            # are genuinely feasible when handled by the multi-hour LP path.
            max_new = min(
                float(gd["output_max"]),
                p_prev + gd["ramp_up_rate"],
            )
            s += max(0.0, max_new - p_curr) # 發電機的剩餘可用產能
        for pv in self.pv_ids:
            cap_avail = self.pvs[pv]["capacity"] * self.pv_forecast[pv][t] * self.robust
            curr = rec["P"].get(pv, 0.0)
            s += max(0.0, cap_avail - curr) # 沒用到的 Renewable (err_std)
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
                # ramp_down bound removed: _cascade_sell propagates any raise
                # to subsequent ON-hours so ramp_down feasibility is maintained
                # there. Raises blocked by an imminent OFF hour are physically
                # irrecoverable; _cascade_sell stops at the OFF hour boundary.
                max_new = min(
                    float(gd["output_max"]),
                    p_prev + gd["ramp_up_rate"],
                )
                avail = max_new - p_curr
                if avail <= EPS:
                    continue
                take = min(remaining, avail)
                rec["P"][g] = round(p_curr + take, 4)
                allocation[g] = round(allocation.get(g, 0) + take, 4)
                remaining -= take
                self._cascade_sell(g, t, rec["P"][g])

        # Step 3: ramp PVs
        if remaining > EPS:
            for pv in self.pv_ids:
                if remaining <= EPS:
                    break
                cap_avail = self.pvs[pv]["capacity"] * self.pv_forecast[pv][t] * self.robust
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

    def _cascade_sell(self, g, t_raised, new_p):
        """After raising P[g, t_raised] to new_p, propagate the increase
        to subsequent hours via sell to maintain ramp_down feasibility.
        When a scheduled-OFF hour is reached the generator is turned ON
        (output attributed to sell) so the ramp_down chain can continue.
        Stops when the ramp_down requirement drops to zero or is already met."""
        gd = self.gens[g]
        rd = float(gd["ramp_down_rate"])
        out_max = float(gd["output_max"])
        out_min = float(gd.get("output_min", 0.0))
        t = t_raised + 1
        p_prev = new_p
        while t <= H:
            min_needed = p_prev - rd
            if min_needed <= EPS:
                break  # ramp_down satisfied — cascade complete
            rec = self.schedule[t - 1]
            p_curr = rec["P"].get(g, 0.0)
            if p_curr >= min_needed - EPS:
                break  # already ramp_down-feasible at this hour
            # Raise generator; for an OFF hour enforce output_min floor
            new_p_t = min(out_max, max(out_min if p_curr <= EPS else 0.0, min_needed))
            extra = new_p_t - p_curr
            if extra <= EPS:
                break
            rec["P"][g] = round(new_p_t, 4)
            rec["sell"] = round(rec["sell"] + extra, 4)
            p_prev = new_p_t
            t += 1


# =============================================================================
# Online Phase — time-ordered sporadic + aperiodic processing
# =============================================================================

INF = float("inf")


def online_phase(schedule, sporadic_input, aperiodic_jobs, proc,
                 pv_forecast, err_std=0.0, price_arr=None):
    """Replaces Phase 2 + Phase 3 with a single release-time-ordered pass.

    Behaviour:
      - Arrivals are sorted by release hour; sporadic jobs win ties so the
        hard-deadline policy takes precedence over the soft-deadline one.
      - Each job sees the schedule as already mutated by every earlier
        arrival — true online semantics.
      - For sporadic: standard acceptance test. Slots are the e hours in
        [release, hard_deadline] with the smallest sell-borrow cost. Accept
        if a feasible set exists; else reject and log a reason.
      - For aperiodic: per spec C4 (must execute e hours by H) we always
        place. Prefer the cheapest on-time placement; fall back to the
        cheapest late placement in [release, H] only when on-time is
        infeasible. "Late" means missed_soft_deadline=True (counts toward
        f1) but the job still executes.
      - Commits use `_commit_min_sell` (gen + PV ramp first, sell only for
        the residual), so peak-price sell hours are preserved by default.

    Returns (acceptance_log, aperiodic_log).
    """
    absorber = SlackAbsorber(schedule, proc, pv_forecast, err_std=err_std)

    def _normal_slack(t):
        """Slack without sell — gen ramp + PV underuse."""
        return absorber.slack_at(t) - schedule[t - 1]["sell"]

    def _hour_cost(t, w):
        """Estimated $ of sell borrow at hour t to satisfy w MWh.
        Returns INF if even 100% sell + gen + PV cannot satisfy w.
        """
        free = _normal_slack(t)
        avail = free + schedule[t - 1]["sell"]
        if avail < w - EPS:
            return INF
        sell_needed = max(0.0, w - free)
        if price_arr is None:
            return sell_needed              # rank by raw MWh of sell if no prices
        return sell_needed * float(price_arr[t])

    def _find_min_cost(r, end, e_len, w_need, preempt):
        """Return (slots, total_cost) for the cheapest feasible placement of
        e_len hours of w_need MWh inside [r, end], or (None, INF) if none.
        For preempt=1: pick the e cheapest individual hours.
        For preempt=0: pick the cheapest sliding window of e consecutive hours.
        """
        if r > end or end - r + 1 < e_len:
            return None, INF
        if preempt:
            ranked = sorted(range(r, end + 1), key=lambda t: _hour_cost(t, w_need))
            chosen = ranked[:e_len]
            total = sum(_hour_cost(t, w_need) for t in chosen)
            if total >= INF:
                return None, INF
            return sorted(chosen), total
        best_window, best_cost = None, INF
        for start in range(r, end - e_len + 2):
            window = list(range(start, start + e_len))
            total = sum(_hour_cost(t, w_need) for t in window)
            if total < best_cost:
                best_cost, best_window = total, window
        if best_cost >= INF:
            return None, INF
        return best_window, best_cost

    def _commit_min_sell(t, w, jid):
        """Allocate w MWh at hour t, using gen + PV ramp first and sell only
        for the unmet residual. Returns (residual, sell_taken).
        """
        rec = schedule[t - 1]
        # Step 1: gen + PV (sell hidden so commit_at skips Step 1).
        saved_sell = rec["sell"]
        rec["sell"] = 0.0
        residual = absorber.commit_at(t, w, jid)
        rec["sell"] = saved_sell
        if residual <= EPS:
            return 0.0, 0.0
        # Step 2: take from sell for the residual.
        sell_take = min(residual, rec["sell"])
        if sell_take <= EPS:
            return residual, 0.0
        rec["sell"] = round(rec["sell"] - sell_take, 4)
        if "day_ahead_commit" in rec:
            rec["day_ahead_commit"] = round(
                max(0.0, rec["day_ahead_commit"] - sell_take), 4
            )
        already = {i: 0.0 for i in rec["P"]}
        for k_ent in rec["k"].values():
            for i, v in k_ent.items():
                if i in already:
                    already[i] += v
        to_dist = sell_take
        for i, p_val in sorted(rec["P"].items(), key=lambda kv: -kv[1]):
            free = p_val - already.get(i, 0.0)
            give = min(to_dist, free)
            if give > EPS:
                if jid in rec["k"]:
                    rec["k"][jid][i] = round(rec["k"][jid].get(i, 0.0) + give, 4)
                else:
                    rec["k"][jid] = {i: round(give, 4)}
                already[i] = already.get(i, 0.0) + give
                to_dist -= give
            if to_dist <= EPS:
                break
        committed_sell = sell_take - to_dist
        return residual - committed_sell, committed_sell

    def _group_consecutive(slots):
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

    def _commit_run_min_sell(run, w, jid):
        """Joint LP commit over a consecutive run of hours.

        Solves a small LP that simultaneously decides Δx (extra gen output),
        Δp (extra PV output), and s (sell take) at every hour in the run,
        respecting ramp_up/ramp_down constraints LINKING consecutive hours
        inside the run as well as the original-schedule values at the run's
        boundaries. Single-hour runs fall back to `_commit_min_sell`.

        Returns total sell MWh borrowed across the run.
        """
        if len(run) == 1:
            _, taken = _commit_min_sell(run[0], w, jid)
            return taken

        T = run
        prob = pulp.LpProblem(f"joint_{jid}_{T[0]}_{T[-1]}", pulp.LpMinimize)

        dx = {g: {t: pulp.LpVariable(f"dx_{g}_{t}", lowBound=0)
                  for t in T} for g in absorber.gen_ids}
        dp = {pv: {t: pulp.LpVariable(f"dp_{pv}_{t}", lowBound=0)
                   for t in T} for pv in absorber.pv_ids}
        s_t = {t: pulp.LpVariable(f"s_{t}", lowBound=0) for t in T}

        # Objective: minimise sell × price (or raw MWh if no prices)
        if price_arr is not None:
            prob += pulp.lpSum(s_t[t] * float(price_arr[t]) for t in T), "MinSellCost"
        else:
            prob += pulp.lpSum(s_t[t] for t in T), "MinSellMWh"

        # Per-hour demand and sell cap
        for t in T:
            prob += (pulp.lpSum(dx[g][t] for g in dx)
                     + pulp.lpSum(dp[pv][t] for pv in dp)
                     + s_t[t] == w), f"dem_{t}"
            prob += s_t[t] <= schedule[t - 1]["sell"], f"sell_cap_{t}"

        # Generator constraints: cap, ramp_up (incl. boundary in), ramp_down (boundary out)
        for g in absorber.gen_ids:
            gd = absorber.gens[g]
            ru = float(gd["ramp_up_rate"])
            rd = float(gd["ramp_down_rate"])
            out_max = float(gd["output_max"])
            for idx, t in enumerate(T):
                rec = schedule[t - 1]
                p_curr = rec["P"].get(g, 0.0)
                # Off generators cannot be turned on inside Phase 3 (UT/DT not
                # re-enforced); freeze Δx at 0.
                if p_curr <= EPS:
                    prob += dx[g][t] == 0, f"goff_{g}_{t}"
                    continue
                # Output cap
                prob += p_curr + dx[g][t] <= out_max, f"omx_{g}_{t}"
                # ramp_up at t (from t-1)
                if idx == 0:
                    p_prev = (schedule[t - 2]["P"].get(g, 0.0) if t > 1
                              else float(gd.get("initial_energy", 0)))
                    # If gen was off at t-1, ramp_up reduces to: new_P[t] <= ramp_up
                    # (since p_prev = 0). Same form below handles it.
                    prob += p_curr + dx[g][t] - p_prev <= ru, f"ru_{g}_{t}"
                else:
                    tp = T[idx - 1]
                    p_prev_curr = schedule[tp - 1]["P"].get(g, 0.0)
                    # Adjacent-hour ramp_up: (P[t]+dx[t]) - (P[tp]+dx[tp]) <= ru
                    prob += ((p_curr + dx[g][t]) - (p_prev_curr + dx[g][tp])
                             <= ru), f"ru_{g}_{t}"
                # ramp_down at t (toward t+1).
                # Boundary constraint at T_last is omitted: after the LP
                # solution is applied, _cascade_sell propagates any raise at
                # T_last to subsequent ON-hours via sell, maintaining
                # ramp_down feasibility outside the run window.
                if idx == len(T) - 1:
                    pass  # cascade handles post-run ramp_down
                else:
                    tn = T[idx + 1]
                    p_next_curr = schedule[tn - 1]["P"].get(g, 0.0)
                    # Symmetric: (P[t]+dx[t]) - (P[tn]+dx[tn]) <= rd
                    prob += ((p_curr + dx[g][t]) - (p_next_curr + dx[g][tn])
                             <= rd), f"rd_{g}_{t}"

        # PV: robust capacity cap
        for pv in absorber.pv_ids:
            cap = absorber.pvs[pv]["capacity"]
            for t in T:
                rec = schedule[t - 1]
                p_curr = rec["P"].get(pv, 0.0)
                max_pv = cap * absorber.pv_forecast[pv][t] * absorber.robust
                prob += p_curr + dp[pv][t] <= max_pv, f"pvmx_{pv}_{t}"

        solver = pulp.PULP_CBC_CMD(msg=False)
        prob.solve(solver)

        if prob.status != pulp.LpStatusOptimal:
            # Should not happen if _find_min_cost said feasible; fall back to
            # hour-by-hour commit so we still place the job.
            total = 0.0
            for tt in T:
                _, taken = _commit_min_sell(tt, w, jid)
                total += taken
            return total

        def _val(x):
            v = pulp.value(x)
            return float(v) if v is not None else 0.0

        # Apply solution to schedule
        total_sell_borrowed = 0.0
        for t in T:
            rec = schedule[t - 1]
            # Increment gen P and attribute to job
            for g in absorber.gen_ids:
                inc = _val(dx[g][t])
                if inc <= EPS:
                    continue
                rec["P"][g] = round(rec["P"].get(g, 0.0) + inc, 4)
                rec["k"].setdefault(jid, {})
                rec["k"][jid][g] = round(rec["k"][jid].get(g, 0.0) + inc, 4)
            # Increment PV P and attribute to job
            for pv in absorber.pv_ids:
                inc = _val(dp[pv][t])
                if inc <= EPS:
                    continue
                rec["P"][pv] = round(rec["P"].get(pv, 0.0) + inc, 4)
                rec["k"].setdefault(jid, {})
                rec["k"][jid][pv] = round(rec["k"][jid].get(pv, 0.0) + inc, 4)
            # Sell take: redirect existing free P at this hour to the job
            sell_take = _val(s_t[t])
            if sell_take <= EPS:
                continue
            total_sell_borrowed += sell_take
            rec["sell"] = round(rec["sell"] - sell_take, 4)
            if "day_ahead_commit" in rec:
                rec["day_ahead_commit"] = round(
                    max(0.0, rec["day_ahead_commit"] - sell_take), 4)
            already = {i: 0.0 for i in rec["P"]}
            for k_ent in rec["k"].values():
                for i, v in k_ent.items():
                    if i in already:
                        already[i] += v
            to_dist = sell_take
            for i, p_val in sorted(rec["P"].items(), key=lambda kv: -kv[1]):
                free = p_val - already.get(i, 0.0)
                give = min(to_dist, free)
                if give > EPS:
                    rec["k"].setdefault(jid, {})
                    rec["k"][jid][i] = round(rec["k"][jid].get(i, 0.0) + give, 4)
                    already[i] += give
                    to_dist -= give
                if to_dist <= EPS:
                    break

        # Cascade ramp_down from the last run hour to subsequent ON-hours.
        # For each generator raised at T[-1], propagate the new P level
        # forward so the static schedule hours after the run are ramp_down
        # feasible; extra generation at those hours is attributed to sell.
        T_last = T[-1]
        for g in absorber.gen_ids:
            new_p_last = schedule[T_last - 1]["P"].get(g, 0.0)
            absorber._cascade_sell(g, T_last, new_p_last)

        return total_sell_borrowed

    # ------- Build unified release-ordered arrival list ----------------------
    sporadic_items = (
        sporadic_input.items() if isinstance(sporadic_input, dict)
        else [(t.get("id", f"s{i}"), t) for i, t in enumerate(sporadic_input)]
    )
    arrivals = []
    for sid, sj in sporadic_items:
        r = int(sj.get("r", sj.get("release")))
        arrivals.append((r, 0, "sporadic", str(sid), sj))   # tie-break: 0 = sporadic first
    for aj in aperiodic_jobs:
        arrivals.append((aj["release"], 1, "aperiodic", aj["id"], aj))
    arrivals.sort(key=lambda x: (x[0], x[1]))

    acceptance_log = []
    aperiodic_log = []

    # ------- Process each arrival in release order ---------------------------
    for release_t, _, kind, jid, j in arrivals:
        if kind == "sporadic":
            sj = j
            r = int(sj.get("r", sj.get("release")))
            d_abs = (min(r + int(sj["d"]) - 1, H) if "d" in sj
                     else int(sj.get("hard_deadline", sj.get("deadline", H))))
            e = int(sj["e"])
            w = float(sj["w"])
            preempt = int(sj.get("preempt", 1))

            slots, cost = _find_min_cost(r, min(d_abs, H), e, w, preempt)
            if slots is None:
                acceptance_log.append({
                    "job_id": jid, "decision": "reject",
                    "arrival": r, "release": r, "deadline": d_abs,
                    "e": e, "w": w,
                    "reason": (f"no feasible {e}h placement of {w}MW "
                               f"in [{r},{d_abs}] (even at 100% sell)"),
                    "caused_violation": False,
                })
                if 1 <= r <= H:
                    schedule[r - 1]["rejected_sporadic"].append(jid)
                continue

            sell_borrowed = 0.0
            for run in _group_consecutive(slots):
                sell_borrowed += _commit_run_min_sell(run, w, jid)
            acceptance_log.append({
                "job_id": jid, "decision": "accept",
                "arrival": r, "release": r, "deadline": d_abs,
                "e": e, "w": w,
                "slots": slots,
                "sell_borrowed": round(sell_borrowed, 4),
                "cost_estimate": round(cost, 2),
                "caused_violation": False,
            })

        else:  # aperiodic — must execute (C4)
            r = j["release"]
            d_soft = j["deadline"]
            e = j["e"]
            w = j["w"]
            preempt = j.get("preempt", 1)

            # Prefer on-time when feasible.
            slots, cost = _find_min_cost(r, min(d_soft, H), e, w, preempt)
            on_time = slots is not None
            if not on_time:
                slots, cost = _find_min_cost(r, H, e, w, preempt)

            if slots is None:
                # Even late+100% sell can't fit → genuine infeasibility.
                aperiodic_log.append({
                    "job_id": jid, "decision": "infeasible",
                    "release": r, "soft_deadline": d_soft,
                    "e": e, "w": w, "slots": [],
                    "completion": None, "tardiness": None,
                    "missed_soft_deadline": True,
                    "via_sell_borrow": False,
                    "reason": (f"no feasible {e}h placement of {w}MW "
                               f"in [{r},{H}] even at 100% sell"),
                })
                if 1 <= d_soft <= H:
                    schedule[min(d_soft, H) - 1]["missed_aperiodic"].append(jid)
                continue

            sell_borrowed = 0.0
            for run in _group_consecutive(slots):
                sell_borrowed += _commit_run_min_sell(run, w, jid)
            completion = max(slots)
            tardiness = max(0, completion - d_soft)
            decision = "scheduled_on_time" if on_time else "scheduled_late"
            missed = not on_time
            aperiodic_log.append({
                "job_id": jid, "decision": decision,
                "release": r, "soft_deadline": d_soft,
                "e": e, "w": w, "slots": slots,
                "completion": completion, "tardiness": tardiness,
                "missed_soft_deadline": missed,
                "via_sell_borrow": (sell_borrowed > EPS),
                "sell_borrowed": round(sell_borrowed, 4),
                "cost_estimate": round(cost, 2),
            })
            if missed and 1 <= d_soft <= H:
                schedule[d_soft - 1]["missed_aperiodic"].append(jid)

    return acceptance_log, aperiodic_log


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

    # Single time-ordered online pass: sporadic acceptance test +
    # aperiodic force-placement, processed in release-time order.
    acceptance_log, aperiodic_log = online_phase(
        schedule, sporadic_input, aperiodic_jobs, proc, pv_forecast,
        err_std=err_std, price_arr=price_arr,
    )

    write_outputs(schedule, acceptance_log, aperiodic_log)
    print_summary(schedule, obj, periodic_jobs, aperiodic_jobs, sporadic_input)


if __name__ == "__main__":
    main()