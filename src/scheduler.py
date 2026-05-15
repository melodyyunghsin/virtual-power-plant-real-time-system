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
ALPHA = 10000       # aperiodic miss penalty ($ per miss)
BIG_M = 10000       # big-M for mutex / disjunctive constraints
EPS = 1e-6          # numerical zero-threshold for output filtering

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


def parse_price(price):
    """Returns list[H+1] of $/MWh, 1-indexed."""
    arr = [0.0] * (H + 1)
    for v in price["price"]:
        arr[int(v["hour"])] = float(v["market_price"])
    return arr


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
        jobs.append({
            "id":       str(tid),
            "task_id":  str(tid),
            "release":  int(t["release"]),
            "deadline": int(t.get("soft_deadline", t.get("deadline", H))),
            "e":        int(t["e"]),
            "w":        float(t["w"]),
            "preempt":  int(t.get("preempt", 1)),
            "kind":     "aperiodic",
        })
    return jobs


# =============================================================================
# Phase 1 — Day-ahead static schedule (ILP)
# =============================================================================

def phase1_static_schedule(proc, pv_forecast, price_arr, real_jobs):
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
    P = pulp.LpVariable.dicts("P", (proc_ids, T), lowBound=0)

    u     = pulp.LpVariable.dicts("u",     (gen_ids, T), cat="Binary")
    z_on  = pulp.LpVariable.dicts("zon",   (gen_ids, T), cat="Binary")
    z_off = pulp.LpVariable.dicts("zoff",  (gen_ids, T), cat="Binary")

    chg   = pulp.LpVariable.dicts("chg",   (bat_ids, T), lowBound=0)
    dis   = pulp.LpVariable.dicts("dis",   (bat_ids, T), lowBound=0)
    soc   = pulp.LpVariable.dicts("soc",   (bat_ids, T), lowBound=0)
    v_chg = pulp.LpVariable.dicts("vchg",  (bat_ids, T), cat="Binary")

    sell        = pulp.LpVariable.dicts("sell",       T, lowBound=0)
    sell_share  = pulp.LpVariable.dicts("sellshare",  (proc_ids, T), lowBound=0)

    # job execution variables (only inside [release, deadline] window)
    x = {}   # x[jid][t]
    y = {}   # y[jid][s] — only for non-preemptive
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
    K = {}
    for j in real_jobs:
        jid = j["id"]
        K[jid] = {i: {} for i in proc_ids}
        for i in proc_ids:
            for t in x[jid]:
                K[jid][i][t] = pulp.LpVariable(
                    f"k_{jid}_{i}_{t}", lowBound=0
                )

    # Kchg[chg_id][i][t] — battery-charging allocation (i must NOT be the
    # battery being charged itself; cross-battery charging is allowed but
    # bounded by mutex on the source battery's discharge mode).
    Kchg = {}
    for cj in chg_jobs:
        cid = cj["id"]
        Kchg[cid] = {}
        for i in bat_ids:
            if i == cj["battery"]:
                continue
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
    )
    f3 = -pulp.lpSum(price_arr[t] * sell[t] for t in T)
    prob += ALPHA * f1 + f2 + f3, "TotalCost"

    # ----------------------------------------------------------- job execution
    for j in real_jobs:
        jid = j["id"]
        e   = j["e"]
        total_x = pulp.lpSum(x[jid].values())
        if j["kind"] == "aperiodic":
            # missed → no execution; otherwise exactly e hours
            prob += total_x == e * (1 - miss[jid]), f"jobE_{jid}"
        else:
            prob += total_x == e, f"jobE_{jid}"

        # non-preemptive: pick exactly one start, x = sum of covering starts
        if j["preempt"] == 0:
            sum_y = pulp.lpSum(y[jid].values())
            if j["kind"] == "aperiodic":
                prob += sum_y == 1 - miss[jid], f"oneStart_{jid}"
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
    for pv in pv_ids:
        cap = pv_by_id[pv]["capacity"]
        fc  = pv_forecast[pv]
        for t in T:
            prob += P[pv][t] <= cap * fc[t], f"pvmax_{pv}_{t}"

    # ----------------------------------------------------------- batteries
    for b in bat_ids:
        bd = bat_by_id[b]
        soc_min, soc_max = bd["soc_min"],     bd["soc_max"]
        chg_max, dis_max = bd["charge_max"],  bd["discharge_max"]
        soc_init         = bd["soc_init"]
        for t in T:
            prob += chg[b][t] <= chg_max * v_chg[b][t],     f"cmx_{b}_{t}"
            prob += dis[b][t] <= dis_max * (1 - v_chg[b][t]), f"dmx_{b}_{t}"
            prob += soc[b][t] >= soc_min, f"smin_{b}_{t}"
            prob += soc[b][t] <= soc_max, f"smax_{b}_{t}"
            prev = soc[b][t - 1] if t > 1 else soc_init
            prob += soc[b][t] == prev + chg[b][t] - dis[b][t], f"sdyn_{b}_{t}"
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
        rec = {
            "t":                 t,
            "P":                 {},
            "k":                 {},
            "sell":              round(val(sell[t]), 4),
            "soc":               {b: round(val(soc[b][t]), 4) for b in bat_ids},
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
# Phase 2 — Sporadic acceptance test
# =============================================================================

def phase2_acceptance(schedule, sporadic_input, proc):
    """For each arriving sporadic job, scan free capacity (current `sell`
    surplus) within [release, deadline] and try to slot in `e` hours of
    contiguous (or any, if preempt=1) execution at rate w. Accept if feasible
    without disturbing periodic jobs already placed.

    Returns log entries; mutates `schedule` in-place.
    """
    log = []
    if not sporadic_input:
        return log

    items = sporadic_input.items() if isinstance(sporadic_input, dict) else \
            [(t.get("id", f"s{i}"), t) for i, t in enumerate(sporadic_input)]

    for sid, sj in items:
        sid = str(sid)
        r       = int(sj["release"])
        d_abs   = int(sj.get("hard_deadline", sj.get("deadline", H)))
        e       = int(sj["e"])
        w       = float(sj["w"])
        preempt = int(sj.get("preempt", 1))

        window = [t for t in range(r, min(d_abs, H) + 1)]
        feasible_slots = [t for t in window if schedule[t - 1]["sell"] >= w - EPS]

        chosen = []
        if preempt == 0:
            # find e contiguous feasible slots
            for start in range(r, min(d_abs, H) - e + 2):
                block = list(range(start, start + e))
                if all(schedule[t - 1]["sell"] >= w - EPS for t in block):
                    chosen = block
                    break
        else:
            if len(feasible_slots) >= e:
                chosen = feasible_slots[:e]

        if chosen:
            # commit: peel `w` MW off `sell` and route through processors
            # that currently feed sell_share most heavily.
            for t in chosen:
                rec = schedule[t - 1]
                # Reduce sell, add allocation entry for sporadic job.
                rec["sell"] = round(rec["sell"] - w, 4)
                # Greedy: take from processors with output (record["P"]),
                # without breaking energy-balance bookkeeping.
                remaining = w
                allocation = {}
                for i, p_val in sorted(rec["P"].items(), key=lambda kv: -kv[1]):
                    take = min(remaining, p_val)
                    if take > EPS:
                        allocation[i] = round(take, 4)
                        remaining -= take
                    if remaining <= EPS:
                        break
                rec["k"][sid] = allocation
            log.append({
                "job_id":           sid,
                "decision":         "accept",
                "arrival":          r,
                "release":          r,
                "deadline":         d_abs,
                "e":                e,
                "w":                w,
                "slots":            chosen,
                "caused_violation": False,
            })
        else:
            log.append({
                "job_id":           sid,
                "decision":         "reject",
                "arrival":          r,
                "release":          r,
                "deadline":         d_abs,
                "e":                e,
                "w":                w,
                "reason":           "no feasible slack window",
                "caused_violation": False,
            })
            if 1 <= r <= H:
                schedule[r - 1]["rejected_sporadic"].append(sid)

    return log


# =============================================================================
# Output writing
# =============================================================================

def write_outputs(schedule, acceptance_log):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sched_path = OUTPUT_DIR / "schedule_result.json"
    with open(sched_path, "w", encoding="utf-8") as f:
        json.dump({"schedule_result": schedule}, f, indent=2)
    print(f"  wrote {sched_path}")

    log_path = OUTPUT_DIR / "acceptance_test_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({"acceptance_test_log": acceptance_log}, f, indent=2)
    print(f"  wrote {log_path}")


def print_summary(schedule, obj, real_jobs):
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
    print(f"  Jobs scheduled        : {len(real_jobs)}")
    print(f"  Active hours per proc :")
    for i, h in sorted(on_hours.items(), key=lambda kv: -kv[1]):
        print(f"    {i:<14}  {h:>3} h")
    print("=" * 60)


# =============================================================================
# Main
# =============================================================================

def main():
    proc, price_data, task_set = load_inputs()
    pv_forecast = parse_renewable_forecast(proc)
    price_arr   = parse_price(price_data)

    periodic_jobs  = expand_periodic(task_set.get("periodic", {}))
    aperiodic_jobs = expand_aperiodic(task_set.get("aperiodic", []))
    sporadic_input = task_set.get("sporadic", [])

    real_jobs = periodic_jobs + aperiodic_jobs
    print(f"[Input] {len(periodic_jobs)} periodic instances, "
          f"{len(aperiodic_jobs)} aperiodic, "
          f"{len(sporadic_input)} sporadic inbound")

    schedule, obj = phase1_static_schedule(
        proc, pv_forecast, price_arr, real_jobs
    )

    acceptance_log = phase2_acceptance(schedule, sporadic_input, proc)

    write_outputs(schedule, acceptance_log)
    print_summary(schedule, obj, real_jobs)


if __name__ == "__main__":
    main()
