import json
import math
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H = 72


def load_json(rel_path: str):
    with open(os.path.join(BASE, rel_path), encoding="utf-8") as f:
        return json.load(f)


def population_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def main():
    task_set   = load_json("output/task_set.json")
    schedule   = load_json("output/schedule_result_advanced.json")["schedule_result"]
    acc_full   = load_json("output/acceptance_test_log_advanced.json")
    # New rubric-prescribed output schema: one combined list distinguished by
    # entry["type"]. Split back into sporadic / aperiodic so the rest of the
    # evaluator (which references both buckets) keeps working unchanged.
    _entries = acc_full["acceptance_test_log"]
    acc_log = [e for e in _entries if e.get("type") == "sporadic"]
    aperiodic_log = [e for e in _entries if e.get("type") == "aperiodic"]
    prices     = load_json("input/price_72hr.json")["price"]
    proc       = load_json("input/processor_settings.json")

    # Demo-time dynamic jobs (sporadic + aperiodic). Same file as scheduler reads.
    # Falls back to task_set for backward compatibility when the file is absent.
    dynamic_path = os.path.join(BASE, "input", "aperiodic_n_sporadic.json")
    if os.path.exists(dynamic_path):
        with open(dynamic_path, encoding="utf-8") as f:
            dynamic = json.load(f)
    else:
        dynamic = {}

    price_map = {entry["hour"]: entry["market_price"] for entry in prices}
    gen_cfg   = {g["generator_id"]: g for g in proc["generator"]}

    # ── collect per-job execution hours and per-hour sell/P ──────────────────
    job_hours: dict[str, list[int]] = {}
    sell_by_hour: dict[int, float]  = {}
    P_by_hour: dict[int, dict]      = {}
    missed_aperiodic_ids: list      = []

    for slot in schedule:
        t = slot["t"]
        sell_by_hour[t] = slot.get("sell", 0.0)
        P_by_hour[t]    = slot.get("P", {})
        missed_aperiodic_ids.extend(slot.get("missed_aperiodic", []))
        for jid in slot.get("k", {}):
            if not jid.endswith("_chg"):
                job_hours.setdefault(jid, []).append(t)

    for jid in job_hours:
        job_hours[jid].sort()

    # ── expand periodic instances ────────────────────────────────────────────
    periodic = task_set.get("periodic", {})

    # Sporadic + aperiodic: prefer dynamic_jobs.json; fall back to task_set.
    # Normalize both list-of-dict and dict-of-dict formats to dict.
    def _to_dict(x):
        if isinstance(x, dict):
            return x
        if isinstance(x, list):
            return {t.get("id", f"j{i}"): t for i, t in enumerate(x)}
        return {}

    sporadic        = _to_dict(dynamic.get("sporadic",  task_set.get("sporadic",  {})))
    aperiodic_tasks = _to_dict(dynamic.get("aperiodic", task_set.get("aperiodic", {})))

    periodic_instances: list[dict] = []
    completions_by_task: dict[str, list[int]] = {}

    for tid, task in periodic.items():
        r, p_val, e, d = task["r"], task["p"], task["e"], task["d"]
        hours = job_hours.get(tid, [])
        num_instances = len(hours) // e

        for i in range(num_instances):
            chunk = hours[i * e : (i + 1) * e]
            completion = max(chunk)
            release  = r + i * p_val
            # Spec C3: execution window is [r, r+d-1]; last allowed slot = r+d-1
            deadline = release + d - 1
            periodic_instances.append({
                "task_id":    tid,
                "instance":   i,
                "release":    release,
                "deadline":   deadline,
                "completion": completion,
            })
            completions_by_task.setdefault(tid, []).append(completion)

    # ── sporadic instances from acceptance log ───────────────────────────────
    accepted_sporadic = [e for e in acc_log if e.get("accepted")]
    rejected_sporadic = [e for e in acc_log if not e.get("accepted")]

    sporadic_instances: list[dict] = []
    for entry in accepted_sporadic:
        jid      = entry["job_id"]
        release  = entry.get("release_time")
        deadline = entry.get("abs_deadline")
        e_val    = entry.get("execution_time", 0)
        hours    = job_hours.get(jid, [])
        completion = max(hours) if hours else None
        sporadic_instances.append({
            "job_id":     jid,
            "release":    release,
            "deadline":   deadline,
            "e":          e_val,
            "completion": completion,
            "caused_violation": False,
        })

    # ── aperiodic instances from phase-3 log ─────────────────────────────────
    # Each entry has type=aperiodic with assigned_hours (empty if skipped),
    # plus release_time / abs_deadline / accepted.
    aperiodic_instances: list[dict] = []
    for entry in aperiodic_log:
        slots    = entry.get("assigned_hours", [])
        deadline = entry.get("abs_deadline")
        completion = max(slots) if slots else None
        missed = (not entry.get("accepted", False)
                  or (completion is not None and completion > deadline))
        aperiodic_instances.append({
            "job_id":     entry["job_id"],
            "release":    entry.get("release_time"),
            "deadline":   deadline,
            "e":          entry.get("execution_time", 0),
            "completion": completion,
            "missed":     missed,
        })

    # ── all hard-deadline job records ────────────────────────────────────────
    hard_jobs = periodic_instances + [
        s for s in sporadic_instances if s["completion"] is not None
    ]
    # All jobs with a deadline that actually completed (for tardiness/response)
    all_completed_jobs = hard_jobs + [
        a for a in aperiodic_instances if a["completion"] is not None
    ]

    # ── METRIC: hard_deadline_miss_rate ──────────────────────────────────────
    hard_misses = sum(1 for j in hard_jobs if j["completion"] > j["deadline"])
    hard_deadline_miss_rate = hard_misses / len(hard_jobs) if hard_jobs else 0.0

    # ── METRIC: soft_deadline_miss_rate ─────────────────────────────────────
    total_missed_ap = sum(1 for a in aperiodic_instances if a["missed"])
    total_ap = len(aperiodic_tasks)
    soft_deadline_miss_rate = total_missed_ap / total_ap if total_ap > 0 else 0.0

    # ── METRIC: tardiness  T_j = max(0, C_j - d_j) ───────────────────────────
    # Spec rubric 5-3: applies to any job with a deadline.
    tardiness_vals = [max(0, j["completion"] - j["deadline"])
                      for j in all_completed_jobs]
    avg_tardiness = sum(tardiness_vals) / len(tardiness_vals) if tardiness_vals else 0.0
    max_tardiness = max(tardiness_vals) if tardiness_vals else 0.0

    # ── METRIC: response time  R_j = C_j - r_j ───────────────────────────────
    response_vals = [j["completion"] - j["release"] for j in all_completed_jobs]
    avg_response_time = sum(response_vals) / len(response_vals) if response_vals else 0.0
    max_response_time = max(response_vals) if response_vals else 0.0

    # ── METRIC: completion_time_jitter ───────────────────────────────────────
    # Defined as the population std of response times (R_j = C_j - r_j) across
    # instances of the same periodic task. Raw completion times span the whole
    # horizon and would yield meaninglessly large numbers.
    response_by_task: dict[str, list[int]] = {}
    for j in periodic_instances:
        response_by_task.setdefault(j["task_id"], []).append(
            j["completion"] - j["release"]
        )
    completion_time_jitter = {
        tid: round(population_std(rs), 4)
        for tid, rs in response_by_task.items()
    }

    # ── METRIC: acceptance_test ──────────────────────────────────────────────
    acceptance_test = {
        "total":    len(acc_log),
        "accepted": len(accepted_sporadic),
        "rejected": len(rejected_sporadic),
    }

    # ── METRIC: sporadic_value_rate ──────────────────────────────────────────
    # Denominator = ALL demo sporadic jobs (accepted + rejected), per spec.
    # Numerator   = accepted jobs whose completion <= hard deadline.
    total_sp_exec   = sum(entry.get("execution_time", 0) for entry in acc_log)
    ontime_sp_exec  = sum(
        s["e"] for s in sporadic_instances
        if s["completion"] is not None and s["completion"] <= s["deadline"]
    )
    sporadic_value_rate = (
        ontime_sp_exec / total_sp_exec if total_sp_exec > 0 else None
    )

    # ── METRIC: post_acceptance_violation_rate ───────────────────────────────
    post_violations = sum(1 for s in sporadic_instances if s["caused_violation"])
    post_acceptance_violation_rate = (
        post_violations / len(accepted_sporadic) if accepted_sporadic else 0.0
    )

    # ── METRIC: generator_cost ───────────────────────────────────────────────
    generator_cost = 0.0
    for slot in schedule:
        for gid, power in slot.get("P", {}).items():
            if gid in gen_cfg and power > 0:
                cfg = gen_cfg[gid]
                generator_cost += cfg["cost_fixed"] + cfg["cost_variable"] * power

    # ── METRIC: market_revenue ───────────────────────────────────────────────
    rt_factors  = {e["hour"]: e.get("realtime_price_factor", 1.0) for e in prices}
    cancel_rate = prices[0].get("cancellation_penalty_rate", 0.0)
    is_advanced = any("cancellation_penalty" in slot for slot in schedule)

    market_revenue = 0.0
    for slot in schedule:
        t      = slot["t"]
        sell_t = slot.get("sell", 0.0)
        p_da   = price_map.get(t, 0.0)
        if is_advanced:
            commit_t       = slot.get("day_ahead_commit", sell_t)
            committed_sold = min(sell_t, commit_t)
            overage        = max(0.0, sell_t - commit_t)
            cancel         = max(0.0, commit_t - sell_t)
            p_rt           = p_da * rt_factors.get(t, 1.0)
            market_revenue += (p_da * committed_sold
                               + p_rt * overage
                               - cancel_rate * p_da * cancel)
        else:
            market_revenue += p_da * sell_t

    # ── METRIC: objective_value ───────────────────────────────────────────────
    alpha = 10000
    f1 = total_missed_ap          # aperiodic miss count
    f2 = generator_cost
    f3 = -market_revenue
    objective_value = alpha * f1 + f2 + f3

    # ── METRIC: relaxed_assumptions (Level 2 storage metrics) ────────────────
    bat_cfg = {b["storage_id"]: b for b in proc["storage"]}
    total_aging_cost = 0.0
    soc_dep_binding_hours = 0
    for slot in schedule:
        t_slot = slot["t"]
        for bid, bat in bat_cfg.items():
            dis_val = slot.get("P", {}).get(bid, 0.0)
            aging = float(bat.get("aging_cost", 0.0))
            total_aging_cost += aging * dis_val
            prev_soc = (schedule[t_slot - 2]["soc"].get(bid, bat["soc_init"])
                        if t_slot > 1 else bat["soc_init"])
            if prev_soc < 0.3 * bat["soc_max"]:
                soc_dep_binding_hours += 1

    # ── renewable uncertainty metrics (Assumption I) ──────────────────────────
    ren_cfg     = {r["renewable_id"]: r for r in proc["renewable_capacity"]}
    pv_fc_map   = {}   # {pv_id: {hour: pv_forecast}}
    pv_act_map  = {}   # {pv_id: {hour: pv_actual}}
    err_std_val = 0.08
    for entry in proc["renewable_forecast"]:
        for pv_id, points in entry.items():
            pv_fc_map[pv_id]  = {int(v["hour"]): float(v["pv_forecast"])   for v in points}
            pv_act_map[pv_id] = {int(v["hour"]): float(v.get("pv_actual", v["pv_forecast"])) for v in points}

    total_forecast_MWh   = 0.0
    total_actual_MWh     = 0.0
    total_abs_error_MWh  = 0.0
    hours_shortfall      = 0
    hours_surplus        = 0
    robust_reduction_MWh = 0.0

    for pv_id, cap_info in ren_cfg.items():
        cap = float(cap_info["capacity"])
        for t in range(1, H + 1):
            fc  = pv_fc_map.get(pv_id, {}).get(t, 0.0)
            act = pv_act_map.get(pv_id, {}).get(t, fc)
            total_forecast_MWh  += cap * fc
            total_actual_MWh    += cap * act
            total_abs_error_MWh += cap * abs(act - fc)
            if act < fc - 1e-6:
                hours_shortfall += 1
            elif act > fc + 1e-6:
                hours_surplus   += 1
            robust_reduction_MWh += cap * fc * err_std_val

    renewable_uncertainty = {
        "forecast_error_std":        err_std_val,
        "total_forecast_MWh":        round(total_forecast_MWh, 2),
        "total_actual_MWh":          round(total_actual_MWh, 2),
        "total_absolute_error_MWh":  round(total_abs_error_MWh, 2),
        "hours_with_shortfall":      hours_shortfall,
        "hours_with_surplus":        hours_surplus,
        "robust_tightening_total_MWh": round(robust_reduction_MWh, 2),
    }

    # ── market mechanism metrics (Assumption III) ─────────────────────────────
    # rt_factors and cancel_rate already parsed above for market_revenue
    total_commit = sum(slot.get("day_ahead_commit", slot.get("sell", 0.0))
                       for slot in schedule)
    total_sell   = sum(slot.get("sell", 0.0) for slot in schedule)
    # penalty = sum(cancel_rate * price * max(0, commit - sell)) — 0 in static mode
    total_penalty = sum(
        cancel_rate * price_map.get(slot["t"], 0.0)
        * max(0.0, slot.get("day_ahead_commit", 0.0) - slot.get("sell", 0.0))
        for slot in schedule
    )
    rt_vals = list(rt_factors.values())
    market_mechanism = {
        "cancellation_penalty_rate": cancel_rate,
        "total_day_ahead_commit_MWh": round(total_commit, 4),
        "total_realtime_sell_MWh":    round(total_sell, 4),
        "total_cancellation_penalty": round(total_penalty, 2),
        "net_market_revenue":         round(market_revenue - total_penalty, 2),
        "realtime_price_stats": {
            "min_factor": round(min(rt_vals), 4),
            "max_factor": round(max(rt_vals), 4),
            "avg_factor": round(sum(rt_vals) / len(rt_vals), 4),
        },
    }

    relaxed_assumptions = {
        "assumption": "II_Real_Storage_Operation_III_FlexibleMarket_I_RenewableUncertainty",
        "charge_efficiency": 0.95,
        "discharge_efficiency": 0.92,
        "self_discharge_rate": 0.002,
        "aging_cost_per_MWh": 8.0,
        "total_aging_cost": round(total_aging_cost, 2),
        "soc_dep_discharge_binding_hours": soc_dep_binding_hours,
        "market_mechanism": market_mechanism,
        "renewable_uncertainty": renewable_uncertainty,
    }

    # ── assemble results dict ─────────────────────────────────────────────────
    results = {
        "hard_deadline_miss_rate":       round(hard_deadline_miss_rate, 6),
        "soft_deadline_miss_rate":       round(soft_deadline_miss_rate, 6),
        "average_tardiness":             round(avg_tardiness, 4),
        "max_tardiness":                 round(max_tardiness, 4),
        "average_response_time":         round(avg_response_time, 4),
        "max_response_time":             round(max_response_time, 4),
        "completion_time_jitter":        completion_time_jitter,
        "acceptance_test":               acceptance_test,
        "sporadic_value_rate":           sporadic_value_rate,
        "post_acceptance_violation_rate": round(post_acceptance_violation_rate, 6),
        "generator_cost":                round(generator_cost, 2),
        "market_revenue":                round(market_revenue, 2),
        "objective_value":               round(objective_value, 2),
        "relaxed_assumptions":           relaxed_assumptions,
    }

    out_path = os.path.join(BASE, "output", "evaluation_results_advanced.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # ── stdout report ────────────────────────────────────────────────────────
    sep = "=" * 64
    print(sep)
    print("  VPP Real-Time Scheduling – Evaluation Report")
    print(sep)

    # per-task instance table
    print("\n── Periodic Instance Detail ──")
    header = f"{'Task':<6} {'Inst':>4} {'Release':>8} {'Deadline':>9} {'Complete':>9} {'Tardy':>6} {'Miss':>5}"
    print(header)
    print("-" * len(header))
    for j in sorted(periodic_instances, key=lambda x: (x["task_id"], x["instance"])):
        tardy = max(0, j["completion"] - j["deadline"])
        miss  = "YES" if j["completion"] > j["deadline"] else "no"
        print(f"{j['task_id']:<6} {j['instance']:>4} {j['release']:>8} "
              f"{j['deadline']:>9} {j['completion']:>9} {tardy:>6} {miss:>5}")

    print(f"\n── Deadline Performance ──")
    print(f"  Hard-deadline jobs (periodic + accepted sporadic): {len(hard_jobs)}")
    print(f"  Hard misses:           {hard_misses}")
    print(f"  hard_deadline_miss_rate: {hard_deadline_miss_rate:.4%}")
    print(f"  Aperiodic total:       {total_ap}")
    print(f"  Aperiodic misses:      {total_missed_ap}")
    print(f"  soft_deadline_miss_rate: {soft_deadline_miss_rate:.4%}")

    print(f"\n── Tardiness  (Tj = max(0, Cj-dj)) ──")
    print(f"  Jobs counted (periodic + sporadic + aperiodic): {len(all_completed_jobs)}")
    print(f"  average_tardiness:  {avg_tardiness:.4f} h")
    print(f"  max_tardiness:      {max_tardiness:.4f} h")
    # Break out by category for transparency
    hard_tardy = [max(0, j["completion"] - j["deadline"]) for j in hard_jobs]
    soft_tardy = [max(0, a["completion"] - a["deadline"])
                  for a in aperiodic_instances if a["completion"] is not None]
    print(f"  hard-jobs avg / max:   "
          f"{(sum(hard_tardy)/len(hard_tardy)) if hard_tardy else 0:.4f} / "
          f"{max(hard_tardy) if hard_tardy else 0:.4f} h")
    print(f"  soft (aperiodic) avg / max: "
          f"{(sum(soft_tardy)/len(soft_tardy)) if soft_tardy else 0:.4f} / "
          f"{max(soft_tardy) if soft_tardy else 0:.4f} h")

    print(f"\n── Response Time  (Rj = Cj - rj) ──")
    print(f"  average_response_time:  {avg_response_time:.4f} h")
    print(f"  max_response_time:      {max_response_time:.4f} h")

    print(f"\n── Completion-Time Jitter (population std of response time) ──")
    for tid in sorted(completion_time_jitter):
        rs = response_by_task.get(tid, [])
        print(f"  {tid}: {completion_time_jitter[tid]:.4f} h  "
              f"(R_j: {rs})")

    print(f"\n── Sporadic Acceptance Test ──")
    print(f"  Total arrived:  {acceptance_test['total']}")
    print(f"  Accepted:       {acceptance_test['accepted']}")
    print(f"  Rejected:       {acceptance_test['rejected']}")
    svr = f"{sporadic_value_rate:.4%}" if sporadic_value_rate is not None else "N/A (no sporadic)"
    print(f"  sporadic_value_rate:              {svr}")
    print(f"  post_acceptance_violation_rate:   {post_acceptance_violation_rate:.4%}")

    print(f"\n── Economic Metrics ──")
    print(f"  generator_cost:   ${generator_cost:>14,.2f}")
    print(f"  market_revenue:   ${market_revenue:>14,.2f}")
    print(f"  net (rev-cost):   ${market_revenue - generator_cost:>14,.2f}")

    print(f"\n── Objective Value  (α·f1 + f2 + f3,  α={alpha}) ──")
    print(f"  f1 (aperiodic misses):  {f1}")
    print(f"  f2 (generator cost):    ${f2:,.2f}")
    print(f"  f3 (-market_revenue):   ${f3:,.2f}")
    print(f"  objective_value:        ${objective_value:,.2f}")

    print(f"\n  Results saved → output/evaluation_results_advanced.json")
    print(sep)


if __name__ == "__main__":
    main()
