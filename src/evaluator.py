import json
import math
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
    schedule   = load_json("output/schedule_result.json")["schedule_result"]
    acc_log    = load_json("output/acceptance_test_log.json")["acceptance_test_log"]
    prices     = load_json("input/price_72hr.json")["price"]
    proc       = load_json("input/processor_settings.json")

    # Demo-time dynamic jobs (sporadic + aperiodic). Falls back to task_set
    # for backward compatibility when dynamic_jobs.json doesn't exist.
    dynamic_path = os.path.join(BASE, "input", "dynamic_jobs.json")
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
            deadline = release + d
            periodic_instances.append({
                "task_id":    tid,
                "instance":   i,
                "release":    release,
                "deadline":   deadline,
                "completion": completion,
            })
            completions_by_task.setdefault(tid, []).append(completion)

    # ── sporadic instances from acceptance log ───────────────────────────────
    accepted_sporadic = [e for e in acc_log if e.get("decision") == "accept"]
    rejected_sporadic = [e for e in acc_log if e.get("decision") == "reject"]

    sporadic_instances: list[dict] = []
    for entry in accepted_sporadic:
        jid      = entry["job_id"]
        release  = entry.get("arrival", entry.get("release"))
        deadline = entry.get("deadline")
        e_val    = entry.get("e", 0)
        hours    = job_hours.get(jid, [])
        completion = max(hours) if hours else None
        sporadic_instances.append({
            "job_id":     jid,
            "release":    release,
            "deadline":   deadline,
            "e":          e_val,
            "completion": completion,
            "caused_violation": entry.get("caused_violation", False),
        })

    # ── all hard-deadline job records ────────────────────────────────────────
    hard_jobs = periodic_instances + [
        s for s in sporadic_instances if s["completion"] is not None
    ]

    # ── METRIC: hard_deadline_miss_rate ──────────────────────────────────────
    hard_misses = sum(1 for j in hard_jobs if j["completion"] > j["deadline"])
    hard_deadline_miss_rate = hard_misses / len(hard_jobs) if hard_jobs else 0.0

    # ── METRIC: soft_deadline_miss_rate ─────────────────────────────────────
    total_missed_ap = len(missed_aperiodic_ids)
    completed_ap = sum(
        len(job_hours.get(jid, [])) // task.get("e", 1)
        for jid, task in aperiodic_tasks.items()
    )
    total_ap = total_missed_ap + completed_ap
    soft_deadline_miss_rate = total_missed_ap / total_ap if total_ap > 0 else 0.0

    # ── METRIC: tardiness ────────────────────────────────────────────────────
    tardiness_vals = [max(0, j["completion"] - j["deadline"]) for j in hard_jobs]
    avg_tardiness = sum(tardiness_vals) / len(tardiness_vals) if tardiness_vals else 0.0
    max_tardiness = max(tardiness_vals) if tardiness_vals else 0.0

    # ── METRIC: response time ────────────────────────────────────────────────
    response_vals = [j["completion"] - j["release"] for j in hard_jobs]
    avg_response_time = sum(response_vals) / len(response_vals) if response_vals else 0.0
    max_response_time = max(response_vals) if response_vals else 0.0

    # ── METRIC: completion_time_jitter (per-task population std) ─────────────
    completion_time_jitter = {
        tid: round(population_std(comps), 4)
        for tid, comps in completions_by_task.items()
    }

    # ── METRIC: acceptance_test ──────────────────────────────────────────────
    acceptance_test = {
        "total":    len(acc_log),
        "accepted": len(accepted_sporadic),
        "rejected": len(rejected_sporadic),
    }

    # ── METRIC: sporadic_value_rate ──────────────────────────────────────────
    total_sp_exec   = sum(s["e"] for s in sporadic_instances)
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
    market_revenue = sum(price_map[t] * sell for t, sell in sell_by_hour.items())

    # ── METRIC: objective_value ───────────────────────────────────────────────
    alpha = 10000
    f1 = total_missed_ap          # aperiodic miss count
    f2 = generator_cost
    f3 = -market_revenue
    objective_value = alpha * f1 + f2 + f3

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
    }

    out_path = os.path.join(BASE, "output", "evaluation_results.json")
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
    print(f"  average_tardiness:  {avg_tardiness:.4f} h")
    print(f"  max_tardiness:      {max_tardiness:.4f} h")

    print(f"\n── Response Time  (Rj = Cj - rj) ──")
    print(f"  average_response_time:  {avg_response_time:.4f} h")
    print(f"  max_response_time:      {max_response_time:.4f} h")

    print(f"\n── Completion-Time Jitter (population std per task) ──")
    for tid in sorted(completion_time_jitter):
        comps = completions_by_task.get(tid, [])
        print(f"  {tid}: {completion_time_jitter[tid]:.4f} h  "
              f"(instances: {comps})")

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

    print(f"\n  Results saved → output/evaluation_results.json")
    print(sep)


if __name__ == "__main__":
    main()
