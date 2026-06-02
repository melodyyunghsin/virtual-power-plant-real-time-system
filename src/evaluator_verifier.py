"""
Independent re-computation of evaluator outputs.

Re-derives every numeric metric in evaluation_results.json (static) and
evaluation_results_advanced.json (dynamic) directly from the raw schedule,
inputs, and acceptance logs — then compares against what the evaluator
reported. Any mismatch beyond the per-metric tolerance is logged.

Run:
    python3 src/evaluator_verifier.py            # both
    python3 src/evaluator_verifier.py static     # static only
    python3 src/evaluator_verifier.py advanced   # advanced only
"""

import json
import math
import sys
from pathlib import Path

ROOT       = Path(__file__).resolve().parent.parent
INPUT_DIR  = ROOT / "input"
OUTPUT_DIR = ROOT / "output"
H          = 72
ALPHA      = 10000


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def population_std(values):
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _to_dict(x, prefix="j"):
    """Normalize aperiodic/sporadic section to dict-of-dict."""
    if isinstance(x, dict):
        return x
    if isinstance(x, list):
        return {t.get("id", f"{prefix}{i}"): t for i, t in enumerate(x)}
    return {}


def _abs_deadline(task, fallback_key):
    """Convert new ('d' = relative) or old (absolute) format to absolute hour."""
    r = int(task.get("r", task.get("release", 0)))
    if "d" in task:
        return min(r + int(task["d"]) - 1, H)
    return int(task.get(fallback_key, task.get("deadline", H)))


def _periodic_instances(periodic_set, job_hours):
    """Re-derive periodic instance records (task_id, release, deadline, completion)."""
    out = []
    for tid, task in periodic_set.items():
        r, p, e, d = task["r"], task["p"], task["e"], task["d"]
        hrs = job_hours.get(tid, [])
        num = len(hrs) // e
        for i in range(num):
            chunk = hrs[i * e : (i + 1) * e]
            release  = r + i * p
            deadline = release + d - 1
            out.append({
                "task_id":    tid,
                "instance":   i,
                "release":    release,
                "deadline":   deadline,
                "completion": max(chunk),
            })
    return out


def _job_hours_and_sell(sched):
    """Walk the schedule once → {jid: sorted hours}, {t: sell}, {t: P}, missed list."""
    job_hours, sell_by_hour, P_by_hour = {}, {}, {}
    missed_ap = []
    for slot in sched:
        t = slot["t"]
        sell_by_hour[t] = slot.get("sell", 0.0)
        P_by_hour[t]    = slot.get("P", {})
        missed_ap.extend(slot.get("missed_aperiodic", []))
        for jid in slot.get("k", {}):
            if not jid.endswith("_chg"):
                job_hours.setdefault(jid, []).append(t)
    for jid in job_hours:
        job_hours[jid].sort()
    return job_hours, sell_by_hour, P_by_hour, missed_ap


def _gen_cost(sched, gen_cfg):
    total = 0.0
    for slot in sched:
        for gid, power in slot.get("P", {}).items():
            if gid in gen_cfg and power > 0:
                g = gen_cfg[gid]
                total += g["cost_fixed"] + g["cost_variable"] * power
    return total


# ---------------------------------------------------------------------------
# Comparison primitives
# ---------------------------------------------------------------------------

class Report:
    def __init__(self, label):
        self.label  = label
        self.checks = []  # list of (name, got, want, ok, detail)

    def check(self, name, got, want, tol=0.01):
        if got is None and want is None:
            self.checks.append((name, got, want, True, "both None"))
            return
        if got is None or want is None:
            self.checks.append((name, got, want, False,
                                f"one side None (got={got}, want={want})"))
            return
        diff = abs(float(got) - float(want))
        ok   = diff <= tol
        self.checks.append((name, got, want, ok, f"|diff|={diff:.6g} (tol={tol})"))

    def check_int(self, name, got, want):
        ok = got == want
        self.checks.append((name, got, want, ok, "equal" if ok else "differ"))

    def check_dict(self, name, got, want, tol=1e-4):
        got  = got or {}
        want = want or {}
        keys = set(got) | set(want)
        bad  = [k for k in keys
                if abs(float(got.get(k, 0)) - float(want.get(k, 0))) > tol]
        ok = not bad
        self.checks.append((name, len(got), len(want), ok,
                            "match" if ok else f"differs on: {bad}"))

    def print(self):
        print("=" * 60)
        print(f"  {self.label}")
        print("=" * 60)
        fails = [c for c in self.checks if not c[3]]
        for name, got, want, ok, detail in self.checks:
            tag = "OK  " if ok else "FAIL"
            print(f"  [{tag}] {name:<42} got={got}  want={want}   {detail}")
        if fails:
            print(f"\n  {len(fails)} mismatch(es) out of {len(self.checks)} checks.")
        else:
            print(f"\n  All {len(self.checks)} metric checks passed.")
        print("=" * 60)
        return len(fails)


# ---------------------------------------------------------------------------
# Static evaluator verifier
# ---------------------------------------------------------------------------

def verify_static():
    r = Report("Static Evaluator Verifier (evaluation_results.json)")

    # ---- inputs / outputs --------------------------------------------------
    proc      = load_json(INPUT_DIR / "processor_settings.json")
    price     = load_json(INPUT_DIR / "price_72hr.json")
    task_set  = load_json(OUTPUT_DIR / "task_set.json")
    sched     = load_json(OUTPUT_DIR / "schedule_result.json")["schedule_result"]
    acc_full  = load_json(OUTPUT_DIR / "acceptance_test_log.json")
    # New rubric output schema: single combined list with `type` field.
    _entries  = acc_full["acceptance_test_log"]
    acc_log   = [e for e in _entries if e.get("type") == "sporadic"]
    ap_log    = [e for e in _entries if e.get("type") == "aperiodic"]
    expected  = load_json(OUTPUT_DIR / "evaluation_results.json")

    demo_path = INPUT_DIR / "aperiodic_n_sporadic.json"
    demo = load_json(demo_path) if demo_path.exists() else {}

    periodic        = task_set.get("periodic", {})
    sporadic_tasks  = _to_dict(demo.get("sporadic",  task_set.get("sporadic",  {})), "s")
    aperiodic_tasks = _to_dict(demo.get("aperiodic", task_set.get("aperiodic", {})), "a")

    gen_cfg   = {g["generator_id"]: g for g in proc["generator"]}
    bat_cfg   = {b["storage_id"]:   b for b in proc["storage"]}
    ren_cfg   = {x["renewable_id"]: x for x in proc["renewable_capacity"]}
    price_map = {v["hour"]: v["market_price"] for v in price["price"]}
    rt_factors  = {v["hour"]: v.get("realtime_price_factor", 1.0)
                   for v in price["price"]}
    cancel_rate = price["price"][0].get("cancellation_penalty_rate", 0.0)

    job_hours, _, _, _ = _job_hours_and_sell(sched)

    # ---- periodic instances -----------------------------------------------
    periodic_inst = _periodic_instances(periodic, job_hours)

    # ---- sporadic instances (from accept log + schedule completion) -------
    accepted = [e for e in acc_log if e.get("accepted")]
    rejected = [e for e in acc_log if not e.get("accepted")]
    sporadic_inst = []
    for entry in accepted:
        jid = entry["job_id"]
        hrs = job_hours.get(jid, [])
        sporadic_inst.append({
            "release":    entry.get("release_time"),
            "deadline":   entry.get("abs_deadline"),
            "e":          entry.get("execution_time", 0),
            "completion": max(hrs) if hrs else None,
            "caused_violation": False,
        })

    # ---- aperiodic instances (from phase-3 log) ---------------------------
    aperiodic_inst = []
    for entry in ap_log:
        slots      = entry.get("assigned_hours", [])
        deadline   = entry.get("abs_deadline")
        completion = max(slots) if slots else None
        missed = (not entry.get("accepted", False)
                  or (completion is not None and completion > deadline))
        aperiodic_inst.append({
            "release":    entry.get("release_time"),
            "deadline":   deadline,
            "e":          entry.get("execution_time", 0),
            "completion": completion,
            "missed":     missed,
        })

    hard_jobs = periodic_inst + [s for s in sporadic_inst if s["completion"] is not None]
    all_completed = hard_jobs + [a for a in aperiodic_inst if a["completion"] is not None]

    # ---- metrics -----------------------------------------------------------
    hard_misses = sum(1 for j in hard_jobs if j["completion"] > j["deadline"])
    hard_miss_rate = hard_misses / len(hard_jobs) if hard_jobs else 0.0

    missed_ap   = sum(1 for a in aperiodic_inst if a["missed"])
    soft_miss_rate = missed_ap / len(aperiodic_tasks) if aperiodic_tasks else 0.0

    tardiness_vals = [max(0, j["completion"] - j["deadline"]) for j in all_completed]
    avg_tardy = sum(tardiness_vals) / len(tardiness_vals) if tardiness_vals else 0.0
    max_tardy = max(tardiness_vals) if tardiness_vals else 0.0

    response_vals = [j["completion"] - j["release"] for j in all_completed]
    avg_resp = sum(response_vals) / len(response_vals) if response_vals else 0.0
    max_resp = max(response_vals) if response_vals else 0.0

    # completion-time jitter = pop-std of response time per periodic task
    by_task = {}
    for j in periodic_inst:
        by_task.setdefault(j["task_id"], []).append(j["completion"] - j["release"])
    jitter = {tid: round(population_std(rs), 4) for tid, rs in by_task.items()}

    total_sp_e   = sum(e.get("e", 0) for e in acc_log)
    ontime_sp_e  = sum(s["e"] for s in sporadic_inst
                       if s["completion"] is not None
                       and s["completion"] <= s["deadline"])
    svr = ontime_sp_e / total_sp_e if total_sp_e > 0 else None

    post_violations = sum(1 for s in sporadic_inst if s["caused_violation"])
    pavr = post_violations / len(accepted) if accepted else 0.0

    gen_cost = _gen_cost(sched, gen_cfg)

    is_advanced = any("cancellation_penalty" in slot for slot in sched)
    market_rev = 0.0
    for slot in sched:
        t      = slot["t"]
        sell_t = slot.get("sell", 0.0)
        p_da   = price_map.get(t, 0.0)
        if is_advanced:
            commit_t = slot.get("day_ahead_commit", sell_t)
            committed_sold = min(sell_t, commit_t)
            overage  = max(0.0, sell_t - commit_t)
            cancel   = max(0.0, commit_t - sell_t)
            p_rt     = p_da * rt_factors.get(t, 1.0)
            market_rev += (p_da * committed_sold + p_rt * overage
                           - cancel_rate * p_da * cancel)
        else:
            market_rev += p_da * sell_t

    obj = ALPHA * missed_ap + gen_cost - market_rev

    # ---- relaxed_assumptions metrics --------------------------------------
    total_aging = 0.0
    soc_dep_hours = 0
    for slot in sched:
        t = slot["t"]
        for bid, bat in bat_cfg.items():
            dis_val = slot.get("P", {}).get(bid, 0.0)
            total_aging += float(bat.get("aging_cost", 0.0)) * dis_val
            prev_soc = (sched[t - 2]["soc"].get(bid, bat["soc_init"])
                        if t > 1 else bat["soc_init"])
            if prev_soc < 0.3 * bat["soc_max"]:
                soc_dep_hours += 1

    # renewable uncertainty
    pv_fc  = {}
    pv_act = {}
    for entry in proc["renewable_forecast"]:
        for pv_id, pts in entry.items():
            pv_fc[pv_id]  = {int(v["hour"]): float(v["pv_forecast"]) for v in pts}
            pv_act[pv_id] = {int(v["hour"]): float(v.get("pv_actual", v["pv_forecast"]))
                             for v in pts}

    total_fc_MWh = total_act_MWh = total_abs_err = 0.0
    hours_short = hours_surplus = 0
    robust_red = 0.0
    err_std = 0.08
    for pv_id, cap_info in ren_cfg.items():
        cap = cap_info["capacity"]
        for t in range(1, H + 1):
            fc  = pv_fc.get(pv_id,  {}).get(t, 0.0)
            act = pv_act.get(pv_id, {}).get(t, fc)
            total_fc_MWh  += cap * fc
            total_act_MWh += cap * act
            total_abs_err += cap * abs(act - fc)
            if   act < fc - 1e-6: hours_short  += 1
            elif act > fc + 1e-6: hours_surplus += 1
            robust_red += cap * fc * err_std

    total_commit = sum(slot.get("day_ahead_commit", slot.get("sell", 0.0))
                       for slot in sched)
    total_sell   = sum(slot.get("sell", 0.0) for slot in sched)
    total_pen    = sum(cancel_rate * price_map.get(slot["t"], 0.0)
                       * max(0.0, slot.get("day_ahead_commit", 0.0)
                                  - slot.get("sell", 0.0))
                       for slot in sched)
    rt_vals = list(rt_factors.values())

    # ---- checks ------------------------------------------------------------
    r.check("hard_deadline_miss_rate",  round(hard_miss_rate, 6),
                                        expected["hard_deadline_miss_rate"], 1e-5)
    r.check("soft_deadline_miss_rate",  round(soft_miss_rate, 6),
                                        expected["soft_deadline_miss_rate"], 1e-5)
    r.check("average_tardiness",        round(avg_tardy, 4),
                                        expected["average_tardiness"], 1e-3)
    r.check("max_tardiness",            round(max_tardy, 4),
                                        expected["max_tardiness"], 1e-3)
    r.check("average_response_time",    round(avg_resp, 4),
                                        expected["average_response_time"], 1e-3)
    r.check("max_response_time",        round(max_resp, 4),
                                        expected["max_response_time"], 1e-3)
    r.check_dict("completion_time_jitter", jitter,
                                           expected["completion_time_jitter"], 1e-3)
    r.check_int("acceptance_test.total",    len(acc_log),
                                            expected["acceptance_test"]["total"])
    r.check_int("acceptance_test.accepted", len(accepted),
                                            expected["acceptance_test"]["accepted"])
    r.check_int("acceptance_test.rejected", len(rejected),
                                            expected["acceptance_test"]["rejected"])
    r.check("sporadic_value_rate",          svr, expected["sporadic_value_rate"], 1e-5)
    r.check("post_acceptance_violation_rate",
            round(pavr, 6), expected["post_acceptance_violation_rate"], 1e-5)
    r.check("generator_cost",       round(gen_cost, 2),  expected["generator_cost"], 0.02)
    r.check("market_revenue",       round(market_rev, 2), expected["market_revenue"], 0.02)
    r.check("objective_value",      round(obj, 2),        expected["objective_value"], 0.05)

    rax = expected.get("relaxed_assumptions", {})
    if rax:
        r.check("total_aging_cost",  round(total_aging, 2),
                                     rax.get("total_aging_cost"), 0.02)
        r.check_int("soc_dep_discharge_binding_hours",
                    soc_dep_hours, rax.get("soc_dep_discharge_binding_hours"))

        mm = rax.get("market_mechanism", {})
        r.check("market.total_day_ahead_commit_MWh", round(total_commit, 4),
                                                     mm.get("total_day_ahead_commit_MWh"), 0.01)
        r.check("market.total_realtime_sell_MWh",    round(total_sell, 4),
                                                     mm.get("total_realtime_sell_MWh"), 0.01)
        r.check("market.total_cancellation_penalty", round(total_pen, 2),
                                                     mm.get("total_cancellation_penalty"), 0.02)
        r.check("market.net_market_revenue",         round(market_rev - total_pen, 2),
                                                     mm.get("net_market_revenue"), 0.02)
        rps = mm.get("realtime_price_stats", {})
        r.check("market.rt.min_factor", round(min(rt_vals), 4), rps.get("min_factor"), 1e-3)
        r.check("market.rt.max_factor", round(max(rt_vals), 4), rps.get("max_factor"), 1e-3)
        r.check("market.rt.avg_factor", round(sum(rt_vals) / len(rt_vals), 4),
                                        rps.get("avg_factor"), 1e-3)

        ru = rax.get("renewable_uncertainty", {})
        r.check("renew.total_forecast_MWh",   round(total_fc_MWh, 2),
                                              ru.get("total_forecast_MWh"), 0.02)
        r.check("renew.total_actual_MWh",     round(total_act_MWh, 2),
                                              ru.get("total_actual_MWh"), 0.02)
        r.check("renew.total_absolute_error_MWh", round(total_abs_err, 2),
                                                  ru.get("total_absolute_error_MWh"), 0.02)
        r.check_int("renew.hours_with_shortfall", hours_short,
                                                  ru.get("hours_with_shortfall"))
        r.check_int("renew.hours_with_surplus",   hours_surplus,
                                                  ru.get("hours_with_surplus"))
        r.check("renew.robust_tightening_total_MWh", round(robust_red, 2),
                                                     ru.get("robust_tightening_total_MWh"), 0.02)

    return r.print()


# ---------------------------------------------------------------------------
# Advanced evaluator verifier
# ---------------------------------------------------------------------------

def verify_advanced():
    r = Report("Advanced Evaluator Verifier (evaluation_results_advanced.json)")

    proc       = load_json(INPUT_DIR / "processor_settings.json")
    price      = load_json(INPUT_DIR / "price_72hr.json")
    task_set   = load_json(OUTPUT_DIR / "task_set.json")
    sched      = load_json(OUTPUT_DIR / "schedule_result_advanced.json")["schedule_result"]
    acc_full   = load_json(OUTPUT_DIR / "acceptance_test_log_advanced.json")
    _entries   = acc_full["acceptance_test_log"]
    acc_log    = [e for e in _entries if e.get("type") == "sporadic"]
    ap_log     = [e for e in _entries if e.get("type") == "aperiodic"]
    expected   = load_json(OUTPUT_DIR / "evaluation_results_advanced.json")
    static_eval_path = OUTPUT_DIR / "evaluation_results.json"
    static_eval = load_json(static_eval_path) if static_eval_path.exists() else {}

    demo_path = INPUT_DIR / "aperiodic_n_sporadic.json"
    demo = load_json(demo_path) if demo_path.exists() else {}
    aperiodic_tasks = _to_dict(demo.get("aperiodic", task_set.get("aperiodic", {})), "a")

    gen_cfg   = {g["generator_id"]: g for g in proc["generator"]}
    price_map = {v["hour"]: v["market_price"] for v in price["price"]}
    rt_factors  = {v["hour"]: v.get("realtime_price_factor", 1.0)
                   for v in price["price"]}
    cancel_rate = price["price"][0].get("cancellation_penalty_rate", 0.0)

    job_hours, _, _, _ = _job_hours_and_sell(sched)
    periodic = task_set.get("periodic", {})

    # ---- periodic instances (greedy chunking matches advanced_scheduler) --
    periodic_inst = _periodic_instances(periodic, job_hours)
    # advanced_scheduler counts an instance as "missed" if completion None OR > deadline
    accepted = [e for e in acc_log if e.get("accepted")]
    hard_misses = sum(
        1 for inst in periodic_inst
        if inst["completion"] is None or inst["completion"] > inst["deadline"]
    )
    denom = max(1, len(periodic_inst) + len(accepted))
    hard_miss_rate = hard_misses / denom

    # ---- sporadic completion ----------------------------------------------
    # Normalize new-schema sporadic entries to the legacy field names used
    # by the rest of this block.
    accepted_legacy = [{
        "job_id":   rec["job_id"],
        "release":  rec.get("release_time"),
        "deadline": rec.get("abs_deadline"),
        "e":        rec.get("execution_time", 0),
    } for rec in accepted]

    sporadic_completed = []
    for rec in accepted_legacy:
        hrs = job_hours.get(rec["job_id"], [])
        sporadic_completed.append({**rec, "completion": max(hrs) if hrs else None})
    sp_total_e  = sum(rec["e"] for rec in accepted_legacy)
    sp_ontime_e = sum(rec["e"] for rec in sporadic_completed
                      if rec["completion"] is not None
                      and rec["completion"] <= rec["deadline"])
    svr = sp_ontime_e / sp_total_e if sp_total_e > 0 else None

    # ---- aperiodic miss rate ----------------------------------------------
    missed_ap = 0
    for entry in ap_log:
        slots = entry.get("assigned_hours", [])
        deadline = entry.get("abs_deadline")
        completion = max(slots) if slots else None
        if (not entry.get("accepted", False)
                or (completion is not None and completion > deadline)):
            missed_ap += 1
    soft_miss_rate = missed_ap / len(aperiodic_tasks) if aperiodic_tasks else 0.0

    # ---- economic ---------------------------------------------------------
    gen_cost = _gen_cost(sched, gen_cfg)
    market_rev = 0.0
    total_pen  = 0.0
    for slot in sched:
        t      = slot["t"]
        sell_t = slot.get("sell", 0.0)
        commit_t = slot.get("day_ahead_commit", sell_t)
        committed_sold = min(sell_t, commit_t)
        overage  = max(0.0, sell_t - commit_t)
        p_da   = price_map.get(t, 0.0)
        p_rt   = p_da * rt_factors.get(t, 1.0)
        market_rev += p_da * committed_sold + p_rt * overage
        # advanced scheduler also stores per-slot cancellation_penalty
        total_pen  += float(slot.get("cancellation_penalty", 0.0))

    obj = ALPHA * missed_ap + gen_cost - market_rev

    # ---- checks ------------------------------------------------------------
    r.check("hard_deadline_miss_rate",  round(hard_miss_rate, 6),
                                        expected["hard_deadline_miss_rate"], 1e-5)
    r.check("soft_deadline_miss_rate",  round(soft_miss_rate, 6),
                                        expected["soft_deadline_miss_rate"], 1e-5)
    r.check("generator_cost",           round(gen_cost, 2),
                                        expected["generator_cost"], 0.02)
    r.check("market_revenue",           round(market_rev, 2),
                                        expected["market_revenue"], 0.02)
    r.check("cancellation_penalty",     round(total_pen, 2),
                                        expected["cancellation_penalty"], 0.02)
    r.check("net_market_revenue",       round(market_rev - total_pen, 2),
                                        expected["net_market_revenue"], 0.02)
    r.check("objective_value",          round(obj, 2),
                                        expected["objective_value"], 0.05)
    r.check_int("acceptance_test.total",    len(acc_log),
                                            expected["acceptance_test"]["total"])
    r.check_int("acceptance_test.accepted", len(accepted),
                                            expected["acceptance_test"]["accepted"])
    r.check_int("acceptance_test.rejected", len(acc_log) - len(accepted),
                                            expected["acceptance_test"]["rejected"])
    r.check("sporadic_value_rate", svr, expected["sporadic_value_rate"], 1e-5)

    # ---- vs_static block: each value should equal what static eval reports
    vs = expected.get("advanced_scheduler", {}).get("vs_static", {})
    if vs and static_eval:
        r.check("vs_static.objective_value_static",
                static_eval.get("objective_value"),
                vs.get("objective_value_static"), 0.02)
        r.check("vs_static.generator_cost_static",
                static_eval.get("generator_cost"),
                vs.get("generator_cost_static"), 0.02)
        r.check("vs_static.market_revenue_static",
                static_eval.get("market_revenue"),
                vs.get("market_revenue_static"), 0.02)
        r.check("vs_static.soft_miss_rate_static",
                static_eval.get("soft_deadline_miss_rate"),
                vs.get("soft_miss_rate_static"), 1e-5)
        r.check("vs_static.sporadic_value_rate_static",
                static_eval.get("sporadic_value_rate"),
                vs.get("sporadic_value_rate_static"), 1e-5)
        r.check("vs_static.cancellation_penalty_total",
                round(total_pen, 2),
                vs.get("cancellation_penalty_total"), 0.02)

    return r.print()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    failed = 0
    if which in ("static", "both"):
        failed += verify_static()
    if which in ("advanced", "both"):
        failed += verify_advanced()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
