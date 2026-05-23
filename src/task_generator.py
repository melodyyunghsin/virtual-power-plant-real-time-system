"""
VPP Real-Time Scheduling System — Periodic Task Set Generator

Generation strategy (guarantees all constraints):
  - Always targets frame size f=4 (divides H=72, satisfies f>=max_e=4)
  - Tight-deadline tasks are pinned to: e=4, d=4, p divisible by 4
    → ensures 2*4 - gcd(4,p) = 8-4 = 4 = d (frame constraint met)
  - Free tasks have d >= 2*4 - gcd(4,p) by construction
  - Retry (up to 100000 seeds) until all constraints pass + frame size found
"""

import json
import math
import random
import sys
from pathlib import Path

H = 72  # planning horizon (hours)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _find_frame_size(tasks: list[dict]) -> int | None: # Level 1-8 評分標準
    """Return minimum valid f satisfying all three frame-size conditions."""
    max_e = max(t["e"] for t in tasks)
    for f in range(1, H + 1):
        if H % f != 0: # frame size 需能整除 72
            continue
        if f < max_e: # frame size 需大於等於所有任務中最大的execution time
            continue
        if all(2 * f - math.gcd(f, t["p"]) <= t["d"] for t in tasks): # 2f - gcd(f, p) <= d
            return f
    return None


def _count_expanded_jobs(tasks: list[dict]) -> int:
    """Count job instances over [1..H] for each task (release at r, r+p, ...)."""
    return sum((H - t["r"]) // t["p"] + 1 for t in tasks)


# ---------------------------------------------------------------------------
# Single-attempt generator
# ---------------------------------------------------------------------------

def _try_generate(rng: random.Random) -> list[dict] | None:
    TARGET_F = 4

    n = rng.randint(6, 10) # Level 1-2 評分標準: periodic task in [6,10]
    n_tight = max(2, math.ceil(0.2 * n))   # Level 1-6 評分標準: 至少20%的 deadline == execution time
    n_free = n - n_tight

    # --- tight tasks: e=d=4, p divisible by 4 ---
    # 4 <= gcd(4, p) : p 需為 4 的倍數，且在 [6,24] 範圍內
    tight_p_opts = [8, 12, 16, 20, 24]
    tight_tasks: list[dict] = []
    for _ in range(n_tight):
        p = rng.choice(tight_p_opts)
        # 1 <= r <= p 且 最後一個 job 的 release time + execution time <= H
        valid_r = [r for r in range(1, p + 1) if (H - r) % p >= 4]
        if not valid_r:
            return None
        r = rng.choice(valid_r)
        w = rng.randint(6, 18)
        tight_tasks.append({"r": r, "p": p, "e": 4, "d": 4, "w": w, "preempt": 0})

    # --- free tasks ---
    e_pool = [2, 2] # 至少 2 個 periodic task 的 𝒆=𝟐，
    while len(e_pool) < n_free:
        e_pool.append(rng.randint(1, 4))
    rng.shuffle(e_pool)

    all_p = list(range(6, 25))
    free_tasks: list[dict] = []
    for e in e_pool:
        p = rng.choice(all_p)
        # 2f - gcd(f, p) <= d 且 e <= d <= p
        min_d = max(e, 2 * TARGET_F - math.gcd(TARGET_F, p))
        if min_d > p:      # should never happen for p in [6,24], f=4
            return None
        d = rng.randint(min_d, p)
        # 1 <= r <= p 且 最後一個 job 的 release time + execution time <= H
        valid_r = [r for r in range(1, p + 1) if (H - r) % p >= e]
        if not valid_r:
            return None
        r = rng.choice(valid_r)
        w = rng.randint(6, 18)
        preempt = rng.randint(0, 1)
        free_tasks.append({"r": r, "p": p, "e": e, "d": d, "w": w, "preempt": preempt})

    tasks = tight_tasks + free_tasks
    rng.shuffle(tasks)
    for i, t in enumerate(tasks):
        t["id"] = f"p{i + 1}"

    # --- quick pre-checks (cheap, avoid full validation cost) ---
    if len({t["p"] for t in tasks}) < 3:
        return None
    if _count_expanded_jobs(tasks) <= 30: # Level 1-3 評分標準
        return None
    if sum(t["e"] / t["p"] for t in tasks) < 0.7: # Level 1-5 評分標準
        return None

    # --- ensure >=2 tasks with w>=14 (patch in place) ---
    need_w14 = 2 - sum(1 for t in tasks if t["w"] >= 14)
    for t in tasks:
        if need_w14 <= 0:
            break
        if t["w"] < 14:
            t["w"] = rng.randint(14, 18)
            need_w14 -= 1

    return tasks


# ---------------------------------------------------------------------------
# Validation (complete, independent of generator assumptions)
# ---------------------------------------------------------------------------

def validate(tasks: list[dict]) -> list[str]:
    n = len(tasks)
    errs: list[str] = []

    if not (6 <= n <= 10):
        errs.append(f"|Jp|={n} not in [6,10]") # Level 1-2 評分標準
    if not all(6 <= t["p"] <= 24 for t in tasks):
        errs.append("some period outside [6,24]") # Level 1-4-2 評分標準
    if len({t["p"] for t in tasks}) < 3:
        errs.append(f"only {len({t['p'] for t in tasks})} distinct periods (need >=3)")

    expanded = _count_expanded_jobs(tasks)
    if expanded <= 30: 
        errs.append(f"expanded jobs={expanded} not >30") # Level 1-3 評分標準

    for t in tasks:
        tid = t["id"]
        if not (1 <= t["r"] <= t["p"]):
            errs.append(f"{tid}: r={t['r']} not in [1,{t['p']}]") # Level 1-4-1 評分標準
        if not (1 <= t["e"] <= 4):
            errs.append(f"{tid}: e={t['e']} not in [1,4]") # Level 1-4-3 評分標準
        if not (t["e"] <= t["d"] <= t["p"]):
            errs.append(f"{tid}: d={t['d']} not in [{t['e']},{t['p']}]") # Level 1-4-4 評分標準
        if not (6 <= t["w"] <= 18):
            errs.append(f"{tid}: w={t['w']} not in [6,18]") # Level 1-4-5 評分標準
        last_release = t["r"] + ((H - t["r"]) // t["p"]) * t["p"]
        if last_release + t["e"] > H:
            errs.append(f"{tid}: last instance release={last_release}, release+e={last_release + t['e']} > H={H}") # 最後一個 job 的 release time + execution time <= H

    if sum(1 for t in tasks if t["e"] == 2) < 2:
        errs.append("fewer than 2 tasks with e=2") # Level 1-4-3 評分標準
    if not any(t["e"] >= 3 for t in tasks):
        errs.append("no task with e>=3") # Level 1-4-3 評分標準
    if sum(1 for t in tasks if t["w"] >= 14) < 2:
        errs.append("fewer than 2 tasks with w>=14") # Level 1-4-5 評分標準

    dw = sum(t["e"] / t["p"] for t in tasks)
    if dw < 0.7:
        errs.append(f"DW={dw:.4f} < 0.7") # Level 1-5 評分標準

    tight = sum(1 for t in tasks if t["d"] == t["e"])
    req = math.ceil(0.2 * n)
    if tight < req:
        errs.append(f"tight deadlines={tight} < {req} (20% of {n})") # Level 1-6 評分標準

    np_ne1 = sum(1 for t in tasks if t["preempt"] == 0 and t["e"] != 1)
    if np_ne1 < 2:
        errs.append(f"non-preemptive with e!=1: {np_ne1} < 2") # Level 1-7 評分標準

    return errs


# ---------------------------------------------------------------------------
# Top-level generator with retry
# ---------------------------------------------------------------------------

def generate(seed: int | None = None) -> tuple[list[dict], int]:
    rng = random.Random()
    for attempt in range(100_000):
        rng.seed(attempt if seed is None else seed + attempt)
        tasks = _try_generate(rng) # 先設 frame size == 4 ， max_e 寫死為 4
        if tasks is None:
            continue
        if validate(tasks):
            continue
        frame = _find_frame_size(tasks) # 找 frame size ， 但因為 max_e == 4 ， frame size必為4
        if frame is None:
            continue
        return tasks, frame
    raise RuntimeError("Could not generate a valid task set in 100,000 attempts")


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(tasks: list[dict], frame: int) -> None:
    n = len(tasks)
    dw = sum(t["e"] / t["p"] for t in tasks)
    expanded = _count_expanded_jobs(tasks)
    tight_n = sum(1 for t in tasks if t["d"] == t["e"])
    np_ne1 = sum(1 for t in tasks if t["preempt"] == 0 and t["e"] != 1)

    sep = "=" * 68
    print(sep)
    print("  PERIODIC TASK SET  -  VPP 72-hour Scheduling")
    print(sep)
    print(f"  {'ID':<6}  {'r':>4}  {'p':>4}  {'e':>4}  {'d':>4}  {'w':>4}  {'P/NP':>5}  Flags")
    print("-" * 68)
    for t in sorted(tasks, key=lambda x: x["id"]):
        flags = []
        if t["d"] == t["e"]:
            flags.append("tight")
        if t["preempt"] == 0 and t["e"] != 1:
            flags.append("NP,e!=1")
        pnp = "NP" if t["preempt"] == 0 else "P"
        print(
            f"  {t['id']:<6}  {t['r']:>4}  {t['p']:>4}  {t['e']:>4}"
            f"  {t['d']:>4}  {t['w']:>4}  {pnp:>5}  {', '.join(flags)}"
        )
    print("-" * 68)
    print(f"  Tasks (|Jp|):                {n}          [6..10]")
    print(f"  Distinct periods:            {sorted({t['p'] for t in tasks})}  [>=3]")
    print(f"  Workload density DW:         {dw:.4f}     [>=0.70]")
    print(f"  Frame size f:                {frame}          [min valid]")
    print(f"  Expanded jobs (H={H}):       {expanded}         [>30]")
    print(f"  Tasks with e=2:              {sum(1 for t in tasks if t['e'] == 2)}          [>=2]")
    print(f"  Tasks with e>=3:             {sum(1 for t in tasks if t['e'] >= 3)}          [>=1]")
    print(f"  Tasks with w>=14:            {sum(1 for t in tasks if t['w'] >= 14)}          [>=2]")
    print(f"  Tight deadlines (d=e):       {tight_n}/{n} = {tight_n/n*100:.0f}%   [>=20%]")
    print(f"  Non-preemptive e!=1:         {np_ne1}          [>=2]")
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    tasks, frame = generate()

    errs = validate(tasks)
    if errs:
        print("INTERNAL ERROR: post-generation validation failed:", file=sys.stderr)
        for e in errs:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)

    output = {
        "periodic": {
            t["id"]: {
                "r":       t["r"],
                "p":       t["p"],
                "e":       t["e"],
                "d":       t["d"],
                "w":       t["w"],
                "preempt": t["preempt"],
            }
            for t in sorted(tasks, key=lambda x: x["id"])
        }
    }

    out_path = Path(__file__).parent.parent / "output" / "task_set.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)

    print_summary(tasks, frame)
    print(f"\nOutput written to: {out_path}")


if __name__ == "__main__":
    main()
