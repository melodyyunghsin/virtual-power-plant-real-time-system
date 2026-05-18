# VPP Real-Time Scheduling System

## Project Structure
```
├── CLAUDE.md
├── README.md
├── report.pdf
├── src/
│   ├── task_generator.py
│   ├── scheduler.py
│   ├── evaluator.py
│   └── verifier.py
├── input/
│   ├── processor_settings.json
│   ├── price_72hr.json
│   └── sporadic_aperiodic_demo.json   # optional, demo-time
└── output/
    ├── task_set.json
    ├── schedule_result.json
    ├── evaluation_results.json
    └── acceptance_test_log.json
```

## Language & Dependencies
- Python 3.10+
- PuLP (CBC solver, bundled)
- json, math standard libraries

## System Overview
72-hour VPP scheduling. Time index t = 1..72, Δt = 1 hour.
Processors supply energy to jobs. Three job types:
- Periodic   — hard deadline, repeats by period; included in Phase 1 ILP
- Sporadic   — hard deadline, online acceptance test in Phase 2
- Aperiodic  — soft deadline, EDF queue post-processed in Phase 3

## Pipeline (scheduler.py)
1. **Phase 1 — Day-ahead static ILP** (PuLP): schedules periodic jobs only.
   Objective: `min α·f1 + f2 + f3` (α=10000, f1=aperiodic miss count,
   f2=generator cost, f3=-market revenue).
2. **Phase 2 — Sporadic acceptance test**: each arriving sporadic uses
   `SlackAbsorber` to check slack in [release, hard_deadline] (contiguous
   for non-preemptive). Mutates schedule in place.
3. **Phase 3 — Aperiodic EDF queue**: each aperiodic tries on-time placement
   in [release, soft_deadline]; falls back to late placement in [release, H].

## Processor Summary (from processor_settings.json)
Generators:
- thermal_1: min=15, max=80 MW, ramp=15 MW/h, UT=3, DT=2, cost_fixed=1200$/h, cost_var=42$/MWh, initially OFF
- thermal_2: min=10, max=45 MW, ramp=20 MW/h, UT=2, DT=2, cost_fixed=600$/h, cost_var=70$/MWh, initially OFF

Renewables:
- pv_1: capacity=60 MW (forecast in processor_settings.json)
- pv_2: capacity=80 MW (same forecast profile as pv_1)
- Available mainly hour 7-18, 31-43, 54-67 (solar windows)

Storage:
- battery_1: soc_min=20, soc_max=100 MWh, dis=20, chg=20 MW/h, soc_init=45
- battery_2: soc_min=10, soc_max=60 MWh, dis=15, chg=15 MW/h, soc_init=25
- Cannot charge and discharge simultaneously (mutex via binary v_chg)

## Reservation Strategy (Section 6)
`RESERVE_PER_GEN = 5` MW: each ON generator must leave 5 MW headroom below
output_max. Provides spinning reserve that Phase 2 can absorb without
violating ramp limits. Tunable trade-off knob (higher = more sporadic
absorption capacity, but lower revenue / higher cost).

## Key Constraints to Always Enforce (C1–C23 from spec §1.3)
1. Energy balance every hour (C23): sum(P_i,t) = sum(k_j,i,t) + sum(charging) + Sell_t
2. Non-preemptive jobs must execute consecutively (C5)
3. Sporadic acceptance test must not break existing periodic schedule
4. Aperiodic must execute e hours by H (C4 third bullet — even if soft deadline missed)
5. Battery simultaneous charge/discharge = 0 (C19, via v_chg binary)
6. Battery cannot charge other batteries (C21 — Kchg restricted to gens ∪ PVs)
7. Generator ramp up/down limits (C7), min up/down time (C9–C10)
8. C11/C12: initial-state UT/DT carry-over when TN/TF < UT/DT

## Output Formats
All outputs must be valid JSON (no comments, no trailing commas).
Required field names from spec §3.4 must not be renamed.
- task_set.json, schedule_result.json, evaluation_results.json,
  acceptance_test_log.json (latter contains both `acceptance_test_log`
  and `aperiodic_log` keys).

## Task Generator Constraints (task_generator.py)
- 6 ≤ |Jp| ≤ 10
- Expanded periodic jobs > 30
- 3+ distinct period values, all periods in [6, 24]
- 1 ≤ r_j ≤ period_j
- 1 ≤ e_j ≤ 4, at least 2 tasks with e=2, at least 1 with e≥3
- e_j ≤ deadline_j ≤ period_j
- 6 ≤ w_j ≤ 18, at least 2 tasks with w≥14
- Workload density DW = sum(e/p) ≥ 0.7
- At least 20% of tasks: deadline = e (tight deadline)
- At least 2 non-preemptive tasks with e ≠ 1
- Frame size f: f ≥ max(e), H mod f = 0, 2f - gcd(f,p) ≤ deadline for all tasks

## Verifier (verifier.py)
Independent constraint checker that re-reads schedule_result.json and
re-validates C1, C5–C7, C9–C10, C13–C19, C21–C23. Reads demo input file
so sporadic/aperiodic constraints are also covered.

## Run Order
```
python3 src/task_generator.py     # → output/task_set.json
python3 src/scheduler.py          # → schedule_result.json + acceptance_test_log.json
python3 src/evaluator.py          # → evaluation_results.json
python3 src/verifier.py           # sanity check, prints PASS/violations
```