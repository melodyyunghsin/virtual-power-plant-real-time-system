# VPP Real-Time Scheduling System

## Project Structure
Base path: C:\Users\chenh\OneDrive\桌面\claude cowork\Real time system\

├── CLAUDE.md
├── README.md
├── report.pdf
├── src/
│   ├── task_generator.py
│   ├── scheduler.py
│   └── evaluator.py
├── input/
│   ├── processor_settings.json
│   └── price_72hr.json
└── output/
    ├── task_set.json
    ├── schedule_result.json
    ├── evaluation_results.json
    └── acceptance_test_log.json

## Language & Dependencies
- Python 3.10+
- PuLP (ILP solver)
- json, math standard libraries

## System Overview
72-hour VPP scheduling. Time index t = 1..72, Δt = 1 hour.
Processors supply energy to jobs. Three job types:
- Periodic (hard deadline, repeat by period)
- Sporadic (hard deadline, acceptance test required)
- Aperiodic (soft deadline, waiting queue)

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
- Cannot charge and discharge simultaneously (big-M constraint required)

## Key Constraints to Always Enforce
1. Energy balance every hour: sum(P_i,t) = sum(k_j,i,t) + sum(charging) + Sell_t
2. Non-preemptive jobs must execute consecutively (no gaps)
3. Sporadic jobs: acceptance test only, cannot move existing periodic jobs
4. Aperiodic jobs: soft deadline, queue-based, record miss + tardiness
5. Battery simultaneous charge/discharge = 0 (use binary variable + big-M)
6. Generator ramp up/down limits apply between consecutive hours
7. Generator min up/down time must be respected

## Output Formats
All outputs must be valid JSON (no comments, no trailing commas).
Key files: task_set.json, schedule_result.json, evaluation_results.json, acceptance_test_log.json
See spec for exact field names — do not rename required fields.

## Task Generator Constraints (task_generator.py)
- 6 ≤ |Jp| ≤ 10
- Expanded periodic jobs > 30
- 3+ distinct period values, all periods in [6, 24]
- 1 ≤ r_j ≤ period_j
- 1 ≤ e_j ≤ 4, at least 2 tasks with e=2, at least 1 with e≥3
- e_j ≤ deadline_j ≤ period_j
- 6 ≤ w_j ≤ 18, at least 2 tasks with w≥14
- Workload density DW = sum(e/p) must satisfy 0.7 ≤ DW
- At least 20% of tasks: deadline = e (tight deadline)
- At least 2 non-preemptive tasks with e ≠ 1
- Frame size f: f ≥ max(e), H mod f = 0, 2f - gcd(f,p) ≤ deadline for all tasks

## Scheduler Notes (scheduler.py)
- Use PuLP for ILP. Binary variables needed for: generator on/off, battery charge/discharge mutex.
- big-M = 10000 for mutex constraints.
- Penalty for aperiodic miss: α = 10000 $/miss
- Objective: min α*f1 + f2 + f3 where f3 = -sum(λ_t * Sell_t)