"""
Constraint verifier for the dynamic (advanced) scheduler.

Reads:
  output/schedule_result_advanced.json
  output/acceptance_test_log_advanced.json
  input/processor_settings.json, input/price_72hr.json,
  input/aperiodic_n_sporadic.json, output/task_set.json

Differences vs verifier.py (static):
  * Renewable upper bound = capacity * pv_actual[t]   (no 0.92 robust margin —
    actuals are realised, not forecasts).
  * day_ahead_commit may exceed sell[t]; the shortfall is captured by the
    per-slot cancellation_penalty, which is itself verified:
        cancellation_penalty[t] ≈ cancel_rate * p_DA[t] * max(0, commit - sell)
  * commitment_le_sell is NOT enforced (that was static-only).
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_verifier():
    print("=" * 60)
    print("  VPP Advanced (Dynamic) Schedule Constraint Verifier")
    print("=" * 60)

    # 1. 讀取資料
    try:
        proc  = load_json(INPUT_DIR / "processor_settings.json")
        price = load_json(INPUT_DIR / "price_72hr.json")
        tasks = load_json(OUTPUT_DIR / "task_set.json")
        demo_path = INPUT_DIR / "aperiodic_n_sporadic.json"
        if demo_path.exists():
            demo = load_json(demo_path)
            tasks["sporadic"]  = demo.get("sporadic", [])
            tasks["aperiodic"] = demo.get("aperiodic", [])
        sched_path = OUTPUT_DIR / "schedule_result_advanced.json"
        sched = load_json(sched_path)["schedule_result"]
        acc_full = load_json(OUTPUT_DIR / "acceptance_test_log_advanced.json")
    except Exception as e:
        print(f"[錯誤] 無法讀取檔案: {e}")
        return

    generators = {g["generator_id"]: g for g in proc["generator"]}
    batteries  = {b["storage_id"]:   b for b in proc["storage"]}
    renewables = {r["renewable_id"]: r for r in proc["renewable_capacity"]}

    # 解析 forecast (備用) — 主上限改用每 slot 的 pv_actual
    forecast = {}
    for entry in proc["renewable_forecast"]:
        for rid, data in entry.items():
            forecast[rid] = {int(d["hour"]): float(d["pv_forecast"]) for d in data}

    # 解析價格 / 取消懲罰率
    price_map   = {int(v["hour"]): float(v["market_price"]) for v in price["price"]}
    cancel_rate = float(price["price"][0].get("cancellation_penalty_rate", 0.0))

    # Tag tasks with their kind (sporadic vs aperiodic share "d" key)
    def _ingest(section_key, kind):
        section = tasks.get(section_key, {})
        if isinstance(section, dict):
            entries = section.items()
        else:
            entries = [(t.get("id", f"{kind[0]}{i}"), t)
                       for i, t in enumerate(section)]
        for tid, t in entries:
            all_tasks[tid] = {**t, "_kind": kind}

    all_tasks = {}
    if "periodic" in tasks:
        for tid, t in tasks["periodic"].items():
            all_tasks[tid] = {**t, "_kind": "periodic"}
    _ingest("sporadic",  "sporadic")
    _ingest("aperiodic", "aperiodic")

    # 新版輸出 schema: 單一 acceptance_test_log list,每筆以 type 區分。
    _acc_entries = acc_full["acceptance_test_log"]
    _sporadic_entries  = [e for e in _acc_entries if e.get("type") == "sporadic"]
    _aperiodic_entries = [e for e in _acc_entries if e.get("type") == "aperiodic"]

    # 被拒絕的 sporadic / 沒排上的 aperiodic 不應該出現在排程裡
    rejected_sporadic = {entry["job_id"] for entry in _sporadic_entries
                         if not entry.get("accepted")}
    skipped_aperiodic = {entry["job_id"] for entry in _aperiodic_entries
                         if not entry.get("accepted")}

    violations = []
    def log_violation(constraint_id, msg):
        violations.append(f"[C{constraint_id}] {msg}")

    H = len(sched)
    EPS = 1e-4

    # 狀態追蹤
    gen_state = {g: {"on_time":  generators[g].get("initial_on_time", 0),
                     "off_time": generators[g].get("initial_off_time", 0),
                     "P_prev":   generators[g].get("initial_energy", 0),
                     "is_on":    generators[g].get("initial_on_time", 0) > 0}
                 for g in generators}

    job_exec_hours = {j: [] for j in all_tasks}

    for t_idx, slot in enumerate(sched):
        t    = slot["t"]
        P    = slot.get("P", {})
        K    = slot.get("k", {})
        Sell = slot.get("sell", 0.0)
        SOC  = slot.get("soc", {})

        # ---------- 系統平衡 ----------
        # [C23] 供需平衡
        total_gen = sum(P.values())
        total_load = 0.0
        for jid, alloc in K.items():
            total_load += sum(alloc.values())
            if jid in job_exec_hours:
                job_exec_hours[jid].append(t)

        if abs(total_gen - (total_load + Sell)) > 0.01:
            log_violation(23, f"t={t} 能量不平衡: 發電 {total_gen:.4f} != 消耗+售電 {total_load + Sell:.4f}")

        # [C22] 售電不可為負
        if Sell < -EPS:
            log_violation(22, f"t={t} 售電量為負: {Sell}")

        # [commitment_nonneg] (commit ≤ sell 不再強制 — 動態模式允許短缺並付 penalty)
        commit_val = slot.get("day_ahead_commit", 0.0)
        if commit_val < -EPS:
            log_violation("commitment_nonneg",
                          f"t={t} day_ahead_commit={commit_val:.4f} < 0")

        # [cancellation_penalty] = cancel_rate × p_DA × max(0, commit - sell)
        p_da = price_map.get(t, 0.0)
        expected_pen = cancel_rate * p_da * max(0.0, commit_val - Sell)
        actual_pen   = float(slot.get("cancellation_penalty", 0.0))
        if abs(actual_pen - expected_pen) > 0.01:
            log_violation("cancellation_penalty",
                          f"t={t} cancellation_penalty={actual_pen:.4f}，"
                          f"預期 {expected_pen:.4f} "
                          f"(rate={cancel_rate}, p_DA={p_da}, "
                          f"shortfall={max(0.0, commit_val - Sell):.4f})")

        # ---------- 任務執行 ----------
        for jid, alloc in K.items():
            if jid.endswith("_chg"):
                continue

            base_task_id = jid.split("_")[0]
            task_info = all_tasks.get(base_task_id)
            if not task_info:
                continue

            # [C1] 能量需求必須等於 w_j
            allocated_energy = sum(alloc.values())
            if abs(allocated_energy - task_info["w"]) > EPS:
                log_violation(1, f"t={t} Job {jid} 能量分配 {allocated_energy} "
                                 f"不等於 w_j {task_info['w']}")

            # [C21] 儲能設備不可供電給其他電池的充電 job
            if jid.endswith("_chg"):
                for src in alloc:
                    if src in batteries:
                        log_violation(21, f"t={t} 電池 {src} 供電給充電任務 {jid}")

            # 動態模式檢查：被拒絕的 sporadic / 沒排上的 aperiodic 不該排上
            if base_task_id in rejected_sporadic:
                log_violation("rejected_executed",
                              f"t={t} 被拒絕的 sporadic {base_task_id} "
                              f"竟出現在 k 分配中")
            if base_task_id in skipped_aperiodic:
                log_violation("skipped_executed",
                              f"t={t} 標記為 skipped 的 aperiodic {base_task_id} "
                              f"竟出現在 k 分配中")

        # ---------- 傳統機組 ----------
        for gid, gen in generators.items():
            p_val   = P.get(gid, 0.0)
            prev_p  = gen_state[gid]["P_prev"]
            is_on   = p_val > EPS
            was_on  = gen_state[gid]["is_on"]

            # [C6] 上下限
            if is_on and (p_val < gen["output_min"] - EPS or p_val > gen["output_max"] + EPS):
                log_violation(6, f"t={t} 機組 {gid} 出力 {p_val} 超出範圍 "
                                 f"[{gen['output_min']}, {gen['output_max']}]")

            # [C7] Ramp 限制
            if p_val - prev_p > gen["ramp_up_rate"] + EPS:
                log_violation(7, f"t={t} 機組 {gid} Ramp-Up 違規: "
                                 f"{prev_p} -> {p_val} (Max {gen['ramp_up_rate']})")
            if prev_p - p_val > gen["ramp_down_rate"] + EPS:
                log_violation(7, f"t={t} 機組 {gid} Ramp-Down 違規: "
                                 f"{prev_p} -> {p_val} (Max {gen['ramp_down_rate']})")

            # [C9, C10] Min up / down time
            if is_on and not was_on:
                if 0 < gen_state[gid]["off_time"] < gen["min_down_time"]:
                    log_violation(10, f"t={t} 機組 {gid} 違反 Min Down Time "
                                      f"(僅關機 {gen_state[gid]['off_time']}h)")
                gen_state[gid]["on_time"]  = 1
                gen_state[gid]["off_time"] = 0
            elif not is_on and was_on:
                if 0 < gen_state[gid]["on_time"] < gen["min_up_time"]:
                    log_violation(9, f"t={t} 機組 {gid} 違反 Min Up Time "
                                     f"(僅開機 {gen_state[gid]['on_time']}h)")
                gen_state[gid]["off_time"] = 1
                gen_state[gid]["on_time"]  = 0
            else:
                if is_on:
                    gen_state[gid]["on_time"]  += 1
                else:
                    gen_state[gid]["off_time"] += 1

            gen_state[gid]["P_prev"] = p_val
            gen_state[gid]["is_on"]  = is_on

        # ---------- 再生能源 ----------
        # 動態模式：上限是「實際出力 pv_actual」(stored per-slot)
        pv_actual_slot = slot.get("pv_actual", {})
        for rid, ren in renewables.items():
            p_val   = P.get(rid, 0.0)
            act_fc  = float(pv_actual_slot.get(rid,
                            forecast.get(rid, {}).get(t, 0.0)))
            max_p   = ren["capacity"] * act_fc
            # [C13_actual] 實際 PV 上限
            if p_val > max_p + EPS:
                log_violation("13_actual",
                              f"t={t} 再生能源 {rid} 出力 {p_val:.4f} "
                              f"超過實際上限 {max_p:.4f} "
                              f"(capacity={ren['capacity']}, pv_actual={act_fc})")

        # ---------- 儲能 ----------
        for bid, bat in batteries.items():
            dis_val = P.get(bid, 0.0)
            chg_val = sum(alloc.get(src, 0.0)
                          for jid, alloc in K.items()
                          if jid == f"{bid}_chg"
                          for src in alloc)

            # [C19] 充放電互斥
            if dis_val > EPS and chg_val > EPS:
                log_violation(19, f"t={t} 電池 {bid} 同時充放電 "
                                  f"(Chg: {chg_val}, Dis: {dis_val})")

            # [C14, C15] 充放電上限
            if dis_val > bat["discharge_max"] + EPS:
                log_violation(14, f"t={t} 電池 {bid} 放電 {dis_val} "
                                  f"超過上限 {bat['discharge_max']}")
            if chg_val > bat["charge_max"] + EPS:
                log_violation(15, f"t={t} 電池 {bid} 充電 {chg_val} "
                                  f"超過上限 {bat['charge_max']}")

            # [C17] SOC 上下限
            soc_val = SOC.get(bid, 0.0)
            if soc_val < bat["soc_min"] - EPS or soc_val > bat["soc_max"] + EPS:
                log_violation(17, f"t={t} 電池 {bid} SOC {soc_val} 越界 "
                                  f"[{bat['soc_min']}, {bat['soc_max']}]")

            # [C16_L2] SOC 動態 (含 efficiency 與 self-discharge)
            eta_c = float(bat.get("charge_efficiency",    1.0))
            eta_d = float(bat.get("discharge_efficiency", 1.0))
            sigma = float(bat.get("self_discharge_rate",  0.0))
            prev_soc = (sched[t_idx - 1]["soc"].get(bid, bat["soc_init"])
                        if t > 1 else bat["soc_init"])
            expected_soc = prev_soc * (1 - sigma) + chg_val * eta_c - dis_val / eta_d
            if abs(soc_val - expected_soc) > 0.05:
                log_violation("16_L2", f"t={t} 電池 {bid} SOC 追蹤異常: "
                                       f"預期 {expected_soc:.4f}, 實際 {soc_val:.4f}")

            # [C14_SOC_dep] SOC-dependent 放電上限
            threshold_soc = 0.3 * bat["soc_max"]
            if prev_soc < threshold_soc - EPS:
                limit = bat["discharge_max"] * (prev_soc / threshold_soc)
                if dis_val > limit + 0.01:
                    log_violation("14_SOC_dep",
                                  f"t={t} 電池 {bid} 放電 {dis_val:.4f} "
                                  f"超過 SOC 相依上限 {limit:.4f}")

    # ---------------------------------------------------------
    # 任務全域檢查 (Non-preemptive & Deadlines)
    # ---------------------------------------------------------
    # 動態模式：sporadic 也是用「相對 deadline」(r + d - 1)，與 static 一致。
    for jid, hours in job_exec_hours.items():
        if not hours:
            continue
        base_id   = jid.split("_")[0]
        task_info = all_tasks.get(base_id)
        if not task_info:
            continue

        r_time = task_info.get("r", task_info.get("release", 0))
        period = task_info.get("p")
        e_time = task_info.get("e")
        if period is not None:
            d_rel = task_info.get("d")
            d_abs = None
        else:
            d_rel = None
            if "d" in task_info:
                d_abs = min(int(r_time) + int(task_info["d"]) - 1, H)
            else:
                d_abs = (task_info.get("hard_deadline")
                         or task_info.get("soft_deadline"))
        is_non_preemptive = (task_info.get("preempt") == 0)

        # 依週期分群
        instances = {}
        if period is not None and period > 0:
            for t in hours:
                k = (t - r_time) // period
                instances.setdefault(k, []).append(t)
        else:
            instances[0] = hours

        for k, inst_hours in instances.items():
            inst_hours.sort()
            start_time = inst_hours[0]
            end_time   = inst_hours[-1]
            job_name   = f"{jid} (第 {k} 週期)" if period else jid

            # [C5] Non-preemptive 連續性
            if is_non_preemptive and (end_time - start_time + 1) != len(inst_hours):
                log_violation(5, f"Job {job_name} Non-preemptive 但時間段不連續: "
                                 f"{inst_hours}")

            # 執行長度
            if e_time is not None and len(inst_hours) != e_time:
                log_violation(5, f"Job {job_name} 執行時數異常: "
                                 f"實際 {len(inst_hours)}h，應為 {e_time}h")

            # Deadline 檢查
            if period and d_rel is not None:
                abs_deadline = r_time + (k * period) + d_rel - 1
                if end_time > abs_deadline:
                    log_violation(5, f"Job {job_name} 違反 Deadline: 執行至 "
                                     f"t={end_time}，超出絕對期限 {abs_deadline}")
            elif (not period) and d_abs is not None:
                if end_time > d_abs:
                    # Aperiodic = soft → 不算違規 (tardiness 由 evaluator 報告)
                    if (task_info.get("_kind") == "aperiodic"
                            or task_info.get("soft_deadline") is not None):
                        pass
                    else:
                        log_violation(5, f"Job {job_name} 違反 Hard Deadline: "
                                         f"執行至 t={end_time}，"
                                         f"超出絕對期限 {d_abs}")

    # ---------------------------------------------------------
    # 額外動態檢查：acceptance log 與排程一致性
    # ---------------------------------------------------------
    # 注意：rolling replan 可能把 accepted sporadic / placed aperiodic 搬到
    # 與初次 acceptance log 不同的 slots (仍在 [r, d_abs] 內)，所以這裡
    # 只檢查「該執行的有沒有執行 / 不該執行的有沒有偷跑」，不檢查 slot 精準對應。

    # 已接受的 sporadic 必須出現在排程
    for entry in _sporadic_entries:
        if not entry.get("accepted"):
            continue
        jid = entry["job_id"]
        if not job_exec_hours.get(jid):
            log_violation("acc_missing",
                          f"已接受的 sporadic {jid} 未出現在排程中")

    # 排上的 aperiodic 必須出現在排程
    for entry in _aperiodic_entries:
        if not entry.get("accepted"):
            continue
        jid = entry["job_id"]
        if not job_exec_hours.get(jid):
            log_violation("ap_missing",
                          f"標記為已排上的 aperiodic {jid} 未出現在排程中")

    # ---------------------------------------------------------
    # 報告
    # ---------------------------------------------------------
    if not violations:
        print("\n 驗證通過！動態排程結果完美符合所有 Constraints。")
    else:
        print(f"\n 發現 {len(violations)} 項排程違規:\n")
        for v in violations:
            print(v)
    print("=" * 60)


if __name__ == "__main__":
    run_verifier()
