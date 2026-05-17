import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_verifier():
    print("=" * 60)
    print("  VPP Schedule Constraint Verifier")
    print("=" * 60)

    # 1. 讀取資料
    try:
        proc = load_json(INPUT_DIR / "processor_settings.json")
        tasks = load_json(OUTPUT_DIR / "task_set.json")
        sched = load_json(OUTPUT_DIR / "schedule_result.json")["schedule_result"]
    except Exception as e:
        print(f"[錯誤] 無法讀取檔案: {e}")
        return

    generators = {g["generator_id"]: g for g in proc["generator"]}
    batteries = {b["storage_id"]: b for b in proc["storage"]}
    renewables = {r["renewable_id"]: r for r in proc["renewable_capacity"]}
    
    # 解析預測資料
    forecast = {}
    for entry in proc["renewable_forecast"]:
        for rid, data in entry.items():
            forecast[rid] = {int(d["hour"]): float(d["pv_forecast"]) for d in data}

    all_tasks = {}
    if "periodic" in tasks: all_tasks.update(tasks["periodic"])
    if "sporadic" in tasks: 
        if isinstance(tasks["sporadic"], dict):
            all_tasks.update(tasks["sporadic"])
        else:
            all_tasks.update({t.get("id", f"s{i}"): t for i, t in enumerate(tasks["sporadic"])})

    violations = []
    def log_violation(constraint_id, msg):
        violations.append(f"[C{constraint_id}] {msg}")

    H = len(sched)
    EPS = 1e-4

    # 狀態追蹤
    gen_state = {g: {"on_time": generators[g].get("initial_on_time", 0),
                     "off_time": generators[g].get("initial_off_time", 0),
                     "P_prev": generators[g].get("initial_energy", 0),
                     "is_on": generators[g].get("initial_on_time", 0) > 0} 
                 for g in generators}
    
    job_exec_hours = {j: [] for j in all_tasks}
    
    for t_idx, slot in enumerate(sched):
        t = slot["t"]
        P = slot.get("P", {}) # 設備的充放電量
        K = slot.get("k", {}) # 能量的分配明細
        Sell = slot.get("sell", 0.0)
        SOC = slot.get("soc", {}) # 電池的電量

        # ---------------------------------------------------------
        # 系統平衡檢查
        # ---------------------------------------------------------
        # [C23] 供需平衡: 總發電 = 總負載 + 充電 + 售電
        total_gen = sum(P.values())
        total_load = 0
        for jid, alloc in K.items():
            total_load += sum(alloc.values())
            # 記錄 Job 執行時間
            if jid in job_exec_hours:
                job_exec_hours[jid].append(t)

        if abs(total_gen - (total_load + Sell)) > EPS:
            log_violation(23, f"t={t} 能量不平衡: 發電 {total_gen:.2f} != 消耗+售電 {total_load + Sell:.2f}")

        # [C22] 售電不可為負
        if Sell < -EPS:
            log_violation(22, f"t={t} 售電量為負: {Sell}")

        # ---------------------------------------------------------
        # 任務執行檢查 (Job Execution)
        # ---------------------------------------------------------
        for jid, alloc in K.items():
            if jid.endswith("_chg"): continue
            
            # 找到基礎任務定義 (處理 p1_k0 這種展開後的 ID)
            base_task_id = jid.split("_")[0]
            task_info = all_tasks.get(base_task_id)
            if not task_info: continue

            # [C1] 能量需求必須等於 w_j
            allocated_energy = sum(alloc.values())
            if abs(allocated_energy - task_info["w"]) > EPS:
                log_violation(1, f"t={t} Job {jid} 能量分配 {allocated_energy} 不等於 w_j {task_info['w']}")

            # [C21] 儲能設備的電不能拿去充電 (C21在資料結構上由chg_jobs隔離，此處檢查確保沒有battery_X供電給_chg)
            if jid.endswith("_chg"):
                for src in alloc:
                    if src in batteries:
                        log_violation(21, f"t={t} 電池 {src} 供電給充電任務 {jid}")

        # ---------------------------------------------------------
        # 傳統機組檢查 (Thermal Generators)
        # ---------------------------------------------------------
        for gid, gen in generators.items():
            p_val = P.get(gid, 0.0)
            prev_p = gen_state[gid]["P_prev"]
            is_on = p_val > EPS
            was_on = gen_state[gid]["is_on"]

            # [C6] 上下限
            if is_on and (p_val < gen["output_min"] - EPS or p_val > gen["output_max"] + EPS):
                log_violation(6, f"t={t} 機組 {gid} 出力 {p_val} 超出範圍 [{gen['output_min']}, {gen['output_max']}]")

            # [C7] 升降載率
            if p_val - prev_p > gen["ramp_up_rate"] + EPS:
                log_violation(7, f"t={t} 機組 {gid} Ramp-Up 違規: {prev_p} -> {p_val} (Max {gen['ramp_up_rate']})")
            if prev_p - p_val > gen["ramp_down_rate"] + EPS:
                log_violation(7, f"t={t} 機組 {gid} Ramp-Down 違規: {prev_p} -> {p_val} (Max {gen['ramp_down_rate']})")

            # 狀態轉移與 [C9, C10] 起停時間
            if is_on and not was_on:
                if gen_state[gid]["off_time"] < gen["min_down_time"] and gen_state[gid]["off_time"] > 0:
                     log_violation(10, f"t={t} 機組 {gid} 違反 Min Down Time (僅關機 {gen_state[gid]['off_time']}h)")
                gen_state[gid]["on_time"] = 1
                gen_state[gid]["off_time"] = 0
            elif not is_on and was_on:
                if gen_state[gid]["on_time"] < gen["min_up_time"] and gen_state[gid]["on_time"] > 0:
                     log_violation(9, f"t={t} 機組 {gid} 違反 Min Up Time (僅開機 {gen_state[gid]['on_time']}h)")
                gen_state[gid]["off_time"] = 1
                gen_state[gid]["on_time"] = 0
            else:
                if is_on: gen_state[gid]["on_time"] += 1
                if not is_on: gen_state[gid]["off_time"] += 1
            
            gen_state[gid]["P_prev"] = p_val
            gen_state[gid]["is_on"] = is_on

        # ---------------------------------------------------------
        # 再生能源檢查 (Renewables)
        # ---------------------------------------------------------
        for rid, ren in renewables.items():
            p_val = P.get(rid, 0.0)
            max_p = ren["capacity"] * forecast[rid].get(t, 0)
            # [C13] 預測上限
            if p_val > max_p + EPS:
                log_violation(13, f"t={t} 再生能源 {rid} 出力 {p_val} 超過預測上限 {max_p}")

        # ---------------------------------------------------------
        # 儲能設備檢查 (Batteries)
        # ---------------------------------------------------------
        for bid, bat in batteries.items():
            # 計算當下充放電量
            dis_val = P.get(bid, 0.0)
            chg_val = sum(alloc.get(src, 0.0) for jid, alloc in K.items() if jid == f"{bid}_chg" for src in alloc)
            
            # [C19] 互斥檢查
            if dis_val > EPS and chg_val > EPS:
                log_violation(19, f"t={t} 電池 {bid} 同時充放電 (Chg: {chg_val}, Dis: {dis_val})")

            # [C14, C15] 充放電上限
            if dis_val > bat["discharge_max"] + EPS:
                log_violation(14, f"t={t} 電池 {bid} 放電 {dis_val} 超過上限 {bat['discharge_max']}")
            if chg_val > bat["charge_max"] + EPS:
                log_violation(15, f"t={t} 電池 {bid} 充電 {chg_val} 超過上限 {bat['charge_max']}")

            # [C17] SOC 上下限
            soc_val = SOC.get(bid, 0.0)
            if soc_val < bat["soc_min"] - EPS or soc_val > bat["soc_max"] + EPS:
                log_violation(17, f"t={t} 電池 {bid} SOC {soc_val} 越界 [{bat['soc_min']}, {bat['soc_max']}]")

            # [C16] SOC 動態更新檢查 (需依賴上一個時段，為簡化這裡直接用排程回傳的數值檢查邏輯連續性)
            if t > 1:
                prev_soc = sched[t_idx-1].get("soc", {}).get(bid, bat["soc_init"])
                expected_soc = prev_soc + chg_val - dis_val
                if abs(soc_val - expected_soc) > EPS:
                    log_violation(16, f"t={t} 電池 {bid} SOC 追蹤異常: 預期 {expected_soc}, 實際 {soc_val}")

    
    # ---------------------------------------------------------
    # 任務全域檢查 (Non-preemptive & Deadlines)
    # ---------------------------------------------------------
    for jid, hours in job_exec_hours.items():
        if not hours: continue
        base_id = jid.split("_")[0]
        task_info = all_tasks.get(base_id)
        if not task_info: continue

        # 取得任務的基本參數
        r_time = task_info.get("r", 0)  # 到達時間
        period = task_info.get("p")     # 週期
        e_time = task_info.get("e")     # 預期執行時間
        d_rel = task_info.get("d") or task_info.get("deadline") # 相對 deadline
        is_non_preemptive = (task_info.get("preempt") == 0)

        # 1. 將時間點依照「週期」分群 (針對週期性任務)
        instances = {}
        if period is not None and period > 0:
            for t in hours:
                # 計算該時間點屬於第幾個週期 (k=0, 1, 2...)
                k = (t - r_time) // period
                if k not in instances:
                    instances[k] = []
                instances[k].append(t)
        else:
            # 若不是週期性任務(Sporadic)，預設視為同一個 Instance
            instances[0] = hours

        # 2. 針對每個週期 (Instance) 獨立檢查
        for k, inst_hours in instances.items():
            inst_hours.sort()
            start_time = inst_hours[0]
            end_time = inst_hours[-1]

            # 為了報錯顯示清楚，標示出是第幾個週期
            job_name = f"{jid} (第 {k} 週期)" if period else jid

            # [C5] Non-preemptive 連續性檢查
            if is_non_preemptive:
                if (end_time - start_time + 1) != len(inst_hours):
                    log_violation(5, f"Job {job_name} 為 Non-preemptive，但時間段不連續: {inst_hours}")

            # 總執行長度檢查
            if e_time is not None and len(inst_hours) != e_time:
                log_violation(5, f"Job {job_name} 執行時數異常: 實際 {len(inst_hours)} 小時，應為 {e_time} 小時")

            # Deadline 檢查
            if d_rel is not None:
                # 該週期的絕對 Deadline = 初始到達時間 + (k * 週期) + 相對Deadline
                abs_deadline = r_time + (k * period) + d_rel if period else r_time + d_rel
                
                # 假設 t=19 代表在 19~20 執行，結束於 20。若 abs_deadline 是 20，則 t=19 是合法的。
                # 所以如果 end_time >= abs_deadline (例如 t=20, 代表 20~21執行)，就代表違規。
                if end_time >= abs_deadline:
                    log_violation(5, f"Job {job_name} 違反 Deadline: 執行至 t={end_time}，超出絕對期限 {abs_deadline}")

    # ---------------------------------------------------------
    # 輸出報告
    # ---------------------------------------------------------
    if not violations:
        print("\n 驗證通過！排程結果完美符合所有 Constraints。")
    else:
        print(f"\n 發現 {len(violations)} 項排程違規:\n")
        for v in violations:
            print(v)
    print("=" * 60)

if __name__ == "__main__":
    run_verifier()