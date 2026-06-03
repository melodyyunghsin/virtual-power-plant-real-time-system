# VPP Real-Time Scheduling System

虛擬電廠 (Virtual Power Plant, VPP) 72 小時即時排程系統，分為 Level 1 日前靜態排程 與 Level 2 動態滾動排程兩階段。

---

## 1. 使用語言、版本與套件需求

- **Python**: 3.10 以上 (本專案於 Python 3.14.4 測試)
- **套件**:
  - `pulp >= 3.0` (使用內建 CBC solver，不需額外安裝)
- **作業系統**: macOS / Linux / Windows 皆可

無其他外部套件需求；`json`、`math`、`random`、`copy`、`pathlib` 等皆為 Python 標準函式庫。

---

## 2. 程式編譯方式與環境設定

本專案為純 Python，**無需編譯**。建議使用虛擬環境:

```bash
# 建立虛擬環境
python3 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

# 安裝相依套件
pip install pulp
```

---

## 3. 程式執行流程

依下列順序執行四個主程式即可重現所有提交的 JSON 輸出。所有指令請在專案根目錄下執行。

```bash
# Step 1: 產生 periodic task set
python3 src/task_generator.py
# 產出: output/task_set.json

# Step 2: Level 1 日前靜態排程 (Phase 1 ILP + online_phase)
python3 src/scheduler.py
# 產出: output/schedule_result.json
#       output/acceptance_test_log.json

# Step 3: Level 1 效能評估
python3 src/evaluator.py
# 產出: output/evaluation_results.json

# Step 4: Level 2 進階動態排程 (滾動視窗重排)
python3 src/advanced_scheduler.py
# 產出: output/schedule_result_advanced.json
#       output/acceptance_test_log_advanced.json
#       output/evaluation_results_advanced.json
```


---

## 4. 各程式輸入與輸出檔案說明

### 4.1 src/task_generator.py — Periodic task set 產生程式

| 輸出 | 說明 |
| --- | --- |
| `output/task_set.json` | Periodic task 集合 (含 r, p, e, d, w, preempt) |

執行範例:
```bash
python3 src/task_generator.py
```

### 4.2 src/scheduler.py — Level 1 日前靜態排程

| 輸入 | 說明 |
| --- | --- |
| `input/processor_settings.json` | 發電設備 (傳統機組 / 再生能源 / 儲能設備) 參數表 |
| `input/price_72hr.json` | 72 小時市場售電價格 |
| `input/aperiodic_n_sporadic.json` | Demo 時提供的 sporadic + aperiodic 工作清單 |
| `output/task_set.json` | 由 task_generator 產生的 periodic task set |

| 輸出 | 說明 |
| --- | --- |
| `output/schedule_result.json` | 72 小時日前固定排程，含每小時的 P、k、sell、soc 等欄位 |
| `output/acceptance_test_log.json` | Sporadic 接受/拒絕紀錄 + aperiodic 安排紀錄 |

執行流程:
- **Phase 1**: 使用 PuLP CBC 解 ILP，僅處理 periodic tasks。目標函式為 `min α·f1 + f2 + f3`。
- **Online Phase**: 依 release time 排序，逐一處理 sporadic (acceptance test) 與 aperiodic (force-place per C4)。每個工作用 `_find_min_cost` 找出能量成本最低的時段，並以「PV → 電池 → 機組 → sell」的優先序提交。電池放電會自動進行 SOC 補償 (在未來非充電時段增加 chg、減少 sell)，以維持 SOC 軌跡符合 C17 限制。

### 4.3 src/advanced_scheduler.py — Level 2 動態排程

| 輸入 | 說明 |
| --- | --- |
| `output/schedule_result.json` | 以 Level 1 排程為起點 |
| `input/processor_settings.json` | 含 L2 額外欄位: `pv_actual`、`forecast_error_std`、`charge_efficiency`、`discharge_efficiency`、`self_discharge_rate`、`aging_cost` |
| `input/price_72hr.json` | 含 L2 額外欄位: `realtime_price_factor`、`cancellation_penalty_rate` |
| `input/aperiodic_n_sporadic.json` | 同 Level 1 |

| 輸出 | 說明 |
| --- | --- |
| `output/schedule_result_advanced.json` | 動態執行 72 小時後的實際排程 |
| `output/acceptance_test_log_advanced.json` | Sporadic / aperiodic 線上處理紀錄 |
| `output/evaluation_results_advanced.json` | L2 評估結果 + 與 L1 比較 (`vs_static` block) |

執行流程:
- 每小時揭露 `pv_actual` 與即時電價，並在以下事件觸發 12 小時滾動視窗 ILP:
  - PV 實際出力較預測下降超過 7%
  - Sporadic / aperiodic 工作到達
  - 每 6 小時例行刷新
- 滾動 ILP 只重新優化 dispatch (P, sell, gen on/off, battery)；工作的時段分配自到達時即鎖定。
- 每小時結算時若實際 sell 低於 day-ahead 承諾，累積取消售電 penalty。

### 4.4 src/evaluator.py — Level 1 效能評估

| 輸入 | 說明 |
| --- | --- |
| `output/schedule_result.json` | Level 1 排程結果 |
| `output/acceptance_test_log.json` | Sporadic + aperiodic 紀錄 |
| `input/aperiodic_n_sporadic.json` | Demo 工作清單 (計算 sporadic_value_rate) |
| `input/price_72hr.json` | 市場價格 (計算 market_revenue) |

| 輸出 | 說明 |
| --- | --- |
| `output/evaluation_results.json` | 評估指標: miss rate、tardiness、response time、jitter、objective value 等 |

### 4.4 src/evaluator_advanced.py — Level 2 效能評估

| 輸入 | 說明 |
| --- | --- |
| `output/schedule_result_advanced.json` | Level 2 排程結果 |
| `output/acceptance_test_log_advanced.json` | Sporadic + aperiodic 紀錄 |
| `input/aperiodic_n_sporadic.json` | Demo 工作清單 (計算 sporadic_value_rate) |
| `input/price_72hr.json` | 市場價格 (計算 market_revenue) |

| 輸出 | 說明 |
| --- | --- |
| `output/evaluation_results_advanced.json` | 評估指標: miss rate、tardiness、response time、jitter、objective value 等 |
---

## 5. 如何重現繳交的 output JSON

```bash
cd virtual-power-plant-real-time-system

# (建議) 啟用虛擬環境
source venv/bin/activate

# 依序執行
python3 src/task_generator.py
python3 src/scheduler.py
python3 src/evaluator.py
python3 src/advanced_scheduler.py

# 驗證
python3 src/verifier.py
python3 src/verifier_advanced.py
```

執行後 `output/` 目錄將包含七個 JSON 檔案:

```
output/
├── task_set.json
├── schedule_result.json
├── acceptance_test_log.json
├── evaluation_results.json
├── schedule_result_advanced.json
├── acceptance_test_log_advanced.json
└── evaluation_results_advanced.json
```

### 注意事項

- **`task_generator.py` 為隨機產生**: 每次執行會產生不同的合法 task set。若要重現繳交版本的 `task_set.json`，請直接使用 `output/task_set.json`，不要重新執行 `task_generator.py`。
- **CBC solver**: PuLP 內建，無需額外安裝。
- **L1 與 L2 為前後相依**: `advanced_scheduler.py` 會讀取 `schedule_result.json` 當作初始計劃，因此 L2 執行前必須先跑過 `scheduler.py`。

---

## 6. 專案結構

```
virtual-power-plant-real-time-system/
├── README.md                       # 本檔案
├── report.pdf                      # 報告文件
├── src/
│   ├── task_generator.py           # Periodic task set 產生
│   ├── scheduler.py                # Level 1 日前排程 (Phase 1 ILP + online_phase)
│   ├── evaluator.py                # Level 1 效能評估
│   ├── advanced_scheduler.py       # Level 2 動態排程
│   └── evaluator_advanced.py       # Level 2 效能評估 
├── input/
│   ├── processor_settings.json     # 發電設備參數
│   ├── price_72hr.json             # 市場售電價格
│   └── aperiodic_n_sporadic.json   # Demo 時提供的工作清單
└── output/
    ├── task_set.json
    ├── schedule_result.json
    ├── acceptance_test_log.json
    ├── evaluation_results.json
    ├── schedule_result_advanced.json
    ├── acceptance_test_log_advanced.json
    └── evaluation_results_advanced.json
```
