# Level 2 實作變更摘要

## 新增檔案

- `src/advanced_scheduler.py` — Event-Triggered Rolling Horizon 動態排程器

---

## 修改檔案

### input/processor_settings.json

每個儲能設備新增四個欄位：
- `charge_efficiency: 0.95` — 充電效率 η_c
- `discharge_efficiency: 0.92` — 放電效率 η_d
- `self_discharge_rate: 0.002` — 每小時自放電率
- `aging_cost: 8.0` — 每 MWh 放電的電池老化成本（$/MWh）

每個再生能源預測條目新增兩個欄位：
- `forecast_error_std: 0.08` — 預測誤差標準差
- `pv_actual` — 模擬的實際出力（seed=123 生成）

### input/price_72hr.json

每個時段新增兩個欄位：
- `realtime_price_factor` — 即時市場價格倍率（seed=42 生成，範圍 0.80–1.39）
- `cancellation_penalty_rate: 0.3` — 售電取消懲罰率

### src/scheduler.py（Phase 1 ILP）

新增三項 Level 2 限制式：

**Assumption II — 儲能真實運作**
- SOC 更新方程改為：`SOC[t] = SOC[t-1] × (1−σ) + charge × η_c − discharge / η_d`
- 新增 SOC-dependent discharge limit：SOC < 30% 時放電功率線性降額
- 目標函數 f2 加入 aging cost 項

**Assumption I — 再生能源不確定性**
- Constraint 13 上限收緊：`P[i,t] ≤ renew_max × forecast × 0.92`（保留 8% 安全餘裕）

**Assumption III — 彈性市場機制**
- 新增 `commit[t]` 變數，代表日前售電承諾
- 新增 `commit[t] ≤ sell[t]` 限制式
- 靜態排程強制 `commit[t] = sell[t]`（penalty = 0）
- 基礎設施供動態排程使用

每個排程時段輸出新增欄位：`day_ahead_commit`、`pv_forecast`、`pv_actual`

### src/evaluator.py

- 新增 `relaxed_assumptions` 區塊，記錄三個 assumption 的量化結果
- 修正 `sporadic_value_rate` 分母：改為包含所有 demo sporadic jobs（含被拒絕的），由 1.0 修正為 **0.8**
- 新增動態排程的 realtime-aware revenue 計算

### src/verifier.py

新增四項驗證：
- `C16_L2`：含效率的 SOC 更新方程
- `C14_SOC_dep`：SOC-dependent discharge limit
- `C13_robust`：PV 出力不超過 92% 安全上限
- `commitment_nonneg` / `commitment_le_sell`：承諾變數合法性

### output/ 新增三個檔案

- `schedule_result_advanced.json`
- `acceptance_test_log_advanced.json`
- `evaluation_results_advanced.json`

---

## 關鍵數字比較

| 指標 | Level 1 靜態 | Level 2 靜態 | Level 2 動態 |
|------|:---:|:---:|:---:|
| Objective value | −$34,295 | −$40,904 | −$55,187 |
| Generator cost | $276,400 | $285,100 | $302,700 |
| Market revenue | $340,695 | $336,004 | $357,887 |
| Cancellation penalty | — | $0 | $318 |
| Soft deadline miss rate | 37.5% | 12.5% | 0% |
| Sporadic value rate | 1.0 | 0.8 | 0.8 |
| Hard deadline miss rate | 0% | 0% | 0% |

---

## 動態排程方法（advanced_scheduler.py）

採用 **Event-Triggered Rolling Horizon**，在靜態日前排程的基礎上進行即時調整。

重排觸發條件：
1. PV 實際出力與預測偏差 > 15%
2. 新 sporadic job 到達
3. 每 6 小時定期重排
4. 新 aperiodic job 到達

Rolling window 大小：12 小時，共發生 **25 次 replan**，全部求解成功。

日前承諾（`day_ahead_commit`）在模擬開始時鎖定。動態排程允許實際售電量低於承諾（觸發 $318 penalty），也可高於承諾（以 realtime price 計算額外收益）。
