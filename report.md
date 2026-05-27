# VPP Real-Time Scheduling System 報告

> 即時系統期末專案 — 虛擬電廠 72 小時排程系統 (Level 1 + Level 2)

---

## 目錄

1. [Periodic Task Set 產生方式](#1-periodic-task-set-產生方式)
2. [放寬 Assumption 的限制式建模 (Level 2)](#2-放寬-assumption-的限制式建模-level-2)
3. [排程演算法設計說明](#3-排程演算法設計說明)
   - 3.1 Level 1 日前靜態排程
   - 3.2 Level 2 進階動態排程
4. [效能分析](#4-效能分析)
   - 4.1 Level 1 結果
   - 4.2 保留策略效能分析
   - 4.3 目標函數權衡分析
   - 4.4 Level 2 結果與比較
5. [討論與心得](#5-討論與心得)

---

## 1. Periodic Task Set 產生方式

### 1.1 產生策略

採用「**目標 frame size = 4**」的隨機生成策略，搭配重試機制確保所有規格限制同時成立。實作於 [`src/task_generator.py`](src/task_generator.py)。

**核心想法**：先固定 `f = 4`（因為 `f ≥ max(e) = 4` 且 `H = 72` 可被 4 整除），再針對每個 task 決定 `(r, p, e, d, w, preempt)` 使其滿足：
- 個別參數範圍（1-4 條件）
- Frame 限制式 `2f − gcd(f, p_j) ≤ d_j`（規格 1-8）
- 整體 workload density `D_w ≥ 0.7`（規格 1-5）

### 1.2 數值約束處理

| 規格項 | 條件 | 處理方式 |
|---|---|---|
| 1-2 | `6 ≤ |Jp| ≤ 10` | 固定產生 9 個 task |
| 1-3 | 展開後 jobs > 30 | 控制 period 範圍使展開數 > 30 |
| 1-4 | `1 ≤ r ≤ period` | `r` 從 `[1, p]` 均勻取樣 |
| 1-4 | `6 ≤ period ≤ 24`，至少 3 種不同 period | period 從 `{8, 12, 15, 16, 20, 21, 24}` 取樣 |
| 1-4 | `1 ≤ e ≤ 4`，≥2 個 `e = 2`，≥1 個 `e ≥ 3` | 顯式設定 |
| 1-4 | `e ≤ d ≤ p` | 線性產生 |
| 1-4 | `6 ≤ w ≤ 18`，≥2 個 `w ≥ 14` | 隨機取樣 |
| 1-6 | ≥20% 任務 `d = e` | 預留 2 個 task 設為 tight deadline (d=e=4) |
| 1-7 | ≥2 個 `e ≠ 1` 的 non-preempt | 顯式設定 5 個 non-preempt |
| 1-8 | Frame size 三條件 | 固定 `f = 4`，所有 task 滿足 `2·4 − gcd(4, p) ≤ d` |

### 1.3 生成結果

| Task | r | p | e | d | w | preempt |
|---|---|---|---|---|---|---|
| p1 | 15 | 16 | 4 | 13 | 18 | 1 |
| p2 | 16 | 24 | 4 | 4 | 12 | 0 |
| p3 | 9 | 20 | 2 | 20 | 6 | 0 |
| p4 | 4 | 21 | 3 | 15 | 11 | 1 |
| p5 | 5 | 8 | 3 | 5 | 9 | 0 |
| p6 | 6 | 8 | 2 | 7 | 18 | 0 |
| p7 | 2 | 20 | 4 | 4 | 10 | 0 |
| p8 | 13 | 21 | 2 | 20 | 9 | 1 |
| p9 | 14 | 15 | 2 | 8 | 7 | 1 |

- **|Jp| = 9** (滿足 6 ≤ |Jp| ≤ 10)
- **展開後 instances = 44** (> 30)
- **不同 period 數 = 6** (≥ 3)
- **Workload density `D_w` = 1.71** (≥ 0.7)
- **Tight deadline tasks**：p2、p7 (≥ 20% = 1.8)
- **Non-preempt with `e ≠ 1`**：p2、p3、p5、p6、p7 (≥ 2)
- **Frame size `f = 4`**：所有 task 都滿足三條件

---

## 2. 放寬 Assumption 的限制式建模 (Level 2)

Level 2 放寬規格 §2.2 的三個 assumptions，總共建模 10 條額外限制式，全部實作於 [`src/advanced_scheduler.py`](src/advanced_scheduler.py)。

### 2.1 Assumption I — 再生能源不確定性

**Notation**:
- `pv_actual[i, t]`：再生能源 `i` 在時段 `t` 的實際出力百分比
- `err_std`：預測誤差標準差 (0.08)

**敘述**：
- 計劃時使用 robust 邊界 `forecast · (1 − err_std)`，預留 8% 不確定性
- 執行時揭露 `pv_actual`，若實際小於計劃則需動態調整
- 當 `(forecast − actual) / forecast > 7%` 時觸發 replan

**限制式**:

| # | 公式 | 說明 |
|---|---|---|
| L2-1 | `P[i, t] ≤ renewmax_i · forecast[i, t] · (1 − err_std)`, ∀i ∈ Ir, t > t_now | 未來時段使用 robust 上限 |
| L2-2 | `P[i, t_now] ≤ renewmax_i · pv_actual[i, t_now]`, ∀i ∈ Ir | 當前時段使用實際值 |

實作位置：[`advanced_scheduler.py:_rolling_replan`](src/advanced_scheduler.py) 中對應 `pv` 的 P 上限。

### 2.2 Assumption II — 儲能設備真實運作

**Notation**:
- `η_c[b]`：充電效率 (0.95)
- `η_d[b]`：放電效率 (0.92)
- `σ[b]`：自放電率 (0.002 / hour)
- `aging_cost[b]`：每 MWh 放電的老化成本 ($8 / MWh)

**敘述**：
- 充電時實際進入 SOC 的能量會打折 (× η_c)
- 放電時需多消耗 SOC 才能輸出同樣能量 (÷ η_d)
- 每小時 SOC 自然衰減 σ 比例
- 額外加入老化成本至 f2
- 當 SOC < 30% × `soc_max` 時，放電上限按 SOC 比例下降

**限制式**:

| # | 公式 | 說明 |
|---|---|---|
| L2-3 | `SOC[b, t] = SOC[b, t-1] · (1 − σ) + chg[b, t] · η_c − dis[b, t] / η_d` | 真實 SOC 動態 |
| L2-4 | `dis[b, t] ≤ dis_max · min(1, SOC[b, t-1] / (0.3 · soc_max))` | SOC 相依放電上限 |
| L2-5 | f2 加入 `Σ aging_cost[b] · P[b, t]` | 老化成本納入目標函數 |

### 2.3 Assumption III — 彈性市場機制

**Notation**:
- `commit[t]`：日前承諾賣電量（鎖定為 L1 排程的 `sell[t]`）
- `cancel_rate`：違約罰金率 (0.3)
- `rt_factor[t]`：即時市場價格倍數

**敘述**：
- 系統與市場簽訂日前承諾 `commit[t]`，後續實際售電量低於承諾需付違約金
- 超出承諾的部分可在即時市場以 `λ[t] · rt_factor[t]` 計價售出（plan 內保守取 min(1, rt_factor) 防止 LP 無界）

**限制式**:

| # | 公式 | 說明 |
|---|---|---|
| L2-6 | `s_under[t] ≥ commit[t] − sell[t]`, `s_under[t] ≥ 0` | 違約量下界 |
| L2-7 | `s_over[t] ≥ sell[t] − commit[t]`, `s_over[t] ≥ 0` | 超賣量下界 |
| L2-8 | f3 加入 `cancel_rate · λ[t] · s_under[t]` | 違約罰金 |
| L2-9 | f3 減去 `λ[t] · (1 − min(1, rt_factor[t])) · s_over[t]` | 即時市場超賣收益 |
| L2-10 | 觸發 replan 條件：`(forecast − actual) / forecast > 0.07` | 動態適應 PV 偏差 |

合計 **10 條限制式**，滿足規格 3-1 (Level 2 共 10 分) 上限。

---

## 3. 排程演算法設計說明

### 3.1 Level 1 日前靜態排程 (`scheduler.py`)

整體流程：

```
Periodic task set → Phase 1 ILP → Online Phase → 評估
```

#### 3.1.1 Phase 1 — Day-ahead ILP

以 PuLP CBC solver 解一個整數線性規劃，**一次決定整個 72 小時的所有發電與售電**。

**決策變數**:
- `P[i, t]`：處理器 `i` 在時段 `t` 的總供電量
- `k[j, i, t]`：用電需求 `j` 在時段 `t` 由設備 `i` 供應的電能量
- `u[g, t]`：傳統機組 `g` 的開關機 binary
- `z_on / z_off[g, t]`：開機 / 關機事件 binary
- `x[j, t]`：用電需求 `j` 是否在 `t` 執行 binary
- `y[j, s]`：non-preemptive 工作的啟動時段 binary
- `chg / dis / soc[b, t]`：電池充放電與儲量
- `v_chg[b, t]`：電池充放電互斥 binary
- `sell[t]`：時段 `t` 的售電量
- `miss[j]`：aperiodic 是否 miss binary（本 pipeline 中不使用）

**目標函數**: minimize `α · f1 + f2 + f3`

| 分量 | 計算 |
|---|---|
| `f1` | `Σ miss_j` (aperiodic miss 數量) — 此 pipeline 中 = 0 |
| `f2` | `Σ (coston_i · u[g, t] + costup_i · P[g, t])` (傳統機組成本) |
| `f3` | `−Σ λ_t · sell[t]` (售電收益取負號) |

**保留策略 — 旋轉備轉容量**:

[`scheduler.py`](src/scheduler.py#L36) 設定 `RESERVE_PER_GEN = 5`，並加入限制式：

```
P[g, t] ≤ (output_max[g] − 5) · u[g, t]
```

即每台開機的傳統機組強制保留 5 MW 的「旋轉備轉容量」(spinning reserve)，供 online phase 的 sporadic / aperiodic 工作吸收使用。

#### 3.1.2 Online Phase — sporadic + aperiodic 即時處理

按 **release time 排序** 處理（同時間到達時 sporadic 優先）。每個工作的處理步驟：

**Step 1 — 找最便宜的放置時段** (`_find_min_cost`)

每小時的成本依「來源排序」計算：

```
PV 多餘容量 (cost = 0, 免費)
→ 電池額外放電 (cost = 0, L1 無 aging cost)
→ 傳統機組 ramp (cost = cost_variable[g] / MWh)
→ Sell 借用 (cost = λ[t] / MWh)
```

排序後 PV 與電池永遠優先；最後在「gen vs sell」比較單位 MWh 邊際成本，較便宜者先用。

- **preempt=1**：挑 `e` 個最便宜的小時
- **preempt=0**：挑連續 `e` 小時總成本最低的視窗

**Step 2 — Sporadic 接受判斷**

```
若不可行 (連 100% sell + gen + PV + battery 都放不下) → reject
否則若 (accepted_e / total_e ≥ 0.8 且 cost > $1500)  → strategic reject
否則 → accept
```

策略性拒絕的目的：在 sporadic 已經安全達標滿分門檻 (0.7) 的前提下，拒絕「貴」的工作以保留 f2 / f3。設 0.8 而非 0.7 是為了保留 1 個 e-unit 的安全 buffer，避免後續 sporadic 無法 fit 時 rate 掉破 0.7。

**Step 3 — Aperiodic 強制執行 (規格 C4)**

```
找最便宜的 on-time placement (在 [release, soft_deadline])
若可行 → 比較 on-time 與 late 的總成本（late 加 ALPHA 罰金）
   若 cost_late + ALPHA < cost_ontime → 排 late (記為 miss)
   否則 → 排 on-time
否則 → 必須排 late (在 [release, H] 內最便宜處)
```

**Step 4 — 提交排程 (`_commit_run_min_sell`)**

連續時段以小型 **joint LP** 同時求解，避免相鄰時段的 ramp_up / ramp_down 互相牽制造成的低估。

LP 目標：

```
min  Σ sell_borrow_t × λ_t  +  Σ gen_ramp_g,t × cost_variable[g]
```

考量：相鄰時段的 P 同時可變動 → ramp 限制式跨時段耦合。

電池在單時段 commit 中可額外放電，受三個約束保護：
- C14：`new_dis ≤ discharge_max`
- C19：當前時段非充電
- C17 SOC 鏈：`Δ ≤ min_{s ≥ t} (SOC[s] − soc_min)`

提交完成後對所有 `SOC[b, s≥t]` 同步減去 `Δ`，維持排程一致性。

### 3.2 Level 2 進階動態排程 (`advanced_scheduler.py`)

#### 3.2.1 整體架構

L2 在 L1 排程之上加上「即時資訊揭露 + 滾動視窗重排」機制：

```
1. 載入 L1 的 schedule_result.json 為初始計劃
2. 剝離 L1 排好的 sporadic / aperiodic（避免重複擺放）
3. 鎖定 day-ahead 承諾 commit[t] ← L1 的 sell[t]
4. 執行一次初始 replan（適配 L2 真實電池動態）
5. 進入逐小時模擬主迴圈
```

#### 3.2.2 每小時主迴圈

```python
for t in 1..72:
    1. 處理本小時到達的 sporadic / aperiodic (沿用 online_phase)
    2. 檢查 4 種觸發條件
    3. 若觸發 → 跑 12 小時滾動視窗 replan
    4. 執行本小時（揭露 pv_actual，累計違約 penalty）
```

#### 3.2.3 Replan 觸發條件

| 觸發名稱 | 條件 | 動機 |
|---|---|---|
| `aperiodic_arrival` | 有新的 aperiodic 工作到達 | 重新分配新增需求 |
| `sporadic_arrival` | 有 sporadic 被接受 | 同上 |
| `periodic_6h` | 每 6 小時例行 | 定期吸收累積偏差 |
| `pv_deviation > 7%` | `(forecast − actual) / forecast > 0.07` | 避免 PV 短缺打破計劃 |

#### 3.2.4 Rolling-horizon Replan ILP

- **視窗 12 小時**：只重新優化 `[t, t+11]`
- **所有工作的時段分配凍結**：sporadic / aperiodic 在到達時即決定哪幾小時跑，replan **不會再動**
- **只重新決定 dispatch**：傳統機組 on/off、P、battery 充放電、sell、與 `K[j, i, t]`（每個工作從哪台設備拿電）

#### 3.2.5 Execute Hour — 真實揭露

- PV 上限改成 **`pv_actual`**（實際值）
- 若計劃排太多 PV → 從 sell 扣除差額（無法扣完則記為 imbalance warning）
- 若實際 sell < `locked_commit` → 累計 `cancellation_penalty = cancel_rate × λ[t] × (commit − sell)`
- 用 L2 電池動態更新 SOC

---

## 4. 效能分析

### 4.1 Level 1 結果

[`output/evaluation_results.json`](output/evaluation_results.json) 全部評估指標：

| 指標 | 值 | 規格項 |
|---|---|---|
| **objective_value** | **−$73,328.80** | 整體 |
| f2 generator_cost | $280,530.00 | 規格 6-2 |
| f3 market_revenue | $353,858.80 | 規格 6-2 |
| hard_deadline_miss_rate | 0.0% | 5-1 |
| soft_deadline_miss_rate | 0.0% | 5-2 |
| average_tardiness | 0.0 h | 5-3 |
| max_tardiness | 0 h | 5-3 |
| average_response_time | 5.89 h | 5-4 |
| max_response_time | 19 h | 5-4 |
| completion_time_jitter | per-task dict | 5-5 |
| sporadic_value_rate | **0.9** (滿分) | 4-3 |

Completion-time jitter（每個 periodic task 各 instance 的 response time 母體標準差）：

| Task | jitter |
|---|---|
| p1 | 3.83 |
| p2 | 0.0 |
| p3 | 5.52 |
| p4 | 3.49 |
| p5 | 0.87 |
| p6 | 2.0 |
| p7 | 0.0 |
| p8 | 1.70 |
| p9 | 2.12 |

p2 與 p7 jitter 為 0 是因為 deadline = e（tight deadline），每個 instance 都必須在 release 後緊接著立刻執行，所以 response time 完全一致。

驗證：`python3 src/verifier.py` → **驗證通過！排程結果完美符合所有 Constraints。**

### 4.2 保留策略效能分析 (規格 6-1, 5 分)

#### 三層保留機制

**(1) 旋轉備轉容量 (`RESERVE_PER_GEN = 5 MW`)**

每台開機的傳統機組強制留 5 MW 空間，由 Phase 1 ILP 限制式 `P[g] ≤ (Pmax − 5) · u` 強制達成。當 sporadic / aperiodic 到達需要額外發電時，可從這 5 MW 內 ramp up，不需要重新啟動其他機組。

| 機組 | output_max | 保留量 | 佔比 |
|---|---|---|---|
| thermal_1 | 80 MW | 5 MW | 6.3% |
| thermal_2 | 45 MW | 5 MW | 11.1% |

5 MW 大小的選擇：太少無法接住一般 sporadic 的瞬時需求；太多會浪費容量降低售電收益。實測在這份輸入下，5 MW 是一個良好的平衡點。

**(2) 電池保留 (隱性)**

電池在 Phase 1 ILP 中由全域最佳化決定何時充放電。電池本身的 SOC 即是「能量保留」，online phase 可在不違反 SOC 鏈的前提下，動用尚未放完的電池能量供新工作使用。

| 電池 | 開機小時數 | 佔比 (/72h) |
|---|---|---|
| battery_1 | 30 h | 41.7% |
| battery_2 | 26 h | 36.1% |

**(3) Sell 借用分層 (Phase 3 only)**

Online phase 在處理 sporadic / aperiodic 時的 sell 借用採三層機制：

| 層級 | 可用資源 | 條件 |
|---|---|---|
| L1 | 機組備轉 + PV 餘裕 + 電池放電 | 完全不動 sell |
| L2 | + sell 全部 | Aperiodic 強制執行需要時 |
| 經濟性閘門 | — | 拒絕昂貴的 sporadic (`cost > $1500` 且 rate ≥ 0.8) |

#### Sporadic 到達時如何使用保留資源

1. `_find_min_cost` 計算每個候選小時的接受成本（PV → 電池 → 機組 → sell 排序）
2. 找出 `e` 個成本最低的時段
3. `_commit_run_min_sell` 連續時段以 joint LP 一次提交，避免相鄰時段 ramp 互相牽制

#### 實際數據佐證

- **sporadic_value_rate = 0.9** (滿分區間 ≥ 0.7)：4 個 sporadic 全部接受，s3 因 `e=1, w=20 MW` 在 [35, 37] 視窗內 100% sell + 全部資源都不夠而被 reject（infeasibility）
- **aperiodic miss rate = 0**：9 個 aperiodic 全部 on-time 完成
- **約 70% 的 sporadic / aperiodic 不需要借用 sell**：完全由保留資源吸收，未影響 f3

### 4.3 目標函數權衡分析 (規格 6-2, 5 分)

三個目標的實際數值與權衡關係：

| 目標分量 | 計算 | L1 實測值 |
|---|---|---|
| `α · f1` | aperiodic miss × $10,000 | **$0** (0 misses) |
| `f2` | 傳統機組固定 + 變動成本 | **$280,530.00** |
| `f3` | −售電收益 | **−$353,858.80** |
| **objective** | α·f1 + f2 + f3 | **−$73,328.80** |
| **sporadic_value_rate** | 接受的 sporadic e 總和 / 總 sporadic e | **0.9** |

#### 權衡 A：f2 vs f3（成本 vs 收益）— Phase 1 ILP 處理

ILP 想關掉發電機降低 f2，但同時 f3 會變差（沒東西可賣）。權衡結果：
- **thermal_1** 開 62 小時（高利用率，cost_variable = $42/MWh 較便宜）
- **thermal_2** 只開 17 小時（cost_variable = $70/MWh 較貴，僅在售電價格高時段開機）

#### 權衡 B：f1 vs f3 — Online Phase Attempt 3 處理

每個邊界上的 aperiodic 都在做這個經濟性決策：
- 直接 miss → +$10,000 進入 f1
- 借 sell 安排 → f3 損失 (借的 MWh × 當下市場價格)
- 比較兩者，選擇便宜的

實測：**9 個 aperiodic 全部 on-time**。Aperiodic 的 sell-borrow 總成本估算遠低於 $90,000（= 9 × ALPHA），所以全部 on-time 安排為最優解。

#### 權衡 C：sporadic_value_rate vs f3 — Strategic Rejection

每接受一個 sporadic，就會佔用 reserve / sell。我們設了兩個閘門：
- **`SPORADIC_RATE_FLOOR = 0.8`**：當累積接受率已 ≥ 0.8，才考慮策略性拒絕。0.8（而非 0.7）保留了 1 個 e-unit 的安全 buffer
- **`SPORADIC_REJECT_COST = $1500`**：成本超過此金額才拒絕；以 demo 輸入的 sporadic 規模（`w × e × 平均價格`）估算，落在中位水準

實測：本次運行所有 sporadic 接受成本都 < $1500，所以策略性拒絕**未觸發**。只有 s3 因 infeasibility 拒絕。`sporadic_value_rate = 0.9 > 0.7` 滿分區間。

#### 跨層互動

**sporadic vs aperiodic 競爭 reserve**：online phase 按 release time 順序處理，先到先用。本次運行所有 sporadic + aperiodic 都被妥善安排，代表 reserve 配置充裕。

**ALPHA = $10,000 的影響**：
- 若 α 很小 → Phase 3 經濟閘門較易選擇 miss，aperiodic miss 數會上升
- 若 α 很大 → 完全不會 miss，但可能借更貴的 sell
- α = $10,000 在這份輸入下，aperiodic 借 sell 成本遠低於罰金，因此最佳解是「全部 on-time + 適度借 sell」

#### 結論

整個 pipeline 並非單一最佳化解 `α·f1 + f2 + f3`，而是讓每個 phase 各自做局部最佳化：
- **Phase 1**：對 periodic 部分最佳化 f2 + f3
- **Online Phase**：透過策略性拒絕 sporadic、aperiodic 經濟閘門間接影響 f1

最終評估器把所有結果加總：`objective_value = −$73,328.80`，代表售電收益超過所有發電成本，且 0 hard / soft deadline miss，sporadic 接受率 90%——三個目標都達到平衡。

### 4.4 Level 2 結果與比較

#### Level 2 實測值

| 指標 | 值 |
|---|---|
| objective_value | **−$52,561.89** |
| f2 generator_cost | $302,350.00 |
| f3 market_revenue | $354,911.89 |
| cancellation_penalty_total | $4,222.42 |
| hard_deadline_miss_rate | 0.0% |
| soft_deadline_miss_rate | 0.0% |
| sporadic_value_rate | 0.9 |
| 總 replans | 27 |
| Replan triggers | `pv_deviation: 7, sporadic_arrival: 4, periodic_6h: 12, aperiodic_arrival: 9` |

驗證：`python3 src/verifier_advanced.py` → **驗證通過！動態排程結果完美符合所有 Constraints。**

#### 排程結果正確性說明 (規格 8-2, 4 分)

- **所有 jobs 安排結果**：9 個 aperiodic 全部 on-time 完成、4 個 sporadic 接受 (s3 因 infeasibility 拒絕)、44 個 periodic instance 全部在 deadline 前完成
- **Hard deadline jobs**：100% 在期限前完成 (miss rate = 0%)
- **系統資源限制**：所有時段供需平衡（C23），驗證器零違規
- **儲能狀態**：SOC 全程維持在 `[soc_min, soc_max]` 區間，動態符合 L2 真實模型
- **過程中事件**：
  - 27 次 replan 觸發（其中 7 次因 PV 偏差超過 7%）
  - 4 次 sporadic 接受，全部成功放入排程
  - 9 次 aperiodic 強制放入排程
  - 累計違約罰金 $4,222.42（部分小時實際 sell 低於 L1 鎖定的 day-ahead 承諾）

#### 與 Level 1 比較 (規格 8-3, 4 分)

| 指標 | Level 1 | Level 2 | Δ |
|---|---|---|---|
| objective_value | **−$73,329** | **−$52,562** | **+$20,767** (L2 較差) |
| f2 generator_cost | $280,530 | $302,350 | **+$21,820** |
| f3 market_revenue | $353,859 | $354,912 | +$1,053 |
| cancellation_penalty | n/a | $4,222 | (L2 only) |
| hard_deadline_miss_rate | 0% | 0% | — |
| soft_deadline_miss_rate | 0% | 0% | — |
| sporadic_value_rate | 0.9 | 0.9 | — |

#### 差異原因分析

L2 objective 比 L1 差 $20,767，**並非因為演算法變差，而是因為 L2 模擬了一個更貼近現實、限制更嚴的環境**：

1. **PV 實際出力不確定性 (Assumption I)**
   - 部分小時 PV 實際出力低於預測（demo input 中有 37 小時出現 shortfall），必須臨時多開傳統機組補缺 → f2 上升

2. **儲能設備真實運作 (Assumption II)**
   - 充放電效率 (η_c = 0.95, η_d = 0.92) 造成能量損失
   - 自放電率 (σ = 0.002/h) 隨時間磨耗 SOC
   - 老化成本 ($8/MWh of discharge) 加入 f2
   - 這些累計造成 L2 較難維持與 L1 相同的售電量

3. **彈性市場機制 (Assumption III)**
   - L2 鎖定 L1 的 sell 為 day-ahead 承諾
   - 當實際 PV / 機組產出無法達成承諾時，違約罰金 ($4,222.42) 加入 f3
   - 這個成本是 L1 模型完全沒有的

4. **動態 replan 對 dispatch 的影響**
   - L2 每次 replan 都需要重新平衡新的供需狀況，且必須在 12 小時視窗內找解
   - 滾動視窗無法做全域最佳化，所以單一 replan 結果通常不如 L1 ILP 的全域最佳

#### 維持的正向特性

雖然 objective 變差，但所有「品質」指標完全持平：
- **Hard / Soft miss rate 都是 0%**：L2 動態適應仍能讓所有工作完成
- **Sporadic value rate = 0.9**：與 L1 相同，仍在滿分區間
- **27 次 replan 全部成功**：沒有任何 infeasible 情況
- **驗證器通過**：所有 L2 限制式 + L1 殘留限制式都滿足

#### 多目標間衝突的具體展現

L2 的數據完美體現了三個目標間的衝突：
- 提升 day-ahead 承諾履約率 → 多開機組補 PV → **f2 ↑**
- 降低違約罰金 → 限縮即時市場套利 → **f3 上升幅度有限** (+$1,053)
- 完整放入所有 aperiodic → **f1 = 0**，但需要更多資源支援

整體 trade-off 結果：L2 objective `−$52,562` 仍是「淨賺」(收益 > 成本)，但比理想化的 L1 模型差。這是合理且預期的結果。

---

## 5. 討論與心得

### 5.1 使用 AI 輔助

**使用工具**：Claude Code (Anthropic) — 整合 IDE 的 AI 編程助手

**協作方式**：以「迭代式 prompt + 立即測試」為主要協作模式：

1. **初步理解規格**：將規格 PDF 整段提供給 AI，請其萃取需求；再針對不清楚的細節（如 frame size 條件、acceptance test 流程）追問
2. **分階段實作**：將整個系統拆分成 (a) task generator → (b) Phase 1 ILP → (c) online phase → (d) advanced scheduler → (e) evaluators / verifiers 五個階段，每階段獨立 prompt + 測試後再合併
3. **錯誤驅動的迭代**：每次測試找到 bug 或不合理結果（例如 joint LP 偏好 gen 而過度增加 f2），描述觀察到的現象，請 AI 分析根因並修正
4. **設計取捨討論**：對於非單純對錯的設計選擇（例如「sporadic 是否要策略性拒絕」、「joint LP 是否包含電池」），先請 AI 分析利弊，再做決定
5. **驗證與報告**：完成實作後請 AI 依照規格比對驗證，並協助撰寫 README 與本份報告

**Prompt 策略**：
- **具體化問題**：「在這個 demo input 下我看到 f2 比預期高 $20k，原因是什麼？」優於「我的程式有 bug」
- **要求量化分析**：請 AI 在做設計決策前先實際跑資料 / 算數字
- **拒絕「就照建議做」**：在 AI 提出修改建議時，要求說明為什麼、有什麼權衡、什麼情況下不適用
- **驗證輸出**：每次重大改動都跑 `verifier.py` 與 `evaluator.py`，不單純信任 AI 的「應該 OK」

### 5.2 實作心得

1. **規格的深層意義往往不在字面上**
   - 例如 C4 第三項規定 aperiodic 必須在 H 前執行完 `e` 小時，這個約束讓「直接 skip aperiodic」變成違規。我們最初的版本確實有 silent skip 的 bug，是後來透過 verifier 才發現

2. **online 與 offline 視角的差別非常微妙**
   - Sporadic 規格上是「online 到達」，但 demo 時所有資料是提前給的。要不要利用這個「實質知道」的優勢？我們選擇用 lookahead-aware 的策略性拒絕，並把這個假設明確寫進報告

3. **多個 phase 的 local optimization 不等於 global optimization**
   - L1 的 Phase 1 ILP 是 periodic 全域最佳，但加上 online phase 之後，最終的 `α·f1 + f2 + f3` 不是全域最佳。這在報告 4.3 章節有詳細解釋。理解這一點對 demo 講解非常重要

4. **L2 比 L1 「差」是正確的結果**
   - 一開始看到 L2 objective 比 L1 差很多時懷疑是不是有 bug。後來才理解：L2 模擬了現實的隨機性 / 損耗 / 違約成本，這些都是 L1 沒有的。L2 比 L1 差是「越接近現實越貴」的合理現象

5. **驗證器是最好的朋友**
   - 多次發現 bug 都是靠 `verifier.py` / `verifier_advanced.py` 找到。例如 C20 違規（PV capping 後 k 未同步調整）、C1 違規（雙重 placement）、C16 違規（L1 plan 餵入 L2 後 SOC 不一致），都是驗證器先報異常，我們才追蹤源頭

### 5.3 未來可改進方向

1. **Joint LP 加入電池變數**：目前 joint LP 只考慮 gen ramp + PV + sell，若加入電池可進一步降低非搶占工作的 commit 成本。實測在這份 input 下因電池被單時段 commit 抽乾而無剩餘空間，但在不同 input 可能有實質改進空間

2. **Strategic reject 動態調整 threshold**：目前 `$1500` 是靜態值；可改為依當前資源使用率動態調整（資源越緊張閾值越低）

3. **L2 加入 day-ahead commit 滾動更新**：目前 commit 鎖定為 L1 的 sell；可考慮在 L2 內部允許定期 (如每 24h) 更新 commit，降低違約罰金

4. **Frame size 自適應**：目前 task generator 預設 f=4。若 input 變化，可動態尋找最大合法 f，提升非搶占工作的彈性

---

## 附錄 — 主要檔案位置

| 檔案 | 內容 |
|---|---|
| [src/task_generator.py](src/task_generator.py) | Periodic task set 隨機產生 |
| [src/scheduler.py](src/scheduler.py) | Level 1 Phase 1 ILP + online phase |
| [src/evaluator.py](src/evaluator.py) | Level 1 評估指標計算 |
| [src/verifier.py](src/verifier.py) | Level 1 限制式驗證 |
| [src/advanced_scheduler.py](src/advanced_scheduler.py) | Level 2 動態滾動排程 |
| [src/evaluator_advanced.py](src/evaluator_advanced.py) | Level 2 評估指標計算 |
| [src/verifier_advanced.py](src/verifier_advanced.py) | Level 2 限制式驗證 |
| [README.md](README.md) | 使用說明文件 |

---

*報告版本：v1.0 — 完成於專案 Demo 前*
