# VPP Real-Time Scheduling System 報告

> 即時系統期末專案 — 虛擬電廠 72 小時排程系統 (Level 1 + Level 2)

---

## 目錄

1. [Periodic Task Set 產生方式](#1-periodic-task-set-產生方式)
2. [放寬 Assumption 的限制式建模 (Level 2)](#2-放寬-assumption-的限制式建模-level-2)
3. [排程演算法設計說明](#3-排程演算法設計說明)
4. [效能分析](#4-效能分析)
5. [討論與心得](#5-討論與心得)

---

## 1. Periodic Task Set 產生方式

### 1.1 產生策略

採用「**目標 frame size = 4**」的隨機生成策略，搭配重試機制確保所有規格限制同時成立。實作於 `src/task_generator.py`。

**核心想法**: 先固定 `f = 4`，再針對每個 task 決定 `(r, p, e, d, w, preempt)` 使其滿足:
- 個別參數範圍 (規格 1-4)
- Frame 限制式 `2f − gcd(f, p_j) ≤ d_j` (規格 1-8)
- 整體 workload density `D_w ≥ 0.7` (規格 1-5)

### 1.2 數值約束處理

| 規格項 | 條件 | 處理方式 |
|---|---|---|
| 1-2 | `6 ≤ |Jp| ≤ 10` | 每次產生 6~10 個 task |
| 1-3 | 展開後 jobs > 30 | 控制 period 範圍使展開數 > 30 |
| 1-4 | `1 ≤ r ≤ period` | `r` 從 `[1, p]` 均勻取樣 |
| 1-4 | `6 ≤ period ≤ 24`, 至少 3 種不同 period | period 從合法範圍取樣 |
| 1-4 | `1 ≤ e ≤ 4`, ≥2 個 `e=2`, ≥1 個 `e≥3` | 顯式設定 e_pool |
| 1-4 | `e ≤ d ≤ p` | 取下界 max(e, 2f-gcd(f,p)) 與 p 之間隨機 |
| 1-4 | `6 ≤ w ≤ 18`, ≥2 個 `w≥14` | 隨機取樣 + 必要時 patch |
| 1-6 | ≥20% 任務 `d = e` | 預留 tight tasks (d=e=4) |
| 1-7 | ≥2 個 `e ≠ 1` 的 non-preempt | 強制 tight tasks 設為 non-preempt |
| 1-8 | Frame size 三條件 | 固定 `f=4`, 所有 task 滿足 `2·4 − gcd(4, p) ≤ d` |

### 1.3 為什麼 frame size 固定為 4

由規格 1-4 中 `e ≤ 4` 推導:
- `f ≥ max(e) ≥ 3` (還需要至少一個 e≥3 的 task)
- tight tasks 滿足 `d = e ≤ 4` 同時也要 `2f − gcd(f, p) ≤ d`
- 對於 `f ≥ 5`: `2f − gcd(f, p) ≥ f ≥ 5 > 4 ≥ d` (永遠違反)
- 對於 `f = 4`: 只要 `gcd(4, p) = 4` (即 p 為 4 的倍數)，就能滿足 `2·4 − 4 = 4 ≤ d = 4`

因此 `target_f ∈ {3, 4}` 是唯一可行的選擇，預設使用 4 以保留 `e = 4` 的彈性。

### 1.4 本次提交的 task set

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
- **Workload density D_w = 1.71** (≥ 0.7)
- **Tight deadline tasks (d=e)**: p2, p7 (≥ 20%)
- **Non-preempt with e ≠ 1**: p2, p3, p5, p6, p7 (≥ 2)
- **Frame size f = 4** 滿足所有限制

---

## 2. 放寬 Assumption 的限制式建模 (Level 2)

Level 2 放寬規格 §2.2 的三個 assumptions，總共建模 10 條額外限制式，全部實作於 `src/advanced_scheduler.py` 的 rolling-horizon replan ILP。

### 2.1 Assumption I — 再生能源不確定性

**Notation**:
- `pv_actual[i, t]`: 再生能源 `i` 在時段 `t` 的實際出力百分比
- `err_std`: 預測誤差標準差 (本實作設為 0.08)

**敘述**: 計劃時使用 robust 邊界 `forecast · (1 − err_std)`, 預留 8% 不確定性。執行時揭露 `pv_actual`, 若 PV 實際出力低於預測 7% 以上則觸發 replan。

**限制式**:

| # | 公式 | 說明 |
|---|---|---|
| L2-1 | `P[i, t] ≤ renewmax_i · forecast[i, t] · (1 − err_std)`, ∀i ∈ Ir, t > t_now | 未來時段使用 robust 上限 |
| L2-2 | `P[i, t_now] ≤ renewmax_i · pv_actual[i, t_now]` | 當前時段使用實際值 |

### 2.2 Assumption II — 儲能設備真實運作

**Notation**:
- `η_c[b]`: 充電效率 (0.95)
- `η_d[b]`: 放電效率 (0.92)
- `σ[b]`: 自放電率 (0.002 / hour)
- `aging_cost[b]`: 每 MWh 放電的老化成本 ($8 / MWh)

**敘述**: 充電時實際進入 SOC 的能量會打折; 放電時需多消耗 SOC 才能輸出同樣能量; 每小時 SOC 自然衰減; 加入老化成本; SOC < 30% 時放電上限按比例下降。

**限制式**:

| # | 公式 | 說明 |
|---|---|---|
| L2-3 | `SOC[b, t] = SOC[b, t-1] · (1 − σ) + chg[b, t] · η_c − dis[b, t] / η_d` | 真實 SOC 動態 |
| L2-4 | `dis[b, t] ≤ dis_max · min(1, SOC[b, t-1] / (0.3 · soc_max))` | SOC 相依放電上限 |
| L2-5 | f2 加入 `Σ aging_cost[b] · P[b, t]` | 老化成本納入目標 |

### 2.3 Assumption III — 彈性市場機制

**Notation**:
- `commit[t]`: 日前承諾賣電量 (鎖定為 L1 排程的 `sell[t]`)
- `cancel_rate`: 違約罰金率 (0.3)
- `rt_factor[t]`: 即時市場價格倍數

**敘述**: 系統與市場簽訂日前承諾, 後續實際售電量低於承諾需付違約金。超出承諾的部分可在即時市場以 `λ[t] · rt_factor[t]` 計價售出。

**限制式**:

| # | 公式 | 說明 |
|---|---|---|
| L2-6 | `s_under[t] ≥ commit[t] − sell[t]`, `s_under[t] ≥ 0` | 違約量下界 |
| L2-7 | `s_over[t] ≥ sell[t] − commit[t]`, `s_over[t] ≥ 0` | 超賣量下界 |
| L2-8 | f3 加入 `cancel_rate · λ[t] · s_under[t]` | 違約罰金 |
| L2-9 | f3 減去 `λ[t] · (1 − min(1, rt_factor[t])) · s_over[t]` | 即時市場超賣收益 |
| L2-10 | 觸發 replan 條件: `(forecast − actual) / forecast > 0.07` | 動態適應 PV 偏差 |

合計 **10 條限制式**, 滿足規格 3-1 (Level 2 共 10 分) 上限。

---

## 3. 排程演算法設計說明

### 3.1 Level 1 日前靜態排程 (`scheduler.py`)

整體流程:

```
Periodic task set → Phase 1 ILP → Online Phase → 評估
```

#### 3.1.1 Phase 1 — Day-ahead ILP

以 PuLP CBC solver 解一個整數線性規劃, **一次決定整個 72 小時的所有發電與售電**。

**決策變數**:
- `P[i, t]`: 處理器 `i` 在時段 `t` 的總供電量
- `k[j, i, t]`: 用電需求 `j` 在時段 `t` 由設備 `i` 供應的電能量
- `u[g, t]`: 傳統機組 `g` 的開關機 binary
- `z_on / z_off[g, t]`: 開機 / 關機事件 binary
- `x[j, t]`, `y[j, s]`: 工作執行 / 啟動 binary
- `chg / dis / soc[b, t]`: 電池充放電與儲量
- `v_chg[b, t]`: 電池充放電互斥 binary
- `sell[t]`: 時段 `t` 的售電量

**目標函數**: `minimize α · f1 + f2 + f3`

| 分量 | 計算 |
|---|---|
| `f1` | `Σ miss_j` (aperiodic miss 數量) — 此 pipeline 中 = 0 (aperiodic 不進 ILP) |
| `f2` | `Σ (coston_i · u[g, t] + costup_i · P[g, t])` (傳統機組成本) |
| `f3` | `−Σ λ_t · sell[t]` (售電收益取負號) |

**保留策略 — 旋轉備轉容量**: `RESERVE_PER_GEN = 5`, 強制每台開機機組保留 5 MW 空間:
```
P[g, t] ≤ (output_max[g] − 5) · u[g, t]
```

#### 3.1.2 Online Phase — sporadic + aperiodic 即時處理

按 **release time 排序** 處理 (同時間到達時 sporadic 優先)。每個工作的處理步驟:

**Step 1 — 找最便宜的放置時段 (`_find_min_cost`)**

每小時的成本依「來源排序」計算:

```
PV 多餘容量 (cost = 0)
→ 電池額外放電 (cost = 0, L1 無 aging cost)
→ 傳統機組 ramp (cost = cost_variable / MWh)
→ Sell 借用 (cost = λ[t] / MWh)
```

排序後 PV 與電池永遠優先; 最後在「gen vs sell」比較單位 MWh 邊際成本。

**Step 2 — Sporadic 接受/拒絕**

```
若不可行 (連 100% sell + gen + PV + battery 都放不下) → reject
策略性拒絕: 若 已接受 e 比例 ≥ 0.8 且 cost > $1500 → reject
否則 → accept
```

**Step 3 — Aperiodic 強制執行 (C4)**

預設選 on-time。經濟閘門: 若 `cost_late + ALPHA < cost_ontime` 則故意排 late (極少觸發, 因 ALPHA = $10,000 通常大於 late savings)。

**Step 4 — Commit 排程 (`_commit_min_sell`)**

按「PV → 電池 → 機組 → sell」順序提交。電池放電會同步進行 SOC 補償:
- 在未來 (`comp_after` 之後) 非充電時段, 將 sell 的能量重新導向電池充電
- 確保 SOC 軌跡仍滿足 C17 (`SOC[s] ≥ soc_min`)
- 連續時段 commit 時 `comp_after = max(run)`, 避免同一 job 不同時段競爭同一補償預算

### 3.2 Level 2 進階動態排程 (`advanced_scheduler.py`)

#### 3.2.1 整體架構

```
1. 載入 L1 的 schedule_result.json 為初始計劃
2. 剝離 L1 排好的 sporadic / aperiodic (避免重複擺放)
3. 鎖定 day-ahead 承諾 commit[t] ← L1 的 sell[t]
4. 執行一次初始 replan (適配 L2 真實電池動態)
5. 進入逐小時模擬主迴圈
```

#### 3.2.2 每小時主迴圈

```python
for t in 1..72:
    1. 處理本小時到達的 sporadic / aperiodic (沿用 online_phase)
    2. 檢查 4 種觸發條件
    3. 若觸發 → 跑 12 小時滾動視窗 replan
    4. 執行本小時 (揭露 pv_actual, 累計違約 penalty)
```

#### 3.2.3 4 個 Replan 觸發條件

| 觸發 | 條件 | 動機 |
|---|---|---|
| `aperiodic_arrival` | 新的 aperiodic 工作到達 | 重新分配新增需求 |
| `sporadic_arrival` | 有 sporadic 被接受 | 同上 |
| `periodic_6h` | 每 6 小時例行 | 定期吸收累積偏差 |
| `pv_deviation > 7%` | PV 實際出力比預測低超過 7% | 避免 PV 短缺打破計劃 |

#### 3.2.4 Rolling-horizon Replan ILP

- **視窗 12 小時**: 只重新優化 `[t, t+11]`
- **所有工作的時段分配凍結**: replan 不會再動工作的時段
- **Replan 只重新決定 dispatch**: 機組 on/off、P、battery 充放電、sell、與 `K[j, i, t]`

#### 3.2.5 Execute Hour — 真實揭露

- PV 上限改成 `pv_actual` (實際值)
- 若計劃排太多 PV → 從 sell 扣除差額
- 若實際 sell < `locked_commit` → 累計 cancellation penalty
- 用 L2 電池動態更新 SOC: `SOC[t] = SOC[t-1] · (1-σ) + chg · η_c − dis / η_d`

---

## 4. 效能分析

### 4.1 Level 1 結果

| 指標 | 值 | 規格項 |
|---|---|---|
| **objective_value** | **−$66,479.40** | 整體 |
| f2 generator_cost | $282,070.00 | 規格 6-2 |
| f3 market_revenue | $348,549.40 | 規格 6-2 |
| hard_deadline_miss_rate | 0.0% | 5-1 |
| soft_deadline_miss_rate | 0.0% | 5-2 |
| average_tardiness | 0.0 h | 5-3 |
| max_tardiness | 0 h | 5-3 |
| average_response_time | 5.44 h | 5-4 |
| max_response_time | 15 h | 5-4 |
| **sporadic_value_rate** | **85.71%** (12/14, 滿分 ≥ 70%) | 4-3 |
| completion_time_jitter | (見下表) | 5-5 |

Completion-time jitter (per task):

| Task | jitter | Task | jitter |
|---|---|---|---|
| p1 | 3.83 | p6 | 2.00 |
| p2 | 0.00 | p7 | 0.00 |
| p3 | 5.52 | p8 | 1.70 |
| p4 | 3.49 | p9 | 2.12 |
| p5 | 0.87 | | |

p2 與 p7 jitter 為 0 是因為 `d = e` (tight deadline), 每個 instance 都必須緊接 release time 立刻執行。

**驗證**: `python3 src/verifier.py` → **驗證通過! 排程結果完美符合所有 Constraints。**

### 4.2 保留策略效能分析 (規格 6-1)

#### 三層保留機制

**(1) 傳統機組旋轉備轉容量 (`RESERVE_PER_GEN = 5 MW`)**

每台開機的機組強制留 5 MW 空間, 由 Phase 1 ILP 限制式 `P[g] ≤ (Pmax − 5) · u` 強制達成。當 sporadic / aperiodic 到達需要額外發電時, 可從這 5 MW 內 ramp up。

| 機組 | output_max | 保留量 | 佔比 |
|---|---|---|---|
| thermal_1 | 80 MW | 5 MW | 6.3% |
| thermal_2 | 45 MW | 5 MW | 11.1% |

**(2) 電池保留 + SOC 補償**

電池本身的 SOC 即是「能量保留」。Online phase 中, 電池可額外放電給 sporadic / aperiodic 使用, 並透過 **SOC 補償機制** 維持限制式合法:
- 在未來非充電時段, 增加電池 chg 並減少同時段 sell (能量從 sell 重新導向 chg)
- 結果: 電池在當下放電 Δ, 未來重新充滿 Δ, SOC 軌跡回到原狀

**(3) Sell 借用 + 策略性拒絕**

Online phase 處理 sporadic 時:
- 若可放置且成本 ≤ $1500 → accept
- 若 已接受率 ≥ 0.8 且成本 > $1500 → 策略性 reject (保留 f2/f3)
- 設 0.8 而非 0.7 是為了保留 1 個 e-unit 的安全 buffer

#### 實際數據佐證

| 指標 | 值 | 解讀 |
|---|---|---|
| sporadic_value_rate | **0.857 (12/14)** | 規格 4-3 滿分區間 (≥ 0.7) |
| aperiodic miss rate | **0.0% (0/10)** | 所有 10 個 aperiodic 都 on-time 完成 |
| hard deadline miss rate | **0.0%** | 所有 periodic 與接受的 sporadic 都按時完成 |

被拒絕的 sporadic 是 s2 (r=11, e=2, w=12), 拒絕原因合法: t=11~13 兩個電池都正在 charging (Phase 1 為了 t=21+ 的放電所做的準備), C19 互斥限制阻止電池同時充放電, 因此無法為 s2 提供額外電力。

### 4.3 目標函數權衡分析 (規格 6-2)

三個目標的實際數值:

| 目標分量 | 計算 | L1 實測值 |
|---|---|---|
| `α · f1` | aperiodic miss × $10,000 | **$0** (0 misses) |
| `f2` | 傳統機組固定 + 變動成本 | **$282,070.00** |
| `f3` | −售電收益 | **−$348,549.40** |
| **objective** | α·f1 + f2 + f3 | **−$66,479.40** |
| **sporadic_value_rate** | 接受 sporadic e 總和 / 總 sporadic e | **0.857** |

#### 權衡 A — f2 vs f3 (成本 vs 收益): Phase 1 ILP 處理

ILP 在 minimize f2 + f3 時自動權衡:
- **thermal_1** 開 62 小時 (高利用率, cost_variable = $42/MWh 較便宜)
- **thermal_2** 只開 17 小時 (cost_variable = $70/MWh 較貴, 僅在售電價格高時段開機)

#### 權衡 B — f1 vs f3: Online Phase 經濟性閘門處理

每個 aperiodic 都做以下選擇:
- 直接 miss → +$10,000 進入 f1
- 借 sell / 用電池安排 → f3 損失 (借的 MWh × 當下市場價格)

實測: **10 個 aperiodic 全部 on-time**。能用電池放電的優先用電池 (free), 其次借 sell。

#### 權衡 C — sporadic_value_rate vs f3: 策略性拒絕處理

每接受一個 sporadic 就會佔用 reserve, 影響 f3。設兩個閘門:
- `SPORADIC_RATE_FLOOR = 0.8`: 累積接受率 ≥ 0.8 才考慮拒絕
- `SPORADIC_REJECT_COST = $1500`: 成本 > $1500 才拒絕

實測在本次輸入下未觸發策略性拒絕 (12/14 接受, 唯一拒絕的 s2 是因為 C19 限制不可行)。

#### 結論

整個 pipeline 不是用單一最佳化解 `α·f1 + f2 + f3`, 而是讓每個 phase 各自做局部最佳化:
- **Phase 1**: 對 periodic 部分最佳化 f2 + f3
- **Online Phase**: 透過策略性拒絕 sporadic、aperiodic 經濟閘門間接影響 f1

最終 `objective_value = −$66,479.40` 代表售電收益超過所有發電成本, 且 0 hard / soft deadline miss, sporadic 接受率 85.71% — 三個目標都達到平衡。

### 4.4 Level 2 結果與比較

#### Level 2 實測值

| 指標 | 值 |
|---|---|
| objective_value | **−$47,155.06** |
| f2 generator_cost | $303,300.00 |
| f3 market_revenue | $350,455.06 |
| cancellation_penalty_total | $3,421.21 |
| hard_deadline_miss_rate | 0.0% |
| soft_deadline_miss_rate | 0.0% |
| sporadic_value_rate | 64.29% (9/14) |
| 總 replan 次數 | 27 |

**驗證**: `python3 src/verifier_advanced.py` → **驗證通過! 動態排程結果完美符合所有 Constraints。**

#### 排程結果正確性說明 (規格 8-2)

- **所有 jobs 安排結果**: 10 個 aperiodic 全部 on-time 完成、4 個 sporadic 接受 (s2、s5 因 infeasibility 拒絕)、44 個 periodic instance 全部在 deadline 前完成
- **Hard deadline jobs**: 100% 在期限前完成 (miss rate = 0%)
- **系統資源限制**: 所有時段供需平衡 (C23), 驗證器零違規
- **儲能狀態**: SOC 全程維持在 `[soc_min, soc_max]` 區間
- **過程中事件**: 27 次 replan 觸發, 累計違約罰金 $3,421.21

#### 與 Level 1 比較 (規格 8-3)

| 指標 | Level 1 | Level 2 | Δ |
|---|---|---|---|
| objective_value | **−$66,479** | **−$47,155** | +$19,324 (L2 較差) |
| f2 generator_cost | $282,070 | $303,300 | +$21,230 |
| f3 market_revenue | $348,549 | $350,455 | +$1,906 |
| cancellation_penalty | n/a | $3,421 | (L2 only) |
| hard_deadline_miss_rate | 0% | 0% | — |
| soft_deadline_miss_rate | 0% | 0% | — |
| sporadic_value_rate | 0.857 | 0.643 | -0.214 |

#### 差異原因分析

L2 objective 比 L1 差約 $19k, **並非演算法變差, 而是 L2 模擬了一個更貼近真實的環境**:

1. **PV 實際出力不確定性 (Assumption I)**: 部分小時 PV 實際出力低於預測, 必須多開傳統機組補缺 → f2 上升
2. **儲能設備真實運作 (Assumption II)**: 充放電效率、自放電率、老化成本累計效應
3. **彈性市場機制 (Assumption III)**: L2 鎖定 L1 的 sell 為承諾, 違約罰金 $3,421
4. **動態 replan 對 dispatch 的影響**: 滾動視窗無法做全域最佳化, 單次 replan 結果通常不如 L1 ILP 的全域最佳
5. **電池時段重新分配**: 初始 L2 adaptation replan 為了適配真實電池動態 (η_c, η_d, σ), 將電池充電集中到 [39, 43] 時段, 導致 s2 與 s5 兩個 sporadic 無法用電池滿足 → sporadic_value_rate 從 0.857 降到 0.643

L2 sporadic_value_rate 較低是「真實電池模型」帶來的結果, 而非 bug — 在 L2 動態下, 電池的可用時段與 L1 不同, 部分 sporadic 不再能找到可行配置。

---

## 5. 討論與心得

### 5.1 使用 AI 輔助

**使用工具**: Claude Code (Anthropic) — 整合 IDE 的 AI 編程助手

**協作方式**: 採用「迭代式 prompt + 立即測試」的協作模式:

1. **初步理解規格**: 將規格 PDF 整段提供給 AI, 請其萃取需求; 再針對不清楚的細節 (如 frame size 條件、acceptance test 流程) 追問
2. **分階段實作**: 將整個系統拆分成 task_generator → Phase 1 ILP → online phase → advanced scheduler → evaluator / verifier 五個階段, 每階段獨立 prompt + 測試後再合併
3. **錯誤驅動的迭代**: 每次測試找到 bug 或不合理結果 (例如 joint LP 偏好 gen 而過度增加 f2), 描述觀察到的現象, 請 AI 分析根因並修正
4. **設計取捨討論**: 對於非單純對錯的設計選擇 (例如「sporadic 是否要策略性拒絕」、「joint LP 是否包含電池」), 先請 AI 分析利弊, 再做決定
5. **驗證與報告**: 完成實作後請 AI 依照規格比對驗證, 並協助撰寫 README 與本份報告

**Prompt 策略**:
- **具體化問題**: 「在這個 demo input 下我看到 f2 比預期高 $20k, 原因是什麼?」優於「我的程式有 bug」
- **要求量化分析**: 請 AI 在做設計決策前先實際跑資料 / 算數字
- **拒絕「就照建議做」**: 在 AI 提出修改建議時, 要求說明為什麼、有什麼權衡、什麼情況下不適用
- **驗證輸出**: 每次重大改動都跑 `verifier.py` 與 `evaluator.py`, 不單純信任 AI 的「應該 OK」

### 5.2 實作心得

1. **規格的深層意義往往不在字面上**: 例如 C4 第三項規定 aperiodic 必須在 H 前執行完 `e` 小時, 這個約束讓「直接 skip aperiodic」變成違規。我們最初的版本確實有 silent skip 的 bug, 是後來透過 verifier 才發現。

2. **online 與 offline 視角的差別非常微妙**: Sporadic 規格上是「online 到達」, 但 demo 時所有資料是提前給的。要不要利用這個「實質知道」的優勢? 我們選擇用 lookahead-aware 的策略性拒絕, 並把這個假設明確寫進報告。

3. **多個 phase 的 local optimization 不等於 global optimization**: L1 的 Phase 1 ILP 是 periodic 全域最佳, 但加上 online phase 之後, 最終的 `α·f1 + f2 + f3` 不是全域最佳。理解這一點對 demo 講解非常重要。

4. **L2 比 L1 「差」是正確的結果**: 一開始看到 L2 objective 比 L1 差很多時懷疑是不是有 bug。後來才理解: L2 模擬了現實的隨機性 / 損耗 / 違約成本, 這些都是 L1 沒有的。L2 比 L1 差是「越接近現實越貴」的合理現象。

5. **電池利用比想像中複雜**: 第一次 demo 時 sporadic_value_rate 只有 0.643, 原因不是演算法錯, 而是電池的 SOC 鏈 (chain) 限制過於保守 — 不允許「現在多放電、未來多充電」的補償操作。引入 SOC 補償機制 (在未來非充電時段增加 chg 並減少 sell) 後, sporadic rate 提升到 0.857。

6. **驗證器是最好的朋友**: 多次發現 bug 都是靠 `verifier.py` / `verifier_advanced.py` 找到。例如 C20 違規 (PV capping 後 k 未同步調整)、C1 違規 (雙重 placement)、C16 違規 (L1 plan 餵入 L2 後 SOC 不一致), 都是驗證器先報異常, 我們才追蹤源頭。

### 5.3 Demo 後的修正

第一次 demo 後發現以下問題並進行修正:

1. **Sporadic 接受率 < 0.7 (規格 4-3 未達滿分)**:
   - **原因**: `_battery_avail` 的 SOC 鏈檢查過於嚴格, 不允許「現在多放電 + 未來補充」的補償。即使電池在 sporadic 到達時段有充足電量, 也因為遠未來的 SOC 觸底而被擋住。
   - **修正**: 新增 `_battery_avail` 的 compensation-aware 邏輯, 並讓 `_commit_battery` 實際執行補償 (在未來非充電時段增加 chg、減少 sell, 並重新計算 SOC 軌跡)。
   - **結果**: sporadic_value_rate 從 0.643 提升到 0.857。

2. **L2 的差異**: 同樣的補償邏輯也加入 L2, 但因 L2 初始 adaptation replan 將電池充電集中到部分時段, sporadic 仍有 2 個因 C19 (充放電互斥) 而無法接受。這在 L2 報告中明確說明為「真實電池模型的合理結果」, 而非 bug。

### 5.4 未來可改進方向

1. **Joint LP 加入電池變數**: 目前 commit 階段對連續時段使用單時段 commit 而非 joint LP, 是因為 joint LP 不含電池, 與 battery-aware feasibility 不一致。若將電池與 SOC chain 限制式建模進 joint LP, 可進一步降低成本。
2. **Strategic reject 動態調整 threshold**: 目前 `$1500` 是靜態值; 可改為依當前資源使用率動態調整。
3. **L2 加入 day-ahead commit 滾動更新**: 目前 commit 鎖定為 L1 的 sell; 可考慮在 L2 內部允許定期 (如每 24h) 更新 commit, 降低違約罰金。
4. **Phase 1 ILP 將 sporadic 平均空閒度納入考量**: 若 Phase 1 知道未來會有 sporadic 到達, 可預留電池在那些時段不要 charging, 提升 sporadic 接受率。

---

## 附錄 — 主要檔案位置

| 檔案 | 內容 |
|---|---|
| `src/task_generator.py` | Periodic task set 隨機產生 (CLI 支援 seed 與 target_f) |
| `src/scheduler.py` | Level 1 Phase 1 ILP + online phase (含電池 SOC 補償) |
| `src/evaluator.py` | Level 1 評估指標計算 |
| `src/verifier.py` | Level 1 限制式驗證 |
| `src/advanced_scheduler.py` | Level 2 動態滾動排程 |
| `src/evaluator_advanced.py` | Level 2 評估指標計算 |
| `src/verifier_advanced.py` | Level 2 限制式驗證 |
| `README.md` | 使用說明文件 |

---

*報告版本: v2.0 — 最終提交版 (含 demo 後修正)*
