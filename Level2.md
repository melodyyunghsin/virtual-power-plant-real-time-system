# Level 2 進階動態排程 - 放寬假設與數學建模說明



## I. 放寬假設一：再生能源不確定性 (Renewable Uncertainty)
原本假設再生能源完全遵照預測發電，進階模型引入了實際觀測值與預測誤差。

### 1. 即時實際發電量上限限制 (Real-time Actual Generation Limit)
* **放寬說明**：在即時排程推進到當下時間點 $t_{now}$ 時，再生能源的最大出力必須受限於「真實觀測到的發電比例」，而非原先的預測值。
* **新增參數**：$renewactual_{i,t} \in [0,1]$ 為實際發電比例。
* **數學限制式**：
  $$P_{i,t} \le renewmax_i \cdot \Delta t \cdot renewactual_{i,t}, \quad \forall i \in I_r, t = t_{now}$$
* **程式碼對應**：`bound = cap * self.pv_actual[pv][t]`

### 2. 預測誤差之強健性邊界限制 (Robust Forecast Bound)
* **放寬說明**：對於未來的時間點 $t > t_{now}$，為避免過度信任預測而導致未來的排程不可行，系統將再生能源的預測值乘上保守的安全邊際（扣除誤差標準差）。
* **新增參數**：$\sigma_{err} \in [0,1]$ 為預測誤差標準差。
* **數學限制式**：
  $$P_{i,t} \le renewmax_i \cdot \Delta t \cdot renewforecast_{i,t} \cdot (1 - \sigma_{err}), \quad \forall i \in I_r, t > t_{now}$$
* **程式碼對應**：`bound = cap * self.pv_forecast[pv][t] * (1 - self.err_std)`

---

## II. 放寬假設二：儲能設備真實運作情境 (Realistic Storage Operations)
Level 1 將電池視為無損耗的理想容器，進階模型加入了熱力學損耗、硬體限制與老化成本。

### 3. 充放電轉換效率限制 (Charge & Discharge Efficiency)
* **放寬說明**：電池在充電與放電時會有能量轉換損耗。實際增加的電量會小於充入系統的電量，實際減少的電量會大於放出的電量。
* **新增參數**：$\eta_c \in (0,1]$ 為充電效率，$\eta_d \in (0,1]$ 為放電效率。
* **數學限制式**：
  $$SOC_{i,t} = SOC_{i,t-1} + \eta_c \sum_{j \in J_{chg}} k_{j,i,t} - \frac{P_{i,t}}{\eta_d}, \quad \forall i \in I_b, \forall t \in T$$
* **程式碼對應**：`+ chg[b][t] * eta_c - dis[b][t] / eta_d`

### 4. 電池自放電限制 (Self-Discharge Rate)
* **放寬說明**：儲能設備即便處於閒置狀態，每一時間段仍會隨著時間自然流失固定比例的電量。
* **新增參數**：$\sigma_{self} \in [0,1]$ 為自放電率。
* **數學限制式**：
  $$SOC_{i,t} = SOC_{i,t-1} \cdot (1 - \sigma_{self}) + (\text{Charge/Discharge Dynamics}), \quad \forall i \in I_b, \forall t \in T$$
* **程式碼對應**：`prev * (1 - sigma)`

### 5. 依電量衰減之放電功率上限 (SOC-Dependent Power Limit)
* **放寬說明**：當電池的剩餘電量 (SOC) 低於最大容量的 30% 時，為了保護電池硬體，其「最大放電功率」會隨電量線性遞減。
* **新增變數**：連續輔助變數 $sfrac_{i,t} \in [0,1]$。
* **數學限制式**：
  $$sfrac_{i,t} \cdot (0.3 \cdot stomax_i) \le SOC_{i,t}$$
  $$P_{i,t} \le dis_i \cdot \Delta t \cdot sfrac_{i,t-1}, \quad \forall i \in I_b, \forall t \in T$$
* **程式碼對應**：`sfrac[b][t] * (0.3 * soc_max_b) <= soc[b][t]` 與 `dis[b][t] <= dis_max * prev_sf`

### 6. 電池循環老化成本評估 (Battery Aging Cost)
* **放寬說明**：電池的每一次放電都會帶來硬體的循環壽命耗損，這筆隱性成本需被量化並加入整體目標函數 $f_2$ 中一起最佳化。
* **新增參數**：$cost_{aging, i}$ 為單位放電老化成本 (\$/MWh)。
* **數學限制式**（更新 $f_2$）：
  $$f_2 = \sum_{i \in I_g} \sum_{t \in T} (\text{Gen Cost}) + \sum_{i \in I_b} \sum_{t \in T} (cost_{aging,i} \cdot P_{i,t})$$
* **程式碼對應**：`float(self.bat_by_id[b].get("aging_cost", 0.0)) * P[b][t]`

---

## III. 放寬假設三：彈性市場機制 (Flexible Market Mechanisms)
Level 1 允許隨意改變售電量，進階模型引入了真實電力市場的「日前承諾」、「違約金」與「即時電價」制度。

### 7. 日前售電承諾鎖定 (Day-Ahead Sell Commitment Lock)
* **放寬說明**：在日前排程（Phase 1）中決定的售電量，在即時模擬中被視為不可任意調降的「日前合約承諾值」。
* **新增參數**：常數 $Commit_t \in \mathbb{R}^+$ 為第 $t$ 時段之日前承諾售電量。
* **數學限制式**：
  $$Commit_t = Sell_{t}^{Day\_Ahead\_Plan}$$
* **程式碼對應**：`self.locked_commit = { slot["t"]: float(...) }`

### 8. 售電違約輔助限制式 (Sell Shortfall Linearisation)
* **放寬說明**：若系統在即時運作中，為了補足突發任務而挪用了要賣給市場的電，導致實際售電 $Sell_t$ 低於承諾值，需線性化計算不足量。
* **新增變數**：連續輔助變數 $s\_under_t \ge 0$ 與 $s\_over_t \ge 0$。
* **數學限制式**：
  $$s\_under_t \ge Commit_t - Sell_t, \quad \forall t \in T$$
  $$s\_over_t \ge Sell_t - Commit_t, \quad \forall t \in T$$
* **程式碼對應**：`s_under[t] >= c_t - sell[t]` 與 `s_over[t] >= sell[t] - c_t`

### 9. 違約懲罰與即時市場收益最佳化 (Penalty & Real-time Market Revenue)
* **放寬說明**：售電不足需繳納違約金（以違約費率計算），超額售電（大於承諾值的部分）則以「即時電價浮動係數」重新結算，並整合進目標函數 $f_3$。
* **新增參數**：$R_{cancel}$ 為違約費率，$rt\_factor_t$ 為即時電價乘數。
* **數學限制式**（更新 $f_3$）：
  $$f_3 = \sum_{t \in T} (-\lambda_t \cdot Sell_t + \lambda_t \cdot (1 - \min(rt\_factor_t, 1.0)) \cdot s\_over_t + R_{cancel} \cdot \lambda_t \cdot s\_under_t)$$
  *(註：此式為維持 LP 模型 Bounded 的線性化轉換)*
* **程式碼對應**：`prob += f2 + f3, "TotalCost"`

---

## IV. 動態排程演算法衍生限制 (Dynamic Scheduling Architectural Constraints)
為了實現 Event-Triggered Rolling Horizon，系統必須具備維持過去決策一致性的記憶能力。

### 10. 滾動排程之需求凍結限制 (Frozen Demand Constraint)
* **放寬說明**：在進行局部視窗（Window）的微型 ILP 重排時，必須遵守「不可破壞既有排程」的約定。已承諾的 Periodic 與 Sporadic 任務的供電分配，被強制轉換為不可變更的等式限制條件。
* **新增參數**：$w_{frozen, j, t}$ 為已承諾的任務需求常數。
* **數學限制式**：
  $$\sum_{i \in I} k_{j,i,t} = w_{frozen, j, t}, \quad \forall j \in J_{committed}, \forall t \in T_{window}$$
* **程式碼對應**：`prob += pulp.lpSum(K_frozen[jid][t][i] for i in self.proc_ids) == w`

### 11. 跨視窗狀態繼承與起停延續限制 (Cross-Window Generator State Continuity)
* **放寬說明**：因為排程改為滾動求解，發電機的狀態（開關機、當下輸出）必須在每個 Window 之間傳遞。若某發電機在前一個 Window 剛開機，新 Window 必須強制補足剩餘的 Min-Up Time。
* **新增參數**：$on\_done_{i}$ 與 $off\_done_{i}$ 為滾動到 $t_{now}$ 時已累計的起停時間。
* **數學限制式**：
  $$\sum_{\tau=t_{now}}^{t_{now} + (UT_i - on\_done_i) - 1} \min(1, P_{i,\tau}) = UT_i - on\_done_i, \quad \text{if } u\_init = 1 \text{ and } on\_done_i < UT_i$$
* **程式碼對應**：`prob += u[g][t] == 1, f"finit_on_{g}_{t}"` 
