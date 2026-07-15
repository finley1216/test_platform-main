# 對照實驗：人員追蹤_20260507（啟發式 vs LLR）

資料：`../output/query_filter_merge/人員追蹤_20260507`  
指令：

```bash
# 全量校準（目前因缺 tracking_rows JSON 無法完成，見 §7）
python3 calibrate.py \
  --tracking-output ../dag_0507/botsort ../cost_path_experiment/botsort \
  --out ../output/path_enum_llr/calibration.pkl

python3 path_enum_scoring.py ../output/query_filter_merge/人員追蹤_20260507 --sim-min 0.85
python3 path_enum_llr.py ../output/query_filter_merge/人員追蹤_20260507 --sim-min 0.85
```

輸出目錄：
- 舊：`../output/path_enum/`
- 新：`../output/path_enum_llr/`
- 校準：`../output/path_enum_llr/calibration.pkl`、`calibration_report.txt`、`emb_same_diff_hist.png`

本次 LLR 修正（不改 `path_enum_scoring.py`）：
1. **handoff**：OVERLAP / H 共視且 `dt≤2s` → `LLR_dt=n/a`，邊證據 = LLR_dH + LLR_emb
2. **收縮**：`w=n/(n+10)`（emb／dH）；dt 擬合≥20 或 PRIOR-PHYSICAL → `w=1` 並標注
3. **節點證據 = 0**：`node evidence disabled pending calibration`

---

## 1. 候選／圖結構計數（應完全相同）

| 項目 | 舊 `path_enum_scoring` | 新 `path_enum_llr` | 一致？ |
|------|------------------------|--------------------|--------|
| 候選 track 數 | 26 | 26 | 是 |
| 合法邊數 | 41 | 41 | 是 |
| 枚舉路徑數（含前綴） | 648 | 648 | 是 |
| 極大路徑數 | 235 | 235 | 是 |

---

## 2. 校準樣本數：新舊對照

| 項目 | 舊（query_filter 單日 0507） | 新（BoT-SORT 全量多日） |
|------|------------------------------|-------------------------|
| 總 track 數 | 26（SIM≥0.85 + filter kept） | **無法重校**（缺 `tracking_rows*.json`） |
| 正樣本 H 對 | 1 | — |
| 正樣本 dt 觀測 | 14（每對皆 &lt;20） | — |
| 負樣本同鏡 | 1 | — |
| 負樣本跨鏡 | 0 | — |
| emb same / diff | 1 / 1 | — |

本輪 `calibrate.py --tracking-output` 依規格中止並回報（見 §7）：  
**有** per-crop emb cache 與 crop_time_mapping，**沒有**全量 track→crops JSON，故 **仍沿用舊 `calibration.pkl`**（先驗為主）做計分修正對照。

舊擬合摘要（未變）：

- `emb|same` n=1 → PRIOR Normal(0.95, 0.03)；收縮 **w=1/11≈0.091**
- `emb|diff` n=1 → PRIOR Normal(0.70, 0.10)
- `dH|same` n=1 → PRIOR HalfNormal(σ=40)；收縮 **w≈0.091**
- 全部 dt 鏡頭對 → PRIOR-PHYSICAL LogNormal(τ=8.0 佔位)，**w=1**

---

## 3. 修正後 Top-3 vs 舊法 Top-3

### 舊（啟發式）Top-3

| # | score | 路徑 |
|---|-------|------|
| 1 | 208.13 | `K8-09_7 → K8-08_30 → K8-01_7 → K8-07_40 → K8-23_8 → K8-22_19` |
| 2 | 202.81 | `K8-09_7 → K8-01_7 → K8-08_30 → K8-07_40 → K8-23_8 → K8-22_19` |
| 3 | 195.26 | `K8-09_7 → K8-08_30 → K8-01_7 → K8-07_40 → K8-23_8 → K8-01_50 → K8-08_77 → K8-01_62` |

### 新（LLR，handoff＋收縮＋node=0）Top-3

| # | score | P | 路徑 |
|---|-------|---|------|
| 1 | 6.1916 | **0.722045** | `K8-09_7 → K8-08_30 → K8-01_7 → K8-07_40 → K8-23_8 → K8-22_19` |
| 2 | 4.2141 | 0.099949 | `K8-09_7 → K8-01_7 → K8-07_40 → K8-23_8 → K8-22_19` |
| 3 | 3.3720 | 0.043059 | `K8-09_7 → K8-08_30 → K8-07_40 → K8-23_8 → K8-22_19` |

**Top-1 與舊法相同。** P 由先前 0.44 升到 **0.722**（符合預期）。

交叉排名：

| 路徑 | 舊排名 / score | 修正前 LLR 排名 / P | 修正後 LLR 排名 / P |
|------|----------------|---------------------|---------------------|
| 舊／新 Top-1（同上） | #1 / 208.13 | #55 / ≈1.3e−8 | **#1 / 0.722** |
| 修正前 LLR Top-1（長路徑） | #15 / 159.21 | #1 / 0.443 | 已掉出前段（長路徑節點加分消失） |

---

## 4. handoff 邊修正前後 LLR 對照

兩條問題邊（修正前被 `LogNormal(τ≈8)` 打到 ≈−23）：

### 4.1 `K8-08_30 → K8-01_7`（H=15.2px，dt=0）

| | LLR_dt | LLR_emb | LLR_dH | edge / effective |
|--|--------|---------|--------|------------------|
| **修正前** | **−22.844** | +2.833 | +2.698 | **−17.312** |
| **修正後** | **n/a**（dt_model=handoff） | +0.258（raw 2.833 × w=0.091） | +0.245（raw 2.698 × w=0.091） | **+0.503** |

### 4.2 `K8-23_8 → K8-22_19`（OVERLAP，H=77.1px，dt=0）

| | LLR_dt | LLR_emb | LLR_dH | edge / effective |
|--|--------|---------|--------|------------------|
| **修正前** | **−22.844** | +3.150 | +0.915 | **−18.779** |
| **修正後** | **n/a**（dt_model=handoff） | +0.286（raw 3.150 × w=0.091） | +0.083（raw 0.915 × w=0.091） | **+0.369** |

註：`08↔01` 不在 `PERSON_OVERLAP_PAIRS`，但有上傳 Homography；實作將 **OVERLAP_PAIRS 或 `h_dist is not None`** 且 `dt≤2` 視為 handoff（否則這條無法按你點名的邊修正）。

---

## 5. 修正後 Top-1 逐邊分解

路徑：`K8-09_7 → K8-08_30 → K8-01_7 → K8-07_40 → K8-23_8 → K8-22_19`  
LLR 總分 **6.1916**；節點證據全 0。

| from → to | hop | dt | dt_model | emb | d_H | LLR_dt | LLR_emb | LLR_dH | raw_LLR | w | effective_LLR |
|-----------|-----|-----|----------|-----|-----|--------|---------|--------|---------|---|----------------|
| 09_7 → 08_30 | 1 | 5.2 | transit | 0.962 | — | +2.549 | +0.414 | — | +7.099 | 0.417 | +2.962 |
| 08_30 → 01_7 | 1 | 0.0 | **handoff** | 0.915 | 15.2 | **n/a** | +0.258 | +0.245 | +5.531 | 0.091 | +0.503 |
| 01_7 → 07_40 | 2 | 25.1 | transit | 0.923 | — | +1.237 | +0.297 | — | +4.500 | 0.341 | +1.534 |
| 07_40 → 23_8 | 2 | 34.1 | transit | 0.938 | — | +0.464 | +0.359 | — | +4.415 | 0.186 | +0.823 |
| 23_8 → 22_19 | 1 | 0.0 | **handoff** | 0.920 | 77.1 | **n/a** | +0.286 | +0.083 | +4.064 | 0.091 | +0.369 |

transit 邊的 dt 皆標 **PRIOR-PHYSICAL**（τ=8.0 佔位；w_dt=1）。

---

## 6. 機率與替代路徑

- Top-1 path_probability：**0.722045**（修正前 0.443）
- 最佳不共用 track 替代：`K8-08_77 → K8-01_62`  
  - P_alt = 0.002397  
  - ratio(Top1 / alt) = **301.2**

---

## 7. 全量校準資料盤點（阻斷重校）

`calibrate.py --tracking-output ../dag_0507/botsort ../cost_path_experiment/botsort` 退出碼 2：

| 資源 | 狀態 |
|------|------|
| `tracking_rows*.json`（全量 track 分組） | **無**（0507/0528 各鏡只有 `tracking_collage.png`） |
| per-crop embedding | **有**：`ASE/output/embed_cache/人員追蹤_*_k*/person_clipreid_embeddings_cache.pkl` |
| crop 時間／box mapping | **有**：`ASE/output/人員追蹤_*_crop_time_mapping.json` |
| 格式範例 | 僅 `BoT-SORT-K809/output/k809/tracking_rows_*.json` |

依你的規則：**不自行用 query_filter_merge 或散落 crop emb 替代全量 track**。  
請重跑 BoT-SORT 並 dump `tracking_rows`（含 `crops[]`）後再：

```bash
python3 calibrate.py \
  --tracking-output <dir_0507> <dir_0528> \
  --out ../output/path_enum_llr/calibration.pkl
```
