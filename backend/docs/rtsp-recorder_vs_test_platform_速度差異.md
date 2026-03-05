# rtsp-recorder vs test_segment_pipeline_rtsp 速度差異說明

同為「每段 10 秒影片」分析，實測約 **rtsp-recorder ~1.05s / 段**、**test_platform 腳本 ~5.5s / 段**。  
若**只量 Ollama 分析**、同一模型：**rtsp-recorder ~800ms**、**test_platform ~2s** —— 差異主要來自 **VLM 輸入的 prompt 長度**（見下方 § 2.1）。

---

**幾句話說明時間差異：**  
兩邊的 10 秒片段都是從 RTSP 擷取、在本地切割好的。差異在「誰做分析、有沒有上傳」：**rtsp-recorder** 的 ~1.05s 只量「同一台機器上的分析」（讀檔 → VLM + YOLO + ReID → 寫 DB），不包含網路傳輸。**test_platform** 的 ~5.5s 是從「送出 HTTP 請求」到「收到回應」的端到端時間，包含：本機把 10s 影片上傳到後端、後端收檔與分析、再回傳結果。所以 1.05s 是「純分析」、5.5s 是「上傳 + 後端分析 + 下載」；掐表起終點不同，不能直接比。

**畫流程圖時可這樣區分兩條流程：**

| 流程 | 計時起點 → 終點 | 流程圖要畫的步驟（依序） |
|------|------------------|---------------------------|
| **rtsp-recorder** | 開始分析 → 寫完 DB | ① 讀取 10s 片段（本地檔）→ ② VLM（Ollama）→ ③ YOLO → ④ ReID → ⑤ 寫入 Postgres。**不畫**：上傳、後端收檔。 |
| **test_platform** | 送出請求 → 收到回應 | ① 本機：呼叫 API（開始計時）→ ② **上傳** 10s 影片到後端 → ③ 後端：收檔、寫 temp、prepare_segments → ④ VLM → ⑤ YOLO → ⑥ ReID → ⑦ 寫 DB、組 JSON → ⑧ **回傳**回應到本機（計時結束）。 |

重點：兩條流程的「影片來源」都是本地切好的 10s 檔；差別是 rtsp-recorder 在同一台機器做完分析，test_platform 多出「上傳到後端」與「後端做完再回傳」這兩段，所以端到端時間較長。

---

## 1. 計時範圍不同（最重要）

| 專案 | 計時內容 |
|------|----------|
| **rtsp-recorder** | 只量 **本機分析**：VLM + YOLO + ReID + 寫 DB。影片已在本地，不包含上傳。 |
| **test_platform 腳本** | `process_duration_sec` = 從呼叫 `upload_segment_to_api()` 到收到回應 = **上傳整支 10s MP4** + **後端完整流程** + 下載回應。 |

### 1.1 概念說明：兩邊「從哪裡開始掐表、到哪裡結束」

- **rtsp-recorder**  
  - **情境**：10 秒片段由 RTSP 擷取、切割後寫入本機（例如 `video/某路/xxx.mp4`），**在同一台機器**直接做分析。  
  - **計時**：從「開始對這支影片做分析」到「分析結束並寫完 DB」為止。  
  - **包含**：讀檔 → VLM（Ollama）→ YOLO → ReID → 寫入 Postgres。  
  - **不包含**：沒有「把影片傳到別台機器」這一步，所以**沒有網路傳輸時間**，也沒有「後端收檔、切割」等 I/O。  
  - 可以想成：**同一台機器上，純算「分析這段影片」要花多久**。

- **test_platform 腳本**  
  - **情境**：同樣是從 RTSP 擷取、在本機切好的 10 秒片段；腳本再**透過 HTTP 上傳到後端**，由後端做分析。  
  - **計時**：從腳本呼叫 `upload_segment_to_api(...)`（開始送 request）到該函式 **return**（收到後端 HTTP 回應）為止。  
  - **包含**：  
    1. **上傳**：把整支 10s MP4（約 1–3 MB）經網路傳到後端；  
    2. **後端**：收檔 → 寫入 temp → prepare_segments（copy + FFmpeg 切割）→ VLM → YOLO → ReID → 寫 DB → 組 JSON → 回傳；  
    3. **下載**：後端把 JSON 回應傳回腳本。  
  - 可以想成：**從「按下送出」到「拿到結果」的整段使用者可感知時間**（含網路與後端所有步驟）。

因此：**rtsp-recorder 的 ~1.05s** 只反映「本機分析」；**test_platform 的 ~5.5s** 反映「上傳 + 後端整段 + 下載」。兩邊**掐表的起點與終點不同**，不能直接相減或當成同一種指標；要比「純分析」應以後端內部的 VLM / YOLO / ReID 計時為準。

### 1.2 因此 5.5s 裡大致包含

- **上傳時間**：10 秒 MP4（約 1–3 MB）經 HTTP 上傳到後端  
- **後端**：收檔 → 寫入 temp → prepare_segments → VLM → YOLO → ReID → 寫 DB → save_json → 回傳  

rtsp-recorder 的 1.05s **不包含**上傳與後端 I/O，所以兩邊數字本來就不可直接比。

---

## 2. VLM 差異（同模型 800ms vs 2s 的主因）

### 2.1 Prompt 長度（**影響最大**）

| 專案 | 事件 + 摘要 prompt 來源 | 約略長度 |
|------|-------------------------|----------|
| **rtsp-recorder** | 環境變數或預設短句 | 事件 ~30 字、摘要 ~50 字，合計 **~100 字** |
| **test_platform** | `prompts/frame_prompt.md` + `prompts/summary_prompt.md` | 事件 **~1200 字**、摘要 **~650 字**，合計 **~1850 字** |

腳本送 `event_detection_prompt=""`、`summary_prompt=""` 時，後端會改用 **DEF_EVENT_PROMPT / DEF_SUMMARY_PROMPT**，來自 `main.py` → `prompts` 套件，即上述長版 md。  
送給 Ollama 的 **文字 token 多約 10–20 倍**，同模型、同幀數與解析度下，**VLM 從 ~800ms 變 ~2s 是合理結果**。

### 2.2 幀數與解析度（腳本已對齊）

| 參數 | rtsp-recorder | test_platform 腳本（已改） |
|------|----------------|----------------------------|
| **frames_per_segment** | **4** | **4** |
| **target_short** | **480** | **480** |

腳本已改為 4 幀、480px；若仍用 5 幀 / 720px 會再增加 VLM 負載。

---

## 3. 後端額外步驟（test_platform 才有）

test_platform 收到上傳後會：

1. **收檔**：`file.file.read()` 讀完整個上傳檔。
2. **寫入 temp**：存成暫存檔。
3. **prepare_segments**：
   - `shutil.copy2(source_path, backup_path)` 再拷一份到 segment 目錄；
   - `_split_one_video()` 用 FFmpeg 做切割（即使只有一段 10s，也會跑一次 FFmpeg）。

rtsp-recorder 是 **直接開本地 10s 檔** 做 VLM/YOLO，沒有上傳、沒有 copy、沒有 FFmpeg。這些都會算進你看到的 5.5s。

---

## 4. 小結：為什麼差這麼多

- **計時範圍**：5.5s 含上傳 + 整段後端；1.05s 只含本機分析。
- **VLM 負載**：5 幀 720px vs 4 幀 480px → 後端 VLM 明顯較慢。
- **後端額外成本**：收檔、copy、FFmpeg 切割，都會增加時間。

若要讓「每段分析」的負載與 rtsp-recorder 接近，應讓腳本送 **frames_per_segment=4、target_short=480**（與 rtsp-recorder 預設一致）。剩餘差距主要就是 **上傳時間 + 後端 I/O/切割**。

---

## 5. 建議

1. **VLM 要接近 800ms**：讓後端在「比對用」情境使用與 rtsp-recorder **相同的短 prompt**（見下方實作方式）。
2. **腳本已對齊幀數/解析度**：`frames_per_segment=4`、`target_short=480`。
3. **若要精準對齊「純分析」時間**：在後端回傳或記錄 **vlm_sec / yolo_sec / reid_sec**，排除上傳與 prepare_segments。

### 讓 test_platform 使用「短 prompt」以對齊 rtsp-recorder 的 VLM 時間

- **方式 A**：呼叫 API 時**明確帶上短 prompt**，不送空字串，後端就不會替換成長版：
  - `event_detection_prompt`: `請根據提供的影格輸出事件 JSON。`
  - `summary_prompt`: `請觀察畫面中的人車流動與任何異常狀況，並將其彙整為一句約 50-100 字的繁體中文描述。`
- **方式 B**：在後端用環境變數覆蓋預設（若 `main.py` 或 prompts 有支援從 env 讀取）。
- **方式 C**：後端加一個開關（例如 `use_short_prompts=True`）或專用參數，當為「比對 / 省時」時改用與 rtsp-recorder 相同的兩句短 prompt。
