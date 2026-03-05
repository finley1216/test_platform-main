# Batch 影片分析 Pipeline 實作困難總結（報告用）

## 一、我們在做什麼？

- **目標**：實作「多段影片一次送給 Qwen2.5-VL」的 **batch 推論**，讓模型只載入一次、一次處理多段，提高吞吐量。
- **流程**：客戶端上傳多段影片 → 後端一次取樣多段影格 → 組合成 batch → 呼叫 `processor.apply_chat_template` + `model.generate` → 一次解碼多個輸出 → 解析成每段的 events / summary。
- **環境**：後端用 Hugging Face 的 Qwen2.5-VL-7B、YOLO-World、ReID（torchvision 或 torchreid），跑在 Docker 裡、單機單卡 GPU。

---

## 二、遇到的困難（條列）

### 1. Meta Tensor 錯誤（Cannot copy out of meta tensor; no data!）

- **現象**：載入 Qwen2.5-VL 或 ReID 時程式崩潰，錯誤為「Cannot copy out of meta tensor; no data!」。
- **原因（淺顯說）**：  
  - 使用 `device_map="auto"` 時，Hugging Face 的 accelerate 會先用「空的 placeholder（meta tensor）」佔位，再慢慢把權重填進去。  
  - 若載入後我們又手動呼叫 `model.to(device)`，就等於要「把這些還沒填好的 placeholder 搬到 GPU」，此時就會觸發上述錯誤。
- **對應做法**：  
  - Qwen 載入時使用 `device_map="auto"`（或 `"cuda:0"`），**不再**在載入後呼叫 `model.to(device)`，只呼叫 `model.eval()`。  
  - ReID 若與 Qwen 同 process，為避免被 meta 狀態影響，改為**在 CPU 上建立模型**（`with torch.device("cpu"):` 建立 ResNet50），再 `.to(cuda)`，不在 meta 階段做設備移動。

---

### 2. 多個 Uvicorn Worker 重複載入大模型（VRAM 爆掉）

- **現象**：`nvidia-smi` 看到兩個（或更多）python 進程，每個都佔十幾 GB，總 VRAM 超過單卡容量。
- **原因（淺顯說）**：  
  - Uvicorn 的 `workers=N` 會啟動 **N 個獨立 process**。  
  - 我們的「單例」只在**同一個 process 內**有效，所以每個 worker 都會各自載入一份 Qwen + YOLO + ReID。  
  - 例如 WORKERS=4 且每個 process 約 17GB → 約需 68GB VRAM，單卡無法負擔。
- **對應做法**：  
  - 將 **WORKERS 改為 1**（在 `.env` 與 `docker-compose.yml` 中設定）。  
  - 在啟動時若偵測到 WORKERS>1，印出 VRAM 提醒。  
  - 註明：要「只載入一次、跑 batch」就是單一 worker；多 worker 是給多 process 負載平衡用，會多份模型、多份 VRAM。

---

### 3. 多執行緒同時第一次呼叫時「並發載入」導致 OOM

- **現象**：多個 API 請求同時進來時，日誌出現多個「離線載入」「Loading weights」，接著 CUDA OOM。
- **原因（淺顯說）**：  
  - 雖然設計是「單例：只載入一次」，但多個 **thread** 同時第一次呼叫 `get_model_and_processor()` 時，在「檢查快取」與「真正載入」之間沒有鎖，會有多個 thread 同時進入 `from_pretrained()`，等於同時載入多份 Qwen，VRAM 瞬間爆掉。
- **對應做法**：  
  - 使用 **threading.Lock()**：只有一個 thread 能執行「載入模型」這段程式，其他 thread 在鎖外等待，載入完成後一起複用同一份模型。  
  - 即「單例 + 載入鎖」，避免並發載入。

---

### 4. ReID 載入時 OOM 或與 Qwen 搶顯存

- **現象**：Qwen 載入完成後，接著載入 ReID（torchvision ResNet50）時出現 CUDA OOM，或 ReID 也出現 meta tensor 相關錯誤。
- **原因（淺顯說）**：  
  - 同一個 process 內，Qwen 已佔用大部分 GPU 顯存（約 15–17GB），再在 GPU 上建立 ResNet50 會不夠記憶體。  
  - 若同 process 內有 meta tensor 狀態，在 GPU 上建立 ReID 也可能被影響。
- **對應做法**：  
  - ReID 改為**先在 CPU 上建立**（`with torch.device("cpu"):` 建立模型並替換 `fc`），再 `.to(cuda).eval()`，推論時在 GPU。  
  - 必要時可考慮 ReID 全程在 CPU（較慢但省顯存），或確保只在一份 Qwen 載完且無其他大模型同時載入時再載 ReID。

---

### 5. 推論/解析錯誤：string indices must be integers, not 'str'

- **現象**：Batch 編碼/推論失敗或單段推論失敗，錯誤訊息為「string indices must be integers, not 'str'」，導致整批結果失敗或改為依序推論後仍失敗。
- **原因（淺顯說）**：  
  - 程式某處用「字串當 key」去索引一個物件（例如 `obj["events"]`），但該物件實際是**字串**或**串列**，在 Python 裡字串/串列不能用字串 key 索引，就會出現這個錯誤。  
  - 可能來源包括：  
    (1) 模型輸出的 JSON 頂層是**陣列**（例如 `[{...}]`），程式卻當成**物件**用 `.get("events")` 或 `["events"]`；  
    (2) `apply_chat_template` 在** batch 模式**下回傳的格式與預期不同（例如回傳 list 而非 dict）；  
    (3) 解碼後的 `output_text` 或中間變數在少數情況下不是字串，卻被當成 dict 使用。
- **對應做法**：  
  - **解析階段**：先判斷解析出的 `combined` 是 dict 還是 list；若是 list 且第一個元素是 dict，則取第一個元素當成一個物件；若都不是則當成空物件 `{}`，再從中取 `events`、`persons`、`summary`。  
  - **型別防護**：`persons` 僅在為 list 時使用，否則用空陣列；傳入 `_parse_one_output` 的保證是字串（非 str 則轉成 str）。  
  - **例外處理**：在 `_parse_one_output` 與 `infer_one` 的解析區塊外加 try/except，若出現「string indices」或「must be integers」等錯誤，改回傳**安全預設值**（空 events、空 persons、原始文字當 summary），不讓錯誤往上拋導致整批失敗。  
  - **Batch 路徑**：若 `apply_chat_template` 回傳的是 list，則不強行當 dict 用，改為**依序呼叫**單段推論，避免在 batch 路徑裡對 list 做 `inputs["input_ids"]` 等操作。

---

### 6. 套件版本與環境需配合 Qwen2.5-VL

- **現象**：舊版 transformers / accelerate 可能不支援 Qwen2.5-VL 的架構或 `device_map="auto"` 行為，導致載入失敗或行為異常。
- **原因（淺顯說）**：Qwen2.5-VL 是較新的模型，需要較新版的 transformers 與 accelerate 才能正確載入與分配設備。
- **對應做法**：  
  - 在 Dockerfile 中安裝/升級：`pip install -U transformers accelerate`。  
  - 設定環境變數以減輕顯存碎片：`ENV PYTORCH_ALLOC_CONF=expandable_segments:True`。

---

### 7. torchreid 未安裝或缺少 gdown

- **現象**：日誌出現「No module named 'torchreid'」或「No module named 'gdown'」，ReID 改走 torchvision 備用方案。
- **原因（淺顯說）**：  
  - 精準 ReID 使用 torchreid 套件，但 Docker 映像內未安裝，或 torchreid 依賴的 gdown 未安裝。  
  - 不影響「有備用方案」時的運作，但若希望使用 torchreid 則需補齊依賴。
- **對應做法**：  
  - 在 Dockerfile 中可選安裝：`pip install torchreid`（若需要可一併安裝 gdown）。  
  - 未安裝時維持現有邏輯：fallback 到 torchvision ResNet50，並在 CPU 上建立再 `.to(cuda)`，避免 OOM 與 meta tensor 問題。

---

### 8. offload_buffers 與單卡容量界線

- **現象**：日誌出現「Current model requires 592 bytes of buffer for offloaded layers... If you are experiencing a OOM later, please consider using offload_buffers=True」。
- **原因（淺顯說）**：使用 `device_map="auto"` 時，accelerate 可能會把部分層 offload 到 CPU；若 GPU 剩餘空間很小，會建議開啟 `offload_buffers=True` 以減少 OOM 風險。
- **對應做法**：  
  - 目前以單 worker、單卡為主，若仍 OOM 可考慮在 `from_pretrained` 中加上 `offload_buffers=True` 或調小 batch 大小（`max_inference_batch_size`）。  
  - 先以 WORKERS=1、適度 batch size（例如 2～4）與 PYTORCH_ALLOC_CONF 優化為主。

---

## 三、總結對照表

| 困難 | 現象關鍵字 | 主要對應做法 |
|------|------------|--------------|
| Meta tensor | Cannot copy out of meta tensor | device_map 搭配「不」再 .to(device)；ReID 在 CPU 建立再 .to(cuda) |
| 多 worker 重複載入 | 多個 python 各佔十幾 GB | WORKERS=1；單例僅限同 process |
| 並發載入 OOM | 多個「離線載入」「Loading weights」同時出現 | 單例 + threading.Lock() 載入鎖 |
| ReID OOM / meta | ReID 載入時 OOM 或 meta 錯誤 | ReID 在 CPU 建立後再 .to(cuda) |
| string indices 錯誤 | string indices must be integers, not 'str' | 解析前正規化 list/dict、try/except 安全預設、batch 時 inputs 為 list 則改依序 |
| 套件版本 | 載入或 device_map 異常 | pip install -U transformers accelerate；PYTORCH_ALLOC_CONF |
| torchreid / gdown | No module named 'torchreid' 或 'gdown' | 可選安裝 torchreid、gdown；否則用 torchvision 備用 |
| offload 提示 | offloaded layers... offload_buffers=True | 必要時開啟 offload_buffers 或調小 batch |

---

## 四、報告時可強調的幾句話（淺顯版）

1. **Batch 的目標**：多段影片一次送進模型，模型只載入一次，提高效率。  
2. **單例只在一個 process 裡有效**：開多個 Uvicorn worker 就會有多份模型，所以我們用一個 worker，避免 VRAM 爆掉。  
3. **載入時不能「又用 device_map 又手動 .to(device)」**：否則會觸發 meta tensor 錯誤；ReID 改在 CPU 建立再搬到 GPU，避免同 process 內的 meta 與 OOM。  
4. **多個請求同時第一次進來時**：要用「鎖」保證只有一個 thread 在載入模型，其他等待後複用，才不會並發載入多份導致 OOM。  
5. **模型輸出不一定永遠是「一個 JSON 物件」**：可能是陣列或格式不同，解析前要正規化成「一個物件」，並對解析錯誤做安全回傳，才不會整批失敗。  
6. **環境要跟上模型**：Qwen2.5-VL 需要較新的 transformers/accelerate，並可設定 PYTORCH_ALLOC_CONF 優化顯存。

以上可直接用於報告的「實作困難與對應方案」章節。
