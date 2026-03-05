# 監控日誌分析報告：`monitor_20260227_135607.log`

**測試情境**：33 路 RTSP，跑 1 分鐘，每段 10s，7 個 API 並行  
**監控時間**：13:56:07 ~ 13:59:19（約 3 分鐘）

---

## 一、整體資源用量

| 資源 | 起始 | 峰值 | 說明 |
|------|------|------|------|
| **RAM 已用** | 56 GB | **67 GB** | 總 154 GB，隨測試進行逐漸上升 |
| **Swap 已用** | 1.6 GB | 1.6 GB | 總 4 GB，維持穩定 |
| **CPU** | 約 60–70% 總和 | 約 75–85% 總和 | 多進程並行 |

---

## 二、各進程資源佔用與對應來源

### 1. Backend API（4 個 Uvicorn Worker）

| PID | 進程識別 | CPU% | RAM% | RSS (MB) | 對應程式碼 |
|-----|----------|------|------|----------|------------|
| 3351503 | spawn_main pipe_handle=9 | 18–24% | 18–21% | **29,700 → 34,100** | `backend/src/start.py` → Uvicorn worker |
| 3351502 | spawn_main pipe_handle=7 | 9–12% | 9–13% | **14,800 → 21,500** | 同上 |
| 3351505 | spawn_main pipe_handle=13 | 14–18% | 2.8–3.2% | 4,600 → 5,200 | 同上 |
| 3351504 | spawn_main pipe_handle=11 | 9–11% | 2.2–2.4% | 3,500 → 4,000 | 同上 |

**合計**：約 **52–65 GB** 由 4 個 backend worker 使用。

**每個 worker 內含**：
- `src/core/model_loader.py`：YOLO-World、ReID (ResNet50)、SentenceTransformer
- `src/services/analysis_service.py`：推理、影片處理、ReID embedding
- PyTorch、OpenCV、ultralytics 等套件

**為何 RAM 差異大？**
- Worker 1、2 較高：可能處理較多片段、較長影片或較多物件偵測，暫存與結果較多
- Worker 3、4 較低：負載較輕或處理較少片段

---

### 2. Ollama（VLM 推理）

| 項目 | 數值 |
|------|------|
| PID | 1359088 |
| CPU% | ~2% |
| RAM% | ~1.4% |
| RSS (CPU RAM) | ~2.3–2.4 GB |

**對應**：`/usr/bin/ollama runner`，qwen2.5vl 模型  
**來源**：`docker-compose` 的 ollama 服務，`backend` 透過 `OLLAMA_BASE=http://ollama:11434` 呼叫

**重要**：上表 RSS 僅為 Ollama 進程的 **CPU RAM**（runner 本身、metadata、部分 buffer）。qwen2.5vl 模型權重（約 8 GB）實際在 **VRAM（GPU）**，需用 `nvidia-smi` 查看。本監控腳本只記錄 RAM，不包含 VRAM。

---

### 3. OWL API（Port 18001）

| 項目 | 數值 |
|------|------|
| PID | 3430415 |
| CPU% | 17–30%（測試中出現） |
| RAM% | ~0.4% |
| RSS | ~700 MB |

**對應**：`python -m uvicorn main:app --host 0.0.0.0 --port 18001`  
**來源**：`.env` 的 `OWL_API_BASE=http://127.0.0.1:18001`，推測為 OWL 物件偵測 API

---

### 4. Stream Simulator（FFmpeg）

| 項目 | 數值 |
|------|------|
| PID | 1335208 |
| CPU% | ~9.4% |
| RAM% | ~0.1% |
| RSS | ~160 MB |

**對應**：`ffmpeg -re -stream_loop -1 -i /videos/門禁遮臉入場/...`  
**來源**：`docker-compose` 的 `stream-simulator`，模擬 RTSP 推流

---

### 5. FFmpeg 片段切割（Backend 內）

| 項目 | 數值 |
|------|------|
| 出現時機 | 13:57:42 等 |
| 指令 | `ffmpeg -y -ss 0.0 -t 10.0 -i segment/RTSP_33_2/...` |
| 說明 | 偶發、短暫，用於切割影片片段 |

**來源**：`backend/src/services/` 或 `VideoService` 切割片段時呼叫 FFmpeg

---

## 三、RAM 使用量為何這樣？

### 1. 模型載入（每個 worker 一份）

| 模型 | 預估 RAM |
|------|----------|
| YOLO-World (yolov8s-world) | ~500 MB |
| ReID (ResNet50) | ~100 MB |
| SentenceTransformer (MiniLM) | ~500 MB |
| PyTorch / 其他 | ~1–2 GB |

單一 worker 約 **2–3 GB** 基礎模型載入。

### 2. 推理過程暫存

- 影片幀：`cv2.VideoCapture`、numpy 陣列
- YOLO 批次輸入：多幀同時在記憶體
- ReID embedding：crop 影像與特徵向量
- 請求/回應：JSON、base64 影像

負載高時，單一 worker 可額外使用 **5–15 GB**。

### 3. 多進程累積

- 4 workers × (2–3 GB 基礎 + 5–15 GB 暫存) ≈ **28–72 GB**
- 加上 Ollama、OWL、系統等：約 **60–70 GB** 符合實際

---

## 四、CPU 使用量為何這樣？

| 組件 | CPU 用途 |
|------|----------|
| Backend workers | YOLO 推論、ReID 特徵、影像前處理、JSON 序列化 |
| Ollama | VLM 推理（多為 GPU，CPU 負載較低） |
| OWL API | 物件偵測 |
| FFmpeg | 影片解碼、編碼、切割 |

33 路並行、7 個 API 同時上傳時，backend 會持續做推理與 I/O，整體 CPU 約 60–85%。

---

## 五、總結

| 項目 | 數值 |
|------|------|
| **RAM 峰值** | **67 GB** |
| **主要來源** | 4 個 backend worker（約 52–65 GB） |
| **次要來源** | Ollama 進程 CPU RAM（~2.4 GB）、OWL（~0.7 GB）、FFmpeg、系統。Ollama 模型在 VRAM，不計入此表 |
| **CPU 峰值** | 約 **75–85%** |

**結論**：RAM 與 CPU 主要由 backend 的 4 個 Uvicorn worker 佔用，每個 worker 載入 YOLO、ReID、SentenceTransformer 等模型，並在處理多路 RTSP 時累積暫存與結果，導致整體用量約 60–70 GB。
