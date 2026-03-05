# RTSP 錄影 + API 並行處理流程

將 **rtsp-recorder** 產出的 10 秒短片，改為呼叫 **test_platform 後端 API** 並行處理（不再在本機跑 VLM/YOLO/ReID）。

## 流程概覽

1. **錄影**：`src/main.py` 或 recorder 依 `config/recorder-config.yaml` 寫入 `video/<路名>/`，每段 **10 秒**（`segment_duration: 10`）。
2. **Prompt**：從 `../backend/prompts/frame_prompt.md`、`summary_prompt.md` 讀取，送給 API。
3. **解析度與抽幀**：API 參數 `target_short=432`（對應 768×432）、`frames_per_segment=5`（每段 5 幀送 VLM）。
4. **並行送 API**：`run_api_pipeline.py` 掃描 `video/`，將未處理的 .mp4/.avi 並行 POST 到 `POST /v1/segment_pipeline_multipart`。

## 使用方式

在 **test_platform-main** 專案根目錄執行：

```bash
# 依賴（若尚未安裝）
pip install requests python-dotenv

# 只處理一批後結束
python rtsp-recorder/run_api_pipeline.py --once

# 持續監看（每 30 秒掃一次），並行 8 個
python rtsp-recorder/run_api_pipeline.py --workers 8

# 指定後端與間隔
BACKEND_URL=http://localhost:8080/api API_KEY=your_key python rtsp-recorder/run_api_pipeline.py --interval 20 --workers 4
```

## 環境變數

| 變數 | 說明 |
|------|------|
| `BACKEND_URL` | 後端 API base（預設 `http://140.117.176.88:3000/api`） |
| `API_KEY` 或 `MY_API_KEY` | 後端 X-API-Key |
| `VIDEO_LIB_DIR` | 錄影目錄（預設自動找 `video` 或 `rtsp-recorder/video`） |
| `SEGMENT_DURATION` | 每段秒數（預設 10） |

## 已處理紀錄

已送過 API 的片段會記錄在 `video/_api_pipeline_processed.json`，避免重複上傳。刪除該檔可重新送全部。

## 錄影設定

- `config/recorder-config.yaml` 中 `recording.segment_duration: 10` 已設為 10 秒，與 API 對接。
- **串流數量**：`config/stream-config.yaml` 裡有幾路 `enabled: true`，recorder 就錄幾路；**不必為了 pipeline 改成固定 10 路**。1 路、10 路、任意路數都可以，`run_api_pipeline.py` 會掃描 `video/` 下所有產出的檔並送 API。
- 目前 recorder 寫入 **扁平目錄**（`video/Test_Site-Ch1-2025-02-22_12-00-00.avi`）；pipeline 已支援此結構，並會從檔名推回 `video_stem`（例：`Test_Site-Ch1`）供後端識別。
- 實際錄影解析度由 RTSP 與 OpenCV 決定；**768×432 為 API 端 VLM 輸入短邊**（`target_short=432`）。
