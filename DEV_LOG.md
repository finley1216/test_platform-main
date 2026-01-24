# 開發日誌 (DEV_LOG)

## Git 操作流程

### 基本提交流程
```bash
git status              # 確認今天改了什麼
git add .               # 全部準備打包
git commit -m "訊息"    # 補充今天大概做了什麼
git push                # 全部上傳
```

### 修改上一個 commit
```bash
git commit --amend --no-edit    # 把變更併入「上一個 commit」
git push --force-with-lease     # 強制上傳上去
```

---

## 網頁執行流程

### 啟動步驟

**1. 啟動 63 的 Docker**
```bash
docker compose up -d
```

**2. 在 88 的 PowerShell 執行 SSH 端口轉發**
```powershell
ssh -g -N ^
  -L 0.0.0.0:3000:localhost:3000 ^
  -L 0.0.0.0:8080:localhost:8080 ^
  -L 0.0.0.0:11434:localhost:11434 ^
  -L 0.0.0.0:18001:localhost:18001 ^
  M133040024@140.117.176.63
```

**3. 訪問前端**
```
http://140.117.176.88:3000
```

### 已部署服務端口
- **3000** - 前端服務
- **8080** - 後端 API
- **11434** - Ollama
- **18001** - OWL API
- **5432** - PostgreSQL 資料庫
- **5050** - pgAdmin 圖形化管理工具

---

## PostgreSQL 資料庫操作流程

### 啟動 PostgreSQL 和 pgAdmin

**1. 啟動服務**
```bash
cd /mnt/10THDD/M133040024/SSD/ASE/test_platform-main
docker compose up -d postgres pgadmin
```

**2. 等待服務就緒（約 10-30 秒）**
```bash
docker compose ps postgres pgadmin
```

**3. 初始化資料庫表結構（僅首次需要）**
```bash
docker exec test_platform-main-backend-1 python3 /app/src/init_db.py
```
⚠️ **注意**：執行完後記得刪除 `backend/src/init_db.py`（一次性腳本）

**4. 遷移歷史資料（僅用於舊的 JSON 檔案）**
```bash
# ⚠️ 注意：此腳本只用於遷移歷史資料（已經存在的 JSON 檔案）
# 新分析的影片會自動保存到 PostgreSQL，不需要手動遷移
docker exec test_platform-main-backend-1 python3 /app/src/migrate_segments_to_db.py
```

### 使用 pgAdmin 查看資料

**1. 訪問 pgAdmin**
- 網址：`http://localhost:5050`
- 帳號：`admin@admin.com`
- 密碼：`admin`

**2. 註冊 PostgreSQL 伺服器**
- 右鍵點擊左側 "Servers" → "Register" → "Server"
- **General 標籤**：
  - Name: `PostgreSQL Server`（任意名稱）
- **Connection 標籤**：
  - Host name/address: `postgres`（重要：使用容器名稱）
  - Port: `5432`
  - Maintenance database: `postgres`
  - Username: `postgres`
  - Password: `postgres`
  - 勾選 "Save password"
- 點擊 "Save"

**3. 查看資料**
- 展開左側樹狀結構：
  ```
  Servers → PostgreSQL Server → Databases → postgres → Schemas → public → Tables → summaries
  ```
- 右鍵點擊 `summaries` 表 → "View/Edit Data" → "All Rows"
- 調整 "Limit" 下拉選單為 "No limit" 以查看所有資料

**4. 使用 SQL 查詢**
- 在 Query Tool 中執行 SQL：
  ```sql
  -- 查看所有資料
  SELECT * FROM summaries;
  
  -- 查看總筆數
  SELECT COUNT(*) FROM summaries;
  
  -- 查看各影片的記錄數
  SELECT video, COUNT(*) as count 
  FROM summaries 
  GROUP BY video 
  ORDER BY count DESC;
  ```

### 資料庫資訊

- **資料庫名稱**: `postgres`
- **使用者**: `postgres`
- **密碼**: `postgres`
- **主表**: `summaries`（影片片段摘要資料表）
- **目前資料量**: 409 筆（會隨分析結果增加）

### 資料來源

1. **即時分析（自動保存）**：
   - 當影片在前端執行分析時，分析完成後會**自動保存到 PostgreSQL**
   - 不需要手動遷移，系統會自動調用 `_save_results_to_postgres()` 函數
   - 邏輯：如果記錄已存在（相同 video、segment、time_range）則更新，否則新增

2. **歷史資料遷移（一次性）**：
   - `migrate_segments_to_db.py` 只用於遷移**歷史資料**（已經存在的 JSON 檔案）
   - 例如：系統升級前已經分析好的 JSON 檔案，需要匯入到新的資料庫
   - 新分析的影片不需要使用此腳本

---

## 開發日誌

### 2025-12-31

#### Backend
- **API 模組化重構**
  - 創建 `backend/src/api/` 資料夾結構，將所有 API 端點按功能分類管理
  - 創建 `api/health.py`：健康檢查和認證相關 API（`/health`, `/auth/verify`）
  - 創建 `api/prompts.py`：Prompt 管理 API（`/prompts/defaults`）
  - 創建 `api/video_analysis.py`：影片分析相關 API
    - `/v1/analyze_segment_result`：單一片段分析
    - `/v1/segment_video`：影片切割（嚴格模式）
    - `/v1/segment_pipeline_multipart`：完整分析流程
  - 創建 `api/rag.py`：RAG 相關 API
    - `/rag/index`：索引分析結果
    - `/rag/search`：混合搜索
    - `/rag/answer`：LLM 回答
    - `/rag/stats`：統計資訊
  - 創建 `api/video_management.py`：影片管理 API
    - `/v1/videos/list`：影片列表
    - `/v1/videos/{video_id}`：影片詳情
    - `/v1/videos/{video_id}/event`：事件標籤管理
    - `/v1/videos/categories`：分類列表
    - `/v1/videos/{video_id}/move`：移動影片到分類
  - 修改 `main.py`：移除所有端點定義，保留輔助函數，在文件末尾註冊路由模組
  - 解決循環導入問題：將路由註冊移到所有函數定義之後

- **修復 RAG 索引錯誤**
  - 修改 `_auto_index_to_rag` 函數：添加 `HAS_RAG_STORE` 和 `RAGStore` 可用性檢查
  - 當 RAGStore 不可用時（faiss-cpu 未安裝），返回明確的錯誤訊息，避免 `'NoneType' object has no attribute 'read_index'` 錯誤

- **修復影片詳情 API 404 錯誤**
  - 修改 `api/video_management.py` 的 `get_video_info` 函數：
    - 優先檢查 `segment` 中是否有對應的處理結果（格式：`{category}_{video_name}`）
    - 如果 segment 中有結果，優先使用 segment 的結果
    - 如果 segment 中沒有，再檢查 `video_lib`
    - 兩者都沒有才返回 404
  - 解決前端使用 `fire/1` 格式訪問時找不到影片的問題

- **清理不需要的遷移腳本**
  - 刪除 `add_embedding_column.py`：一次性資料庫遷移腳本
  - 刪除 `generate_embeddings.py`：一次性工具腳本
  - 保留 `add_video_column.py` 和 `migrate_segments_to_db.py`（可能仍需要）

#### Frontend
- **修復無限循環請求問題**
  - 修改 `frontend/src/components/AnalysisResults.js`：
    - 使用 `useMemo` 記憶化 `videoIdInfo`，避免每次渲染都創建新對象
    - 使用 `useRef` 追蹤已載入的請求，避免重複請求相同數據
    - 解決 `useEffect` 依賴項導致無限循環的問題

- **完全移除 FAISS 依賴，全面遷移到 PostgreSQL + pgvector**
  - 修改 `/rag/index` API：改為使用 PostgreSQL 保存數據（通過 `_save_results_to_postgres`）
  - 修改 `/rag/stats` API：改為從 PostgreSQL 查詢統計信息（查詢有 embedding 的記錄數量）
  - 修改 `_auto_index_to_rag` 函數：移除 RAGStore 使用，改為只返回成功狀態（數據已通過 `_save_results_to_postgres` 自動保存）
  - 移除 `main.py` 中的 RAGStore 導入：標記為已棄用，設置 `HAS_RAG_STORE = False`
  - 移除 `_remove_old_rag_records` 函數的實際功能：PostgreSQL 使用更新或新增邏輯，不需要手動刪除
  - 移除 `_rag_index_legacy` 和 `_rag_stats_legacy` 函數中的 RAGStore 使用：標記為已棄用
  - 刪除 `backend/src/rag_store.py` 文件：不再需要 FAISS 相關代碼
  - 修復循環導入問題：將 `router` 定義移到導入之前，`_save_results_to_postgres` 在函數中使用時再導入
  - 所有 RAG 功能現在完全使用 PostgreSQL + pgvector：
    - `/rag/search`：使用 PostgreSQL + pgvector 進行混合搜索
    - `/rag/answer`：使用 PostgreSQL + pgvector 進行搜索後 LLM 回答
    - `/rag/index`：使用 PostgreSQL 保存數據（包含 embedding）
    - `/rag/stats`：從 PostgreSQL 查詢統計信息

#### 備註
- API 模組化完成，所有端點已按功能分類到獨立模組
- 後端服務正常運行，所有 API 端點正常工作
- 前端無限循環請求問題已修復
- 影片詳情 API 現在可以正確處理 `category/video_name` 格式的請求
- **完全移除 FAISS 依賴**：所有 RAG 功能現在完全使用 PostgreSQL + pgvector，不再需要 `faiss-cpu` 套件
- `backend/rag_store/` 資料夾（包含 `meta.jsonl`、`dim.txt`、`index.faiss`）可以刪除，不再使用

### 2025-12-30

#### Backend
- **生成 RAG 查詢流程完整說明文件**
  - 創建 `backend/RAG查詢流程說明.md`：完整的 RAG 查詢流程技術文檔
  - 說明整體架構：PostgreSQL + pgvector 混合搜尋機制
  - 詳細說明時間和關鍵字提取方法：
    - 日期解析：使用正則表達式和負向斷言精準定位數字
    - 關鍵字提取：使用字符串包含檢查和白名單匹配
    - 事件類型映射：中文關鍵字 → 資料庫欄位名稱
  - 說明分數計算機制：cosine_distance 轉換為相似度分數的公式
  - 說明資料庫混合查詢：硬篩選 + 向量搜尋的單一 SQL 查詢實現
  - 說明白名單格式：事件類型映射、關鍵字列表、日期解析關鍵字
  - 提供完整的函式流程圖和 MCP 流程說明

- **MCP 流程說明和優化**
  - 在文檔中詳細說明 MCP（Model Context Protocol）的實際使用方式
  - 說明 MCP 與 LLM 的關係：MCP 用於工具封裝，與 AI 模型無直接關係
  - 說明 MCP 在 `/rag/search` 和 `/rag/answer` 中的使用方式
  - 提供讓 LLM 直接調用 MCP 工具的實現方案（未來擴展）

#### Frontend
- **修復影片下載路徑問題**
  - 修改 `frontend/src/components/RagResults.js`：
    - 修正 `fullVideoPath` 構建邏輯，確保包含 `/segment/` 前綴
    - 處理 `videoPath` 已包含 `/segment/` 的情況，避免重複添加
  - 修改 `frontend/src/services/api.js`：
    - 修正 `downloadFile` 函數的 URL 構建邏輯
    - 確保路徑正確處理，避免移除必要的 `/` 前綴
    - 添加註釋說明 HTTP 頁面上使用 blob URL 的安全警告（預期行為，不影響功能）

#### 備註
- 已部署到 140.117.176.88
- 所有端口已開放並可正常訪問
- 影片下載功能已修復，可正常下載和查看內容
- RAG 查詢流程說明文件已生成，包含完整的技術文檔和流程圖

### 2025-12-29

#### Backend
- **資料庫架構重構：從 FAISS 遷移到 PostgreSQL + pgvector**
  - 修改 `backend/src/models.py`：在 `Summary` 表中添加 `embedding` 欄位（Vector(384)）
  - 修改 `backend/src/database.py`：添加 `ensure_pgvector_extension()` 函數，自動啟用 pgvector 擴展
  - 更新 `docker-compose.yml`：PostgreSQL 服務改用 `ankane/pgvector:v0.5.0-pg15` 映像
  - 執行資料庫遷移：為現有的 `summaries` 表添加 `embedding` 欄位

- **搜索邏輯重寫：實現單一查詢混合搜尋（Single-query Hybrid Search）**
  - 修改 `backend/src/main.py`：
    - 移除 `_merge_and_rank_results` 和 `_calculate_sql_score` 函數（不再需要複雜的合併邏輯）
    - 移除對 `RAGStore` (FAISS) 的所有呼叫
    - 重寫 `rag_search` 和 `rag_answer`：使用 SQLAlchemy + pgvector 進行單一查詢混合搜尋
    - 實現時間範圍硬篩選（`start_timestamp.between()`）+ 向量相似度排序（`embedding.cosine_distance()`）
    - 使用 `SentenceTransformer` 模型（paraphrase-multilingual-MiniLM-L12-v2）生成查詢向量

- **查詢功能增強：日期解析和關鍵字提取**
  - 修改 `_parse_query_filters` 函數：
    - 實現日期解析（支援相對日期、MMDD 格式、絕對日期）
    - 實現關鍵字提取（從查詢中提取關鍵詞）
    - 實現事件類型檢測（從查詢中識別事件類型）
    - 生成「clean query」（去除日期相關詞彙後的查詢文本）用於 embedding 生成
  - 回應格式更新：添加 `date_parsed`、`keywords_found`、`event_types_found`、`embedding_query` 欄位

- **自動 Embedding 生成**
  - 修改 `_save_results_to_postgres` 函數：新資料或更新資料時自動生成並保存 embedding
  - 創建 `backend/src/generate_embeddings.py`：為現有資料補生成 embedding（處理 410 筆缺少 embedding 的記錄）
  - 實現 GPU/CPU 自動檢測和回退機制：
    - 自動檢測 GPU 可用性
    - GPU 記憶體不足時自動回退到 CPU 模式
    - 添加詳細的設備使用日誌

- **依賴更新**
  - 更新 `backend/requirements.txt`：
    - 添加 `pgvector==0.3.0`、`psycopg2-binary==2.9.9`
    - 標記 `faiss-cpu==1.7.4` 為 DEPRECATED（已註解）
    - 更新 `sentence-transformers>=2.3.0,<3.0.0` 和 `huggingface-hub>=0.20.0,<0.37.0` 以確保兼容性
  - 更新 `backend/Dockerfile`：在構建時預下載 embedding 模型，避免運行時網路問題

- **標記舊代碼為 DEPRECATED**
  - 修改 `backend/src/rag_store.py`：標記為 DEPRECATED，保留向後兼容性（允許模組載入但不使用）

#### Frontend
- **修復前端後端連接問題：實現動態 API base URL 檢測**
  - 修改 `frontend/src/utils/constants.js`：根據訪問的 hostname 自動選擇對應的後端 API
    - `localhost:3000` → `localhost:8080`
    - `140.117.176.88:3000` → `140.117.176.88:8080`
  - 修改 `frontend/src/services/api.js`：使用動態獲取的 API base URL，而非靜態值
  - 修改 `frontend/Dockerfile`：移除 `REACT_APP_API_BASE` 環境變數，改為運行時動態檢測
  - 添加詳細的調試日誌，方便排查 API 連接問題

- **顯示查詢解析資訊**
  - 修改 `frontend/src/services/api.js`：
    - 在 `searchRAG` 和 `answerRAG` 函數中添加控制台輸出
    - 顯示日期解析結果（模式、解析到的日期、時間範圍）
    - 顯示提取的關鍵字和事件類型
    - 顯示用於向量搜索的 clean query（embedding_query）
  - 使用彩色控制台輸出，方便開發者查看查詢解析結果

#### 備註
- 已部署到 140.117.176.88
- 所有端口已開放並可正常訪問
- 前端現在會根據訪問的 hostname 自動選擇對應的後端 API，無需手動配置

### 2025-12-25

#### Backend
- 添加 PostgreSQL 資料庫服務（port 5432）和 pgAdmin 圖形化管理工具（port 5050）
- 創建資料庫表結構（summaries 表），包含影片片段摘要和事件檢測欄位
- 遷移歷史資料：將 409 筆 JSON 檔案資料匯入 PostgreSQL
- 確認新分析的影片會自動保存到資料庫（無需手動遷移）
- 更新文檔：添加資料庫操作流程和結構說明

#### Frontend
- (待補充)

#### 備註
- 已部署到 140.117.176.88
- 所有端口已開放並可正常訪問
- PostgreSQL 資料庫已啟用，可透過 pgAdmin 查看和管理資料

### 2025-12-19

#### Backend
- (待補充)

#### Frontend
- (待補充)

#### 備註
- 已部署到 140.117.176.88
- 所有端口已開放並可正常訪問





恢復 VLM 功能
當 GPU 有空閒時，要恢復 VLM 功能，只需要：
打開檔案：test_platform-main/backend/src/api/video_analysis.py
找到第 483 行：SKIP_VLM = True
改為：SKIP_VLM = False
重啟服務
或者告訴我，我可以幫您恢復。









ffmpeg -re -stream_loop -1 -i "Video_衛哨端出入口2.avi" -c:v libx264 -preset ultrafast -tune zerolatency -b:v 2000k -maxrate 2000k -bufsize 4000k -g 60 -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/live

./mediamtx


docker compose restart backend && docker compose restart stream-simulator