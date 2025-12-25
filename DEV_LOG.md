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