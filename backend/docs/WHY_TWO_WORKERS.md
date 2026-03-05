# 為什麼 nvidia-smi 看到兩個 Python？（兩個 worker 來源說明）

## 一、WORKERS 是從哪裡設定的？（依優先順序）

1. **Docker Compose**（`docker-compose.yml`）  
   - backend 服務的 `environment` 裡有：`- WORKERS=1`  
   - 容器啟動時會把這個值傳給後端 process。

2. **環境變數 .env**  
   - 專案目錄下的 `.env` 有 `WORKERS=1`。  
   - 若 compose 用 `${WORKERS:-1}` 這種寫法，會讀 .env；但我們現在是**寫死** `WORKERS=1`，所以**不會**從 .env 覆寫。

3. **config.py**  
   - `self.WORKERS = int(os.getenv("WORKERS", "1"))`  
   - 讀的是**當前 process 的環境變數**（也就是 compose 傳進容器的那個）。

4. **start.py**  
   - `uvicorn.run(..., workers=config.WORKERS if not config.RELOAD else 1)`  
   - 真正決定 Uvicorn 開幾個 worker 的只有這裡。

所以：**理論上**只要 compose 裡是 `WORKERS=1`，後端就只會開 1 個 worker。

---

## 二、為什麼還會看到「兩個」Python process？

有兩種可能：

### 情況 A：Uvicorn 的 1 master + 1 worker（正常）

- 當 `workers=1` 時，Uvicorn 會產生：
  - **1 個 master process**：跑 `start.py`、負責 spawn worker、不處理請求。
  - **1 個 worker process**：載入 FastAPI app、載入 Qwen/YOLO/ReID、處理請求。
- 所以 **nvidia-smi 本來就可能看到 2 個 python**，這是 Uvicorn 的設計。
- 差別在於：
  - **正常**：只有 **1 個** process 佔大量 VRAM（約 15–17GB），那是 worker；另一個幾乎不佔或只佔一點（master 通常不載模型）。
  - **異常**：**兩個** process 都佔十幾 GB（例如 13GB + 17GB）→ 代表有 **2 個 worker**，也就是當時 `WORKERS=2`（或曾經是 2，容器沒重建）。

### 情況 B：容器還是舊的 WORKERS（最常見）

- 若你**改過** compose 或 .env 的 WORKERS，但**沒有重建** backend 容器，舊容器可能還是用之前的環境（例如 WORKERS=2）。
- Docker Compose 只有在「重建」容器時才會把新的 `environment` 寫進去。

---

## 三、怎麼確認「兩個 worker」到底從哪來？

依序做這幾步：

### 1. 強制重建 backend 容器

```bash
cd /mnt/10THDD/M133040024/SSD/ASE/test_platform-main
docker compose up -d --force-recreate backend
```

### 2. 看啟動日誌（確認 workers 數與 PID）

```bash
docker logs test_platform-main-backend-1 2>&1 | head -30
```

預期會看到類似：

- `[start] Uvicorn workers=1 (env WORKERS=1)`
- `[start] 本 process PID=xxxxx（此為 master；worker 會是別的 PID...）`

若這裡是 `workers=2` 或 `WORKERS=2`，就代表**環境還是 2**，要回頭看 compose / 是否有 override。

### 3. 進容器看環境變數

```bash
docker exec test_platform-main-backend-1 env | grep WORKERS
```

應該要出現：`WORKERS=1`。若是 `WORKERS=2`，就是 compose 或某處還傳了 2。

### 4. 看容器內有幾個 Python process

```bash
docker exec test_platform-main-backend-1 ps aux | grep python
```

- 若 **workers=1**：通常會看到 2 個 python（1 master + 1 worker）。
- 若 **workers=2**：會看到 3 個 python（1 master + 2 workers）。

### 5. 對照 nvidia-smi 的 VRAM

- **只有 1 個** python 佔約 15–17GB → 代表只有 1 個 worker 載模型，這是正確的（另一個是 master，不載模型或只佔很少）。
- **兩個** python 都佔約 13–17GB → 代表有 2 個 worker 在載模型，即當時 **WORKERS=2**，要從上面步驟查環境/容器是否沒更新。

---

## 四、總結：兩個 worker 的來源

| 可能來源 | 說明 | 怎麼處理 |
|--------|------|----------|
| **docker-compose 沒生效** | 改過 `WORKERS=1` 但沒重建容器，舊容器仍用舊值（例如 2） | `docker compose up -d --force-recreate backend` |
| **Uvicorn 1 master + 1 worker** | workers=1 時本來就有 2 個 python process | 正常；只要只有 1 個 process 佔大 VRAM 即可 |
| **真的有 2 個 worker** | 環境裡 WORKERS=2（compose / .env / override） | 用上面「三」的步驟確認 env 與 logs，把 WORKERS 改回 1 並重建 |

目前程式與 compose 都已設成 **WORKERS=1**，若你仍看到兩個 python 都佔大量 VRAM，**先做一次** `docker compose up -d --force-recreate backend`，再依「三」的步驟對照日誌與 `env | grep WORKERS`，就能確定兩個 worker 是從哪裡來的。
