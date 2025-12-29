# CPU 模式修改完成總結

## ✅ 已完成的修改

### Step 1: 修改 `backend/src/main.py`
- ✅ 在 `get_embedding_model()` 函數中添加 `os.environ['CUDA_VISIBLE_DEVICES'] = ''`
- ✅ 明確指定 `device='cpu'` 參數
- ✅ 更新 log 訊息為 "CPU Mode"

### Step 2: 修改 `backend/src/generate_embeddings.py`
- ✅ 在 `get_embedding_model()` 函數中添加 `os.environ['CUDA_VISIBLE_DEVICES'] = ''`
- ✅ 明確指定 `device='cpu'` 參數
- ✅ 更新 log 訊息為 "CPU Mode"

### Step 3: 修改 `backend/Dockerfile`
- ✅ 修改模型下載指令為：`CUDA_VISIBLE_DEVICES="" python3 -c "...device='cpu'..."`

### Step 4: 修改 `docker-compose.yml`
- ✅ 添加 volume 掛載：`~/.cache/huggingface:/root/.cache/huggingface:ro`

## ⚠️ 當前問題

**容器內無法連網（DNS 解析失敗）**，導致無法下載模型。

## 🔧 解決方案

### 方案 A: 在構建時下載模型（推薦）

由於構建時可能有網路，可以：

```bash
# 重建容器（不使用緩存，強制下載模型）
cd /mnt/10THDD/M133040024/SSD/ASE/test_platform-main
docker compose build --no-cache backend
docker compose up -d backend
```

### 方案 B: 在主機上下載模型

如果主機有網路，可以在主機上下載模型：

```bash
# 在主機上安裝 sentence-transformers（如果沒有）
pip install sentence-transformers

# 下載模型（CPU 模式）
CUDA_VISIBLE_DEVICES="" python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')"

# 模型會下載到 ~/.cache/huggingface/
# 然後重啟容器，volume 會自動掛載
docker compose restart backend
```

### 方案 C: 使用 Docker 構建時的網路

如果構建時有網路，模型會在構建時下載到容器內，無需額外操作。

## 📝 驗證步驟

模型下載完成後，執行：

```bash
docker exec -w /app test_platform-main-backend-1 python3 src/generate_embeddings.py
```

應該會看到：
- ✓ SentenceTransformer 模型載入: paraphrase-multilingual-MiniLM-L12-v2 (CPU Mode)
- 開始處理 410 筆記錄

## 🎯 預期結果

完成後：
- ✅ 所有 embedding 操作都使用 CPU 模式
- ✅ 不會觸碰 GPU 資源
- ✅ 410 筆記錄的 embedding 可以生成
- ✅ 新資料會自動生成 embedding（CPU 模式）

