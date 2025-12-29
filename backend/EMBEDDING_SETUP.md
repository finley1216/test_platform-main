# Embedding 模型設置說明

## 問題

容器內無法連網，無法從 Hugging Face 下載 `paraphrase-multilingual-MiniLM-L12-v2` 模型。

## 解決方案

### 方案 1：在主機上下載模型（推薦）

1. 在主機上下載模型：
```bash
# 在主機上執行
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"
```

2. 模型會下載到：`~/.cache/huggingface/hub/`

3. 將模型目錄掛載到容器：
在 `docker-compose.yml` 的 backend volumes 中添加：
```yaml
volumes:
  - ~/.cache/huggingface:/root/.cache/huggingface:ro
```

### 方案 2：使用 Docker 構建時下載

在 Dockerfile 中添加模型下載步驟（構建時有網路）：
```dockerfile
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"
```

### 方案 3：暫時跳過 embedding 生成

如果暫時不需要為現有記錄生成 embedding，可以：
- 新資料會自動生成 embedding（當模型可用時）
- 現有記錄可以稍後再生成

## 當前狀態

- ✅ PostgreSQL + pgvector 已設置
- ✅ 資料庫模型已添加 embedding 欄位
- ✅ 新資料插入時會自動生成 embedding
- ⚠️  現有記錄的 embedding 生成需要模型可用

## 臨時解決方案

如果急需生成 embedding，可以：
1. 在主機上下載模型
2. 通過 volume 掛載
3. 重新執行 `generate_embeddings.py`

