# 不使用 Hugging Face Python 庫的下載方案

## 方案 1: 使用 Git LFS 克隆（最推薦）

在主機上執行：

```bash
# 安裝 Git LFS（如果沒有）
git lfs install

# 克隆模型倉庫
cd ~/.cache/huggingface/hub/
git clone https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# 模型會下載到：
# ~/.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/
```

然後重啟容器，volume 會自動掛載。

## 方案 2: 使用 wget/curl 直接下載文件

在主機上執行：

```bash
# 創建目錄
mkdir -p ~/.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots/main

# 下載必要文件（從 Hugging Face 頁面獲取實際 URL）
cd ~/.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/snapshots/main

# 需要下載的文件：
# - modules.json
# - config.json  
# - sentence_bert_config.json
# - tokenizer_config.json
# - vocab.txt
# - pytorch_model.bin (或 model.safetensors)
# - 1_Pooling/config.json

# 示例（需要替換為實際的 commit hash）：
wget https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/main/modules.json
wget https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/resolve/main/config.json
# ... 其他文件
```

## 方案 3: 從其他環境複製

如果有其他已經下載過模型的環境：
```bash
# 從其他機器/容器複製
scp -r user@other-host:~/.cache/huggingface/hub/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2 \
     ~/.cache/huggingface/hub/
```

## 方案 4: 使用 Python 在主機上下載

在主機上（有網路的地方）：
```bash
# 安裝 sentence-transformers
pip install sentence-transformers

# 下載模型（CPU 模式）
CUDA_VISIBLE_DEVICES="" python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')"

# 模型會自動下載到 ~/.cache/huggingface/hub/
```

## 方案 5: 使用 Docker 構建時下載

修改 Dockerfile，在構建時下載（構建時通常有網路）：
```dockerfile
RUN CUDA_VISIBLE_DEVICES="" python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')"
```

然後構建：
```bash
docker compose build --no-cache backend
```

## 推薦執行順序

1. **先試方案 5**（構建時下載）- 最簡單
2. **如果構建時也無法連網，用方案 1**（Git LFS）- 最可靠
3. **如果沒有 Git，用方案 4**（主機 Python 下載）- 最直接

