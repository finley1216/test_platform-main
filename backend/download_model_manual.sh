#!/bin/bash
# 手動下載模型文件的腳本
# 使用 wget 直接下載，不依賴 Python 庫

MODEL_NAME="paraphrase-multilingual-MiniLM-L12-v2"
CACHE_DIR="/root/.cache/huggingface/hub/models--sentence-transformers--${MODEL_NAME}"
SNAPSHOT_DIR="${CACHE_DIR}/snapshots/main"

# 創建目錄
mkdir -p "${SNAPSHOT_DIR}"

# Hugging Face 模型文件 URL（需要替換為實際的 commit hash）
BASE_URL="https://huggingface.co/sentence-transformers/${MODEL_NAME}/resolve/main"

# 需要下載的文件列表
FILES=(
    "modules.json"
    "config.json"
    "sentence_bert_config.json"
    "tokenizer_config.json"
    "vocab.txt"
    "pytorch_model.bin"
    "1_Pooling/config.json"
)

echo "開始下載模型文件..."
cd "${SNAPSHOT_DIR}"

for file in "${FILES[@]}"; do
    echo "下載: ${file}"
    wget -q "${BASE_URL}/${file}" -O "${file}" || echo "警告: ${file} 下載失敗"
done

echo "下載完成！"

