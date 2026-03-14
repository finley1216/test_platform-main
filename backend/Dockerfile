FROM docker.io/library/python:3.11-slim

# 1. 設置 apt 參數：忽略過期檢查、重試次數增加
RUN echo 'Acquire::Check-Valid-Until "false";' > /etc/apt/apt.conf.d/99no-check-valid-until && \
    echo 'Acquire::Retries "5";' > /etc/apt/apt.conf.d/99retries

# 2. 確保使用 HTTP 源 (相容性較好)
RUN sed -i 's/https:/http:/g' /etc/apt/sources.list.d/debian.sources

WORKDIR /app

# 3. 安裝套件
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. 先安裝 Python 依賴，利用 Docker Layer Cache
COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -U transformers accelerate && \
    pip install --no-cache-dir torchao && \
    pip install --no-cache-dir git+https://github.com/ultralytics/CLIP.git || echo "Warning: Failed to install CLIP module"

# 可選：精準 ReID 需 torchreid（未安裝時會 fallback 到 torchvision ResNet50）
RUN pip install --no-cache-dir torchreid || echo "Warning: torchreid not installed, ReID will use torchvision ResNet50"

# 3. 預先下載模型 (加速啟動，但不強制失敗)
RUN python3 -c "import clip; clip.load('ViT-B/32', device='cpu')" || echo "Warning: CLIP download failed" && \
    CUDA_VISIBLE_DEVICES="" python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')" || echo "Warning: SentenceTransformer download failed" && \
    python3 -c "import timm; timm.create_model('resnet50', pretrained=True)" || echo "Warning: timm resnet50 download failed" && \
    python3 -c "import torchvision.models as models; models.resnet50(pretrained=True)" || echo "Warning: torchvision resnet50 download failed"

# 4. 最後才 Copy 程式碼與模型權重 (變動最頻繁)
COPY ./models ./models
COPY ./src ./src

# Set Python path
ENV PYTHONPATH=/app
ENV PROJECT_ROOT=/app
# 優化顯存：減少 fragmentation，有助於 OOM 時釋放與分配
ENV PYTORCH_ALLOC_CONF=expandable_segments:True

# Expose port
EXPOSE 8080

# Run the application using start.py
CMD ["python", "src/start.py"]
