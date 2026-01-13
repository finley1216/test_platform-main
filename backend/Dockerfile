FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*


# Copy requirements and install Python dependencies
COPY ./src ./src
COPY ./requirements.txt ./

# Copy YOLO model if it exists (optional, will use download if not present)
COPY ./models ./models

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install YOLO-World CLIP module (required for YOLO-World)
RUN pip install --no-cache-dir git+https://github.com/ultralytics/CLIP.git || echo "Warning: Failed to install CLIP module, will retry at runtime"

# Pre-download CLIP model for YOLO-World (ViT-B/32) during build
RUN python3 -c "import clip; clip.load('ViT-B/32', device='cpu')" || echo "Warning: Failed to download CLIP model, will retry at runtime"

# Download embedding model during build (CPU mode only, when network is available)
RUN CUDA_VISIBLE_DEVICES="" python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')" || echo "Warning: Failed to download model, will retry at runtime"

# Pre-download ResNet50 models for ReID (用於物件重新識別)
# 嘗試下載 timm ResNet50（優先）
RUN python3 -c "import timm; timm.create_model('resnet50', pretrained=True)" || echo "Warning: Failed to download timm ResNet50, will use fallback at runtime"
# 備用：下載 torchvision ResNet50
RUN python3 -c "import torchvision.models as models; models.resnet50(pretrained=True)" || echo "Warning: Failed to download torchvision ResNet50, will use fallback at runtime"

# Set Python path
ENV PYTHONPATH=/app

# Set project root
ENV PROJECT_ROOT=/app

# Expose port
EXPOSE 8080

# Run the application using start.py
CMD ["python", "src/start.py"]

