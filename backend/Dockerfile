FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


# Copy requirements and install Python dependencies
COPY ./src ./src
COPY ./requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download embedding model during build (CPU mode only, when network is available)
RUN CUDA_VISIBLE_DEVICES="" python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cpu')" || echo "Warning: Failed to download model, will retry at runtime"

# Set Python path
ENV PYTHONPATH=/app

# Set project root
ENV PROJECT_ROOT=/app

# Expose port
EXPOSE 8080

# Run the application using start.py
CMD ["python", "src/start.py"]

