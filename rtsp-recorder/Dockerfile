# Use a lightweight Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install essential tools for server binary installation
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    ca-certificates \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Copy files
COPY ./src ./src
COPY ./config ./config
COPY ./requirements.txt ./

# Install dependencies
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6
RUN pip install --no-cache-dir -r requirements.txt

# Add the current directory to PYTHONPATH
ENV PYTHONPATH=/app

# Set PROJECT_ROOT to match the working directory
ENV PROJECT_ROOT=/app

# Create a volume for video storage
VOLUME ["/app/video"]

# Run the script
CMD ["python", "src/main.py"]