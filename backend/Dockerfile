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

# Set Python path
ENV PYTHONPATH=/app

# Set project root
ENV PROJECT_ROOT=/app

# Expose port
EXPOSE 8080

# Run the application using start.py
CMD ["python", "src/start.py"]

