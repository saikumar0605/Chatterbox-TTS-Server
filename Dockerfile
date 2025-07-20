FROM python:3.11-slim

# System setup
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 ffmpeg git \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install all standard requirements (no torch/torchvision/torchaudio in this file!)
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install CUDA-specific, version-pinned torch/vision/audio last (guaranteed compatibility)
RUN pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --extra-index-url https://download.pytorch.org/whl/cu121

# Confirm the CUDA/tv nms operator is present (important, will fail if mistakenly overwritten)
RUN python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, hasattr(torch.ops.torchvision, 'nms'))"

# Copy app code
COPY . .

# Create dir structure in the image
RUN mkdir -p model_cache reference_audio outputs voices logs hf_cache

EXPOSE 8004

# Production start
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8004", "--workers", "3"]
