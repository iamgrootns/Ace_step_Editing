# ✅ CUDA 12.4.1 with cuDNN
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies and add deadsnakes PPA
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update

# Install Python 3.12 (no distutils needed)
RUN apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Create symlinks
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Upgrade pip
RUN python3.12 -m pip install --upgrade pip setuptools wheel

#Install PyTorch 2.6.0 (latest stable for CUDA 12.4)
RUN pip install --no-cache-dir \
    torch==2.6.0+cu124 \
    torchvision==0.21.0+cu124 \
    torchaudio==2.6.0+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# Install dependencies
RUN pip install --no-cache-dir \
    runpod \
    requests \
    huggingface_hub \
    transformers \
    diffusers \
    accelerate \
    soundfile \
    librosa \
    scipy \
    numpy \
    einops \
    omegaconf \
    safetensors \
    sentencepiece \
    protobuf \
    datasets \
    peft \
    gradio

# Copy handler
COPY handler.py /app/handler.py

CMD ["python", "handler.py"]
