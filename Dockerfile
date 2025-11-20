# Use a verified Runpod PyTorch image with Python 3.11 and CUDA 12.1 (compatible with ACE-Step)
FROM runpod/pytorch:2.4.0-py3.11-cuda12.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies (git, pkg-config, ffmpeg, build tools, libs for audio processing)
RUN apt-get update && apt-get install -y \
    git \
    pkg-config \
    ffmpeg \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libavfilter-dev \
    libswscale-dev \
    libswresample-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
 
# Copy project files (including your FastAPI code and requirements.txt)
COPY . /app

# Upgrade pip to latest version
RUN pip install --upgrade pip

# Install Python dependencies from your requirements.txt (to be provided by you)
RUN pip install --no-cache-dir -r requirements.txt

# (Optional) If a known compatible PyTorch rebuild is needed, uncomment and adjust below
# RUN pip install --no-cache-dir --force-reinstall torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Expose default FastAPI port
EXPOSE 8000

# Command to run your ACE-Step FastAPI app (replace app.py with your script filename)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
