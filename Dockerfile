# SoulX-FlashHead Lite — RunPod Serverless image
# Base: RunPod official PyTorch 2.7.1 + CUDA 12.8 + Python 3.11
FROM runpod/pytorch:1.0.3-cu1281-torch271-ubuntu2204

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/hf_cache \
    SOULX_MODELS_DIR=/runpod-volume/models \
    SOULX_PRELOAD_MODEL=1 \
    SOULX_PRELOAD_MODEL_TYPE=lite

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git wget ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY source/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --ignore-installed blinker -r /app/requirements.txt && \
    pip install --no-cache-dir runpod huggingface_hub

# SoulX 코드 (모델 weights 제외 — Network Volume에 둠)
COPY source/ /app/

# handler
COPY handler.py /app/handler.py

# 모델은 첫 cold start에서 lazy download (HF_HOME=/runpod-volume/hf_cache)
# Volume mount 안 됐을 때 fallback dir
RUN mkdir -p /runpod-volume/models /runpod-volume/hf_cache

CMD ["python", "-u", "handler.py"]
