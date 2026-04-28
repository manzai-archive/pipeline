# syntax=docker/dockerfile:1.7
#
# manzai-archive pipeline — GPU-enabled container.
#
# Build:   docker compose build
# Run:     docker compose run --rm pipeline ingest <url> --group-slug nakagawake
# GPU:     requires nvidia-container-toolkit on host; compose passes --gpus all.

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

# System deps — Python 3.12, ffmpeg for transcoding, curl/git/ca-certs for fetching.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip \
        ffmpeg git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# yt-dlp — pinned to a known-good release. Bump in PR if YouTube changes break things.
ARG YT_DLP_VERSION=2026.03.17
RUN curl -fsSL "https://github.com/yt-dlp/yt-dlp/releases/download/${YT_DLP_VERSION}/yt-dlp" \
        -o /usr/local/bin/yt-dlp \
    && chmod +x /usr/local/bin/yt-dlp

# Isolated Python env (avoids PEP 668 system-managed restriction)
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
RUN pip install --upgrade pip wheel

# PyTorch with CUDA 12.4 wheels
RUN pip install --index-url https://download.pytorch.org/whl/cu124 \
        torch torchaudio

# Pipeline runtime deps. Install separately from the package so the layer
# caches across pipeline source edits.
RUN pip install \
        "faster-whisper>=1.1.0" \
        "pyannote.audio>=3.3.0,<5.0.0" \
        "ctranslate2>=4.4.0" \
        click pyyaml rich python-slugify python-dotenv hf_transfer

# Install the pipeline package itself (editable).
# At runtime compose bind-mounts the source over this path, so edits take
# effect without rebuilding the image.
WORKDIR /workspace/pipeline
COPY pyproject.toml ./
COPY pipeline ./pipeline
RUN pip install -e .

# Defaults assume NVIDIA GPU; override via env if needed.
ENV WHISPER_BACKEND=faster \
    WHISPER_DEVICE=cuda \
    WHISPER_COMPUTE_TYPE=float16 \
    WHISPER_MODEL=large-v3-turbo \
    PYANNOTE_DEVICE=cuda \
    MANZAI_CONTENT_DIR=/workspace/web/src/content/manzai

ENTRYPOINT ["python3", "-m", "pipeline"]
CMD ["--help"]
