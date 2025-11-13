# Multi-stage build for TMF-Matting
# Builder stage
FROM python:3.9-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ninja-build \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender-dev \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch
RUN pip install --no-cache-dir torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install mmediting
RUN git clone https://github.com/open-mmlab/mmediting.git /mmediting
WORKDIR /mmediting
RUN pip install --no-cache-dir mmcv-full==1.3.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.6.0/index.html && \
    pip install --no-cache-dir cython && \
    pip install --no-cache-dir -e .

# Strip unnecessary files from torch
RUN rm -rf /usr/local/lib/python3.*/site-packages/torch/test && \
    rm -rf /usr/local/lib/python3.*/site-packages/torch/include && \
    find /usr/local/lib/python3.*/site-packages/torch -name "*.so" -exec strip {} \;

# Runtime stage - slim base
FROM python:3.9-slim

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /mmediting /mmediting

WORKDIR /workspace
