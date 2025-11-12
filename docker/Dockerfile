# Multi-stage build for TMF-Matting
# Builder stage with CUDA devel for compiling extensions
ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel as builder

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install mmediting
RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmediting.git /mmediting
WORKDIR /mmediting
ENV FORCE_CUDA="1"
RUN pip install mmcv-full==latest+torch1.6.0+cu101 -f https://download.openmmlab.com/mmcv/dist/index.html
RUN pip install cython --no-cache-dir
RUN pip install --no-cache-dir -e .

# Runtime stage - CUDA runtime only (no build tools)
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-runtime

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy conda environment from builder
COPY --from=builder /opt/conda /opt/conda

WORKDIR /workspace
