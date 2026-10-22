# CPU-only Linux compiler. No GPU devices or NVIDIA container runtime needed.
FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c
RUN apt-get update && apt-get install -y --no-install-recommends g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY compiler-python.lock /opt/compiler-python.lock
RUN pip install --no-cache-dir -c /opt/compiler-python.lock torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -c /opt/compiler-python.lock tilelang==0.1.15 \
    nvidia-cuda-nvcc==13.0.88 nvidia-nvvm==13.0.88 \
    nvidia-cuda-runtime==13.0.88 nvidia-cuda-crt==13.0.88 \
    nvidia-cuda-cccl==13.0.85 nvidia-cuda-nvdisasm==13.4.92
ENV CUDA_HOME=/usr/local/lib/python3.12/site-packages/nvidia/cu13
ENV PATH=/usr/local/lib/python3.12/site-packages/nvidia/cu13/bin:$PATH
WORKDIR /work
