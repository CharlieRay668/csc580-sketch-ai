# Dockerfile for Quick, Draw! Classification Training
# MLP, ResNet-18, and ViT on NRP Nautilus

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for Quick, Draw! training
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    tqdm \
    einops \
    matplotlib \
    numpy \
    pillow \
    scipy

# Copy source code
COPY src/ /workspace/src/
COPY kubernetes/train_all_models.py /workspace/

# Create directories for data and results
RUN mkdir -p /workspace/data /workspace/results /workspace/runs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace

# Default command - runs the training script
CMD ["python", "train_all_models.py"]
