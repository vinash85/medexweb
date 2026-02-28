# Use NVIDIA CUDA base for GPU acceleration
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_DOCKER_ARCH=all \
    LLAMA_CUBLAS=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    build-essential \
    cmake \
    git \
    nano \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
# We install llama-cpp-python with CUDA support
RUN pip3 install --upgrade pip
# Symlink CUDA stub so linker finds libcuda.so.1 during build (real driver loads at runtime)
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
# 0.3.16 — supports gemma3 (MedGemma) + multimodal vision (PaliGemma)
RUN LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}" \
    CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 \
    pip3 install llama-cpp-python==0.3.16 --no-cache-dir
RUN pip3 install mcp pillow flask pandas

# Copy your server and client code into the container
#COPY ../code/server.py .
#COPY ../code/app.py .
#COPY ../code/templates/ ./templates/

# Create mount points for models and data
#RUN mkdir /models 
#RUN mkdir /data

#COPY /mnt/sda/kushal/medgemma/models/ ./models/
#COPY /mnt/sda/kushal/medgemma/data/images/ ./data/
# Expose the Flask port
EXPOSE 6565

# Start the application
# Note: In a production MCP setup, the server is usually started via stdio,
# but for this synchronized web-app, we launch the Flask entry point.
