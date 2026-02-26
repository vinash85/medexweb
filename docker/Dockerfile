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
    git \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

RUN apt update && apt install -y nano

# Set working directory
WORKDIR /app

# Install Python dependencies
# We install llama-cpp-python with CUDA support
RUN pip3 install --upgrade pip
#RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip3 install llama-cpp-python
RUN CMAKE_ARGS="-DGGML_CUDA=on" pip3 install llama-cpp-python
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
