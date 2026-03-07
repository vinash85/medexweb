# DermGemma — Project Guide

## Overview
GPU-accelerated skin lesion classification using MedGemma 4B and other VLMs. Three-phase pipeline (specialist observation → comparison review → classification) for dermoscopic images. Runs inside Docker with CUDA.

## Runtime Environment
- **Docker**: all code runs at `/home/project/` inside the container
- **Host mount**: `/mnt/data/medgemma` → `/home/project/`
- **GPU**: NVIDIA CUDA 12.2, all models use `n_gpu_layers=-1`
- **Framework**: llama-cpp-python 0.3.16

## Project Layout
```
code/
  pipeline.py             # Core 3-phase pipeline + MedGemmaChatHandler
  server.py               # Flask API (port 6565)
  index.html              # Web UI
  DermsGemms.csv          # Ground truth: 99 ISIC images (Image, Dx, Lesion attributes)
  test_phase1.py           # Phase 1 specialist query validation
  test_phase2.py           # Phase 2 comparison review validation
  test_phase3.py           # Phase 3 end-to-end classification
  test_vlm_benchmark.py   # Multi-VLM benchmark (5 models × 4 attributes × 20 images)
docker/
  Dockerfile, docker-compose.yml
models/                    # GGUF files (not committed)
  medgemma.gguf + medgemma-mmproj.gguf
  paligemma-3b-mix-224-gguf/
  Llama-3.2-11B-Vision-Instruct-GGUF/
  moondream2-20250414-GGUF/
  InternVL2_5-1B-GGUF/
data/
  images/                  # ISIC dermoscopic JPGs
  *.csv                    # Test/benchmark output results
```

## Key Patterns

### MedGemmaChatHandler
Custom subclass of `Llava15ChatHandler` in `pipeline.py`. Uses Gemma 3 turn template with `<start_of_turn>user/model` format. Forces `Answer:` prefix to bypass thinking mode.

### Image Preparation
All VLM inputs: resize to 224x224, enhance contrast 1.4x, save as JPEG quality 90, base64-encode as `data:image/jpeg;base64,...` URI.

### KV Cache Reset
Call `llm.reset()` between queries/phases to prevent context bleed. ~50ms, no model reload needed.

### Model Lifecycle
Singleton pattern: load once, reuse across images, `del + gc.collect()` to unload between models.

### Response Cleaning
`clean_response()` in pipeline.py strips markdown, extracts keywords per attribute type. Benchmark uses `clean_attribute_response()` with categorical outputs.

## Dataset
99 images from ISIC archive. Classes: NV (67), MEL (12), BKL (11), BCC (5), other (4).

## Running

```bash
# Docker shell
cd docker && docker compose run --rm derm-mcp bash

# Web UI
python3 /home/project/code/server.py

# Phase tests
python3 /home/project/code/test_phase1.py
python3 /home/project/code/test_phase2.py
python3 /home/project/code/test_phase3.py

# Multi-VLM benchmark
python3 /home/project/code/test_vlm_benchmark.py
```

## Branches
- `master` — stable main branch (PR target)
- `vlm-trial` — multi-VLM benchmark work
- Feature branches: `attribute-test`, `classification-phase3`, `fix-web-ui`, `pipeline-speed-optimization`, etc.

## Conventions
- Commit messages: imperative, concise first line describing the change
- Test scripts save CSV results to `data/`
- Models are never committed to git
- All paths in code use `/home/project/` (Docker root), not host paths
