# DermGemma — Three-Phase Dermatology Classification Pipeline

A GPU-accelerated skin lesion classification system using MedGemma 4B, Google's medical vision-language model. The pipeline runs three sequential inference phases — specialist observation, clinical review, and classification — to produce a diagnostic code for dermoscopic images.

Built on quantized GGUF models served through llama-cpp-python with CUDA, designed to run on a single GPU inside Docker.

## Architecture Overview

```
                         ┌─────────────────┐
                         │  Dermoscopic     │
                         │  Image (224x224) │
                         └────────┬────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │  PHASE 1: Specialist       │
                    │  (Visual MedGemma + CLIP)  │
                    │                            │
                    │  4 targeted queries:       │
                    │  • Shape / Architecture    │
                    │  • Pigment network         │
                    │  • Structures              │
                    │  • Colors                  │
                    └─────────────┬─────────────┘
                                  │ raw observations
                        ┌─────── ▼ ──────┐
                        │  Model unload  │
                        │  + GC + sleep  │
                        └─────── ┬ ──────┘
                    ┌────────────▼────────────┐
                    │  PHASE 2: Comparison     │
                    │  (Visual MedGemma + CLIP)│
                    │                          │
                    │  Image + Phase 1 context │
                    │  → Verify observations   │
                    │  → Find additional signs  │
                    └────────────┬────────────┘
                                 │ clinical description
                       ┌──────── ▼ ──────┐
                       │  Model unload   │
                       │  + Sentence     │
                       │    cleaning     │
                       └──────── ┬ ──────┘
                    ┌────────────▼────────────┐
                    │  PHASE 3: Classification │
                    │  (Text-only MedGemma)    │
                    │                          │
                    │  Context + cleaned desc  │
                    │  → MEL / NV / BCC / BKL  │
                    └────────────┬────────────┘
                                 │
                         ┌───────▼───────┐
                         │  Diagnostic   │
                         │  Report       │
                         └───────────────┘
```

## The Three Phases

### Phase 1 — Specialist Observations

The visual model (MedGemma multimodal with CLIP projector) examines the dermoscopic image through four targeted queries, each designed to extract a specific clinical feature:

| Query | Expected Output |
|-------|----------------|
| Architecture (shape) | round, oval, or irregular |
| Pigment network | yes or no |
| Structures | dots, globules, streaks, or none |
| Colors | comma-separated color names |

Each response is cleaned — markdown artifacts stripped, known keywords extracted, whitespace normalized. The queries use constrained prompts (one-word or brief answers) with `max_tokens=32` to keep outputs focused.

**Why separate queries instead of one prompt?** Single-question prompts with tight token limits produce more reliable structured output from the 4B model than asking for everything at once.

### Phase 2 — Comparison Review

A fresh instance of the visual model receives the dermoscopic image along with Phase 1's specialist observations and a comparison prompt. The model is asked to:

1. Compare each Phase 1 observation against what it sees in the image (consistent / inconsistent)
2. Identify additional dermoscopic features not mentioned by Phase 1 (asymmetry, border irregularity, regression structures, vascular patterns)

The output is a multi-paragraph clinical description (~200-500 chars). This phase catches errors from Phase 1 and adds clinical context that a single-pass system would miss.

**Cleaning:** Before being passed to Phase 3, the description goes through sentence-level filtering that removes:
- Refusal phrases ("I cannot", "I'm sorry", "as an AI")
- Special token leakage (`<start_of_turn>`, `<eos>`, etc.)
- Gibberish (repeated 3+ char patterns, excessive non-ASCII, low alphabetic ratio)

### Phase 3 — Classification

A text-only MedGemma instance (no CLIP projector — lighter memory footprint) receives the Phase 1 specialist context and the cleaned Phase 2 description, then classifies the lesion into one of four codes:

| Code | Diagnosis |
|------|-----------|
| **MEL** | Melanoma |
| **NV** | Nevus (benign mole) |
| **BCC** | Basal Cell Carcinoma |
| **BKL** | Benign Keratosis |

Classification uses `max_tokens=16` and `temperature=0.3` (low creativity — this is a categorical decision, not generation). Falls back to NV if the model doesn't produce a valid code.

**Why text-only?** Phase 3 doesn't need to see the image. The clinical evidence has already been extracted and verified in Phases 1-2. Using text-only inference avoids loading the 812 MB CLIP projector and eliminates visual noise from the classification decision.

## Singleton Architecture & Memory Optimization

**The model is loaded once and reused across all phases and images — no unload/reload overhead.**

### Model Lifecycle

```
Server Start
    ↓
DermPipeline.__init__()  → Load MedGemma once into GPU VRAM
    ↓
_vlm instance created (singleton)
    ↓
[Image 1] Phase 1 → Phase 2 → Phase 3  (reuse _vlm, reset KV cache between phases)
[Image 2] Phase 1 → Phase 2 → Phase 3  (reuse _vlm, reset KV cache between phases)
[Image N] Phase 1 → Phase 2 → Phase 3  (reuse _vlm, reset KV cache between phases)
    ↓
Server Shutdown
    ↓
Model unloaded
```

### Why Singleton?

1. **Model Loading Cost**: MedGemma 4B + CLIP projector = ~4.7 GB GPU memory. Loading takes 8-15 seconds per image with traditional approach.
2. **Current Optimization**: Model loaded ONCE in `DermPipeline.__init__()`, reused for all subsequent images.
3. **Server Integration**: `server.py` maintains a global `_pipeline` singleton with thread-safe `_pipeline_lock` ensuring only one GPU analysis runs at a time.

### KV Cache Reset Strategy

Between phases, the pipeline calls `vlm.reset()` (3 times per image) to flush the KV cache:

```python
# Phase 1 → Phase 2 transition
vlm.reset()  # Clear stale context from Phase 1 queries

# Phase 2 → Phase 3 transition  
vlm.reset()  # Clear stale context from Phase 2 description
```

**Why?** llama-cpp-python's KV cache can retain previous context tokens, causing hallucination bleed (e.g., Phase 1 answers contaminating Phase 2 reasoning). Reset clears this without reloading the model (~50ms, no sleep overhead needed).

### Performance Impact

| Approach | Model Load | Per-Image Overhead | Total (100 images) |
|----------|-----------|-------------------|-------------------|
| Reload per phase | 3 × 8s per image | 24s per image | 40 minutes |
| **Singleton + Reset (Current)** | **1 × 8s total** | **~0.1s per image** | **~10 seconds** |

**Savings: 99.6% reduction in model management overhead.**

## Model Details

**MedGemma 1.5 4B IT** (google/medgemma-1.5-4b-it), quantized to GGUF format:

- `medgemma.gguf` — 3.9 GB text model (Phases 1, 2, and 3)
- `medgemma-mmproj.gguf` — 812 MB CLIP vision projector (Phases 1 and 2 only)

Served via **llama-cpp-python 0.3.16** with CUDA acceleration (`n_gpu_layers=-1` = all layers on GPU).

### Custom Chat Format

MedGemma defaults to a "thinking mode" that consumes tokens on internal reasoning before answering. The `medgemma-direct` chat format bypasses this by forcing the model to start its response with `Answer:`:

```
<start_of_turn>model
Answer:
```

This is registered as a custom chat format handler in pipeline.py and used for all three phases.

## Project Structure

```
├── code/
│   ├── pipeline.py          # Core three-phase pipeline
│   ├── server.py            # Flask API server (port 6565)
│   ├── index.html           # Web UI
│   ├── DermsGemms.csv       # Ground truth (99 images, ISIC dataset)
│   ├── test_phase1.py       # Phase 1 validation (specialist queries)
│   ├── test_phase2.py       # Phase 2 validation (comparison review)
│   └── test_phase3.py       # Phase 3 validation (end-to-end classification)
├── docker/
│   ├── Dockerfile           # NVIDIA CUDA 12.2, llama-cpp-python 0.3.16
│   └── docker-compose.yml   # GPU-enabled service
├── models/                  # GGUF model files (not committed)
└── data/
    └── images/              # ISIC dermoscopic images
```

## Dataset

99 dermoscopic images from the ISIC (International Skin Imaging Collaboration) archive with ground-truth labels:

| Diagnosis | Count |
|-----------|-------|
| NV (nevus) | 67 |
| MEL (melanoma) | 12 |
| BKL (benign keratosis) | 11 |
| BCC (basal cell carcinoma) | 5 |
| Other (akiec, df) | 4 |

Each image includes a lesion attribute descriptor (e.g., "atypical pigment network", "homogenous", "gyri/ridges") used for ground-truth comparison.

## Running

### Docker Setup

```bash
cd docker
docker compose build
docker compose run --rm derm-mcp bash
```

### Web Application

```bash
cd /home/project/code
python3 server.py
# → http://localhost:6565
```

The API exposes:
- `GET /api/list-images` — available ISIC images
- `POST /api/analyze` — runs full 3-phase pipeline, returns AI classification + ground truth

### Test Scripts

Each phase has a standalone test that validates output quality across all 99 images:

```bash
# Phase 1: specialist observation quality
python3 /home/project/code/test_phase1.py

# Phase 2: comparison review quality (97% clean rate target)
python3 /home/project/code/test_phase2.py

# Phase 3: end-to-end classification
python3 /home/project/code/test_phase3.py
```

Test scripts output per-image diagnostics, summary tables, flag counts, clean rates, and save results to CSV in `/home/project/data/`.

### Validation Flags

The test scripts check for common failure modes:

| Flag | Meaning |
|------|---------|
| `guardrail` | Model refused to answer ("I cannot...") |
| `gibberish_repeat` | Repeated token pattern (degenerate output) |
| `special_token_leak` | Raw tokens like `<eos>` in output |
| `thinking_leak` | `<think>` tags leaked into response |
| `verbose_output` | Classification output too long (>50 chars) |
| `multiple_codes` | Model output contained more than one code |
| `cleaning_emptied` | Sentence filter removed all content |
| `invalid_code` | Output wasn't MEL/NV/BCC/BKL |

## Development History

### Phase 1 — Specialist Queries
Established the base visual model integration with MedGemma multimodal. Iterated on prompt design to get single-word structured responses. Added response cleaning (keyword extraction for structures/colors, markdown stripping). Validated across 99 images with stickiness checks (ensuring the model doesn't always answer the same thing).

### Phase 2 — Comparison Review
Added a second visual pass that cross-references Phase 1 observations against the image. Key challenge was guardrail refusals (~3% of images) and gibberish from token repetition. Solved by simplifying the prompt framing — removing clinical jargon that triggered safety filters — and adding sentence-level output cleaning. Achieved 97% clean rate.

### Phase 3 — Classification
Decoupled classification from the visual model. Phase 3 uses text-only MedGemma (no CLIP projector) to classify based on the accumulated clinical evidence. The two-pass test design (visual model for all images first, then text model for all classifications) avoids repeated model load/unload cycles during batch testing. Sentence-level cleaning sanitizes Phase 2 output before it reaches the classifier.
