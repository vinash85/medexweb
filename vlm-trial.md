# vlm-trial — Multi-VLM Dermoscopy Attribute Benchmark

## Goal
Compare multiple vision-language models on their ability to identify dermoscopic attributes from skin lesion images. Tests whether models can reliably extract structured clinical features (not just classify).

---

## Models (5)

| # | Model | Size | Handler | GGUF Quant |
|---|-------|------|---------|------------|
| 1 | MedGemma | 4B | Custom `MedGemmaChatHandler` (Gemma 3 turn template) | — |
| 2 | PaliGemma | 3B | `PaliGemmaChatHandler` (fallback: Llava15) | Q8_0 |
| 3 | Llama-3.2-Vision | 11B | `Llava15ChatHandler` | Q8_0 |
| 4 | Moondream2 | ~1.8B | `MoondreamChatHandler` (fallback: Llava15) | F16 |
| 5 | InternVL2.5 | 1B | `Llava15ChatHandler` | Q8_0 |

All run via llama-cpp-python with shared config: `n_ctx=2048`, `temperature=0.7`, `max_tokens=32`, `repeat_penalty=1.5`, full GPU offload (`n_gpu_layers=-1`).

Models loaded one at a time (singleton pattern), unloaded with `del + gc.collect()` between runs.

---

## Attributes (4 dermoscopy features)

| Attribute | Query Summary | Possible Values |
|-----------|--------------|-----------------|
| Pigment_Network | Net-like grid of dark lines; uniform vs irregular | absent, present_uniform, present_irregular, present_unspecified |
| Dots_Globules | Solid circular spots; central vs peripheral distribution | absent, present_central, present_peripheral, present_unspecified |
| Streaks | Radial lines or pseudopods extending from lesion border | absent, present |
| Milia_Cysts | Bright white/yellow beads or dark comedo-like openings | absent, milia_only, comedo_only, milia+comedo |

Each attribute is queried independently with a structured prompt that requests a categorical answer. Raw responses are cleaned via keyword matching in `clean_attribute_response()`.

---

## Sample

20 images from `code/DermsGemms.csv`, stratified by diagnosis:

| Class | Count | Description |
|-------|-------|-------------|
| MEL | 5 | Melanoma |
| NV | 5 | Melanocytic Nevus |
| BCC | 5 | Basal Cell Carcinoma |
| BKL | 5 | Benign Keratosis-like Lesion |

Images sorted by ISIC ID, first 5 per class selected. Standard preprocessing: 224x224 resize, 1.4x contrast enhancement, JPEG quality 90, base64-encoded.

---

## Evaluation Metrics

1. **Attribute value distributions** — Per-model breakdown of how often each categorical value is assigned
2. **Stickiness check** — Flags models that give the same answer for every image (degenerate behavior)
3. **Cross-model agreement** — For each attribute, % of images where all models agree on the same value

---

## Script

`code/test_vlm_benchmark.py` — single self-contained script.

### Running (in Docker)
```bash
python3 /home/project/code/test_vlm_benchmark.py
```

### Output
```
data/vlm_benchmark_results.csv    # Per-image, per-model, per-attribute raw + cleaned responses
data/vlm_benchmark_summary.txt    # Distributions, stickiness, cross-model agreement
```

---

## Notes
- Models that are missing from `models/` are automatically skipped
- KV cache reset (`llm.reset()`) between every attribute query to prevent context bleed
- Supports KeyboardInterrupt — saves partial results
- Branch: `vlm-trial` (from `master`)
