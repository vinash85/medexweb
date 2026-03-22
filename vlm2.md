# vlm2 — Multi-VLM Consensus Classification Pipeline

## Goal

Replace the single-model pipeline with a multi-VLM consensus architecture. Three independent vision models extract dermoscopic attributes via chain-of-thought prompting (Phase 1), MedGemma reviews those raw findings against the image and produces concordance/discordance judgments with clinical reasoning (Phase 2), and MedGemma classifies the lesion using Phase 1 observations + Phase 2 judgments + class definitions without seeing the image (Phase 3). This reduces single-model hallucination bias and enables 6-class classification.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: Multi-VLM CoT Attribute Extraction (Ollama)          │
│                                                                 │
│  Image ──► Llama-3.2-Vision ──► 7 CoT attributes (raw labels)  │
│        ──► Qwen2.5-VL       ──► 7 CoT attributes (raw labels)  │
│        ──► LLaVA-OneVision   ──► 7 CoT attributes (raw labels)  │
│                                                                 │
│  Output: raw per-model labels only — NO concordance here        │
└─────────────────────┬───────────────────────────────────────────┘
                      │ Phase 1 raw answers
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2: MedGemma Vision — Judgment + Concordance (GGUF)      │
│                                                                 │
│  Image + Phase 1 answers ──► Same 7 CoT questions               │
│                                                                 │
│  For each attribute, MedGemma:                                  │
│    1. Performs its own CoT visual analysis                       │
│    2. Compares its finding with Phase 1 model answers           │
│    3. Reports:                                                  │
│       - ASSESSMENT: its own label                               │
│       - CONCORDANCE: concordant or discordant with Phase 1      │
│       - JUDGMENT: clinical opinion — what it observes, why it   │
│         agrees/disagrees, diagnostic significance (2-3 sent.)   │
│                                                                 │
│  Output: label + concordance + judgment per attribute            │
└─────────────────────┬───────────────────────────────────────────┘
                      │ Phase 1 raw answers
                      │ + Phase 2 judgments & concordance
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  Phase 3: MedGemma Text-Only Classification (no image)         │
│                                                                 │
│  Input:                                                         │
│    Section 1: Phase 1 raw multi-VLM answers per attribute       │
│    Section 2: Phase 2 MedGemma judgments + concordance          │
│    Section 3: Dermoscopic class definitions (6 classes)         │
│    Section 4: Classification instruction                        │
│  Output:                                                        │
│    - Single classification: MEL | NV | BCC | BKL | AKIEC | DF  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Per Attribute

```
Phase 1 (Ollama VLMs):
  Llama-3.2-Vision → "typical"
  Qwen2.5-VL       → "atypical"
  LLaVA-OneVision   → "typical"
            │
            ▼
Phase 2 (MedGemma Vision):
  Sees image + "Llama: typical, Qwen: atypical, LLaVA: typical"
  Performs own CoT analysis on the image
  Outputs:
    ASSESSMENT: typical
    CONCORDANCE: concordant
    JUDGMENT: "The network shows thin, regular intersecting lines forming
    a uniform honeycomb grid. This is consistent with the majority view
    from Phase 1. The regular pattern suggests a benign melanocytic lesion."
            │
            ▼
Phase 3 (MedGemma Text-Only):
  Sees Phase 1 raw: "Llama=typical, Qwen=atypical, LLaVA=typical"
  Sees Phase 2: "typical [CONCORDANT]"
  Sees Phase 2 opinion: "thin regular lines... suggests benign..."
  Sees class definitions for MEL, NV, BCC, BKL, AKIEC, DF
  → CLASSIFICATION: NV
```

---

## Models

### Phase 1 — Ollama VLMs (3 models)

| Model | Ollama Tag | Size | Role |
|-------|-----------|------|------|
| Llama-3.2-Vision | `llama3.2-vision:11b` | 11B | Independent attribute extractor |
| Qwen2.5-VL | `qwen2.5vl:7b` | 7B | Independent attribute extractor |
| LLaVA-OneVision | `llava:13b` | 13B | Independent attribute extractor |

Config: `temperature=0.7`, `max_tokens=512`, `repeat_penalty=1.5`. Models managed by Ollama server (not deleted between images).

### Phase 2 + 3 — MedGemma (llama-cpp-python GGUF)

| Model | File | Size | Role |
|-------|------|------|------|
| MedGemma 4B | `medgemma.gguf` + `medgemma-mmproj.gguf` | ~4.7 GB | Vision review + judgments (Phase 2), text-only classification (Phase 3) |

Config: `n_ctx=4096`, `n_gpu_layers=-1`, custom `MedGemmaChatHandler` (Gemma 3 turn template). Loaded once as singleton.

---

## Attributes (7 dermoscopic features, CoT prompting)

Each attribute uses a structured chain-of-thought prompt that forces step-by-step visual analysis before a categorical label.

| Attribute | CoT Steps | Labels |
|-----------|-----------|--------|
| Pigment_Network | Grid detection → mesh consistency → line morphology | absent, typical, atypical |
| Dots_Globules | Primitive count → size sorting → spatial map | absent, regular, irregular |
| Vascular_Structures | Color isolation → path analysis → distribution | absent, comma, arborizing, dotted, polymorphous |
| Blue_White_Structures | Spectral scan → opacity check → border definition | absent, blue_white_veil, regression_peppering |
| Streaks | Edge detection → directional mapping → symmetry | absent, regular, irregular |
| Milia_Cysts | Contrast peak → shape check → artifact filter | absent, starry, cloudy |
| Blotches | Texture density → light transmission | absent, regular, irregular |

System prefix avoids medical terminology bias:
> "You are a computer vision expert specializing in pattern recognition and spatial analysis of close-up surface images. Analyze the image strictly based on visual patterns you observe. Do not assume any medical or domain context."

---

## Phase 1 — Raw Multi-VLM Extraction

Phase 1 outputs raw per-model labels only. No concordance is computed here — that is Phase 2's job.

Each Ollama model independently answers all 7 CoT queries on the image. Label extraction uses:
1. Regex for explicit `LABEL: <value>` line
2. Fallback: scan last 3 lines for valid label keywords
3. Fallback: attribute-specific keyword matching on full text

Output structure:
```python
{
    "per_model": {
        "Llama-3.2-Vision":  {"Pigment_Network": {"cot_label": "typical", "raw": "..."}, ...},
        "Qwen2.5-VL":        {"Pigment_Network": {"cot_label": "atypical", "raw": "..."}, ...},
        "LLaVA-OneVision":   {"Pigment_Network": {"cot_label": "typical", "raw": "..."}, ...},
    },
    "models_used": ["Llama-3.2-Vision", "Qwen2.5-VL", "LLaVA-OneVision"],
    "models_failed": [],
}
```

---

## Phase 2 — MedGemma Vision: Judgment + Concordance

For each attribute, MedGemma receives:

```
Three vision models previously analyzed this dermoscopy image for {attribute}.

Their assessments:
- Llama-3.2-Vision: {label}
- Qwen2.5-VL: {label}
- LLaVA-OneVision: {label}

Now examine the image yourself using the same analysis framework.

{full CoT prompt with steps}

After your own analysis, compare your finding with the Phase 1 models above
and provide your clinical judgment.

End with exactly these lines:
ASSESSMENT: <your label>
CONCORDANCE: <concordant or discordant>
JUDGMENT: <your clinical opinion — what you observe in the image, why you
agree or disagree with Phase 1, what this feature suggests about the lesion,
and any diagnostic significance. 2-3 sentences.>
```

### Extraction

- **ASSESSMENT**: regex → `LABEL:` regex → keyword fallback
- **CONCORDANCE**: regex for `concordant`/`discordant`; if missing, inferred by comparing MedGemma's label against Phase 1 labels
- **JUDGMENT**: everything after `JUDGMENT:` tag, cleaned for gibberish/refusals; falls back to full cleaned response if tag is missing

### Output structure
```python
{
    "per_attribute": {
        "Pigment_Network": {
            "label": "typical",
            "concordance": "concordant",
            "judgment": "The network shows thin regular lines in a uniform grid...",
            "phase1_labels": "Llama-3.2-Vision: typical, Qwen2.5-VL: atypical, LLaVA-OneVision: typical",
            "raw": "...",
        },
        ...
    },
    "summary": "Pigment_Network: MedGemma=typical [concordant] (Phase1: ...)\n...",
}
```

---

## Phase 3 — Text-Only Classification

Text-only prompt (no image, `content` as plain string) with 4 sections:

### Section 1: Phase 1 Raw Multi-VLM Observations
```
=== PHASE 1: MULTI-VLM ATTRIBUTE OBSERVATIONS ===
  Pigment Network: Llama-3.2-Vision=typical, Qwen2.5-VL=atypical, LLaVA-OneVision=typical
  Dots and Globules: Llama-3.2-Vision=absent, Qwen2.5-VL=absent, LLaVA-OneVision=regular
  ...
```

### Section 2: Phase 2 MedGemma Clinical Judgments
```
=== PHASE 2: MEDGEMMA CLINICAL JUDGMENTS ===

  Pigment Network: typical [CONCORDANT]
    Opinion: The network shows thin regular lines in a uniform honeycomb grid.
    This is consistent with the majority Phase 1 view. Suggests benign melanocytic lesion.

  Dots and Globules: absent [CONCORDANT]
    Opinion: No circumscribed circular features visible. The surface is smooth
    without discrete dots or globular structures.
  ...
```

### Section 3: Lesion Class Definitions

| Class | Key Dermoscopic Features |
|-------|------------------------|
| MEL | Atypical pigment network, irregular dots/globules, blue-white veil, irregular streaks, regression, asymmetric blotches, polymorphous vessels |
| NV | Typical pigment network, regular dots/globules, symmetric blotches, homogeneous pattern |
| BCC | Arborizing vessels, blue-grey ovoid nests, shiny white structures, ulceration, absent pigment network |
| BKL | Comedo-like openings, milia-like cysts, gyri/ridges, moth-eaten border, clod pattern, fissures |
| AKIEC | Surface scale/crust, radial dots, white clods, rosettes, glomerular/dotted vessels |
| DF | Central white scar-like patch, delicate peripheral pigment network, ring-like globules |

### Section 4: Classification Instruction

> Weight CONCORDANT attributes (MedGemma agrees with Phase 1) more heavily than DISCORDANT ones. When MedGemma disagrees with Phase 1, prefer MedGemma's assessment as it is the medical domain specialist.

Output: `CLASSIFICATION: <code>`

---

## Dataset

99 images from ISIC archive via `code/DermsGemms.csv`:

| Class | Count | Description |
|-------|-------|-------------|
| NV | 67 | Melanocytic Nevus |
| MEL | 12 | Melanoma |
| BKL | 11 | Benign Keratosis-like Lesion |
| BCC | 5 | Basal Cell Carcinoma |
| AKIEC | 4 | Actinic Keratosis / Intraepithelial Carcinoma |
| DF | 1 | Dermatofibroma |

Image preprocessing: 448x448 resize (2x SigLIP tile), 1.4x contrast enhancement, JPEG quality 90, base64-encoded.

---

## Files

| File | Role |
|------|------|
| `code/pipeline.py` | Core 3-phase multi-VLM consensus pipeline (rewritten) |
| `code/test_vlm2_pipeline.py` | Benchmark test — 25 images (5/class), saves all phase results |
| `code/server.py` | Flask API (unchanged, imports DermPipeline) |
| `code/index.html` | Web UI (unchanged) |
| `code/DermsGemms.csv` | Ground truth dataset |
| `docker/docker-compose.yml` | Docker GPU config |

### Test Script Output

`test_vlm2_pipeline.py` saves:
- `data/vlm2_pipeline_results.csv` — per-image: Phase 1 attributes, Phase 2 summary, Phase 3 report, prediction, match/miss, timing
- `data/vlm2_pipeline_summary.txt` — accuracy, per-class breakdown, confusion matrix, timing stats

---

## Key Design Decisions

1. **Concordance is Phase 2's job, not Phase 1** — Phase 1 produces raw independent observations; MedGemma (the medical specialist) determines concordance/discordance by comparing its own visual analysis against Phase 1
2. **Phase 2 produces clinical judgments** — not just labels and flags, but 2-3 sentence opinions with diagnostic reasoning that flow into Phase 3
3. **No rule-based shortcuts** — Phase 3 uses pure LLM classification informed by Phase 2 clinical reasoning, replacing hardcoded MEL/BKL rules
4. **6 classes** (added AKIEC, DF) — the full dataset class set instead of the previous 4
5. **448x448 resolution** — 2x SigLIP tile size for clean tiling in MedGemma's vision encoder
6. **n_ctx=4096** (up from 2048) — Phase 2 prompts include Phase 1 context + CoT; Phase 3 includes judgments from both phases
7. **Ollama models not deleted** after use — reused for subsequent images
8. **Graceful fallback** — if Ollama is unavailable, Phase 1 falls back to MedGemma-only; Phase 2 still runs (reviewing its own Phase 1 answers); Phase 3 unchanged
9. **Server.py interface preserved** — `DermPipeline.process(filename)` returns the same `{ai_code, raw_obs, adj_desc, final_implication}` dict

---

## Running

```bash
# Start Docker
cd docker && docker compose run --rm derm-mcp bash

# Ensure Ollama is running with models available
ollama list  # should show llama3.2-vision:11b, qwen2.5vl:7b, llava:13b

# Pipeline benchmark (25 images, all phases, CSV output)
python3 /home/project/code/test_vlm2_pipeline.py

# Or single-image test
cd /home/project/code
python3 -c "from pipeline import DermPipeline; p = DermPipeline(); print(p.process('ISIC_0024552.jpg'))"

# Web UI (optional)
python3 /home/project/code/server.py
```

---

## Performance

| Phase | Time Estimate | Bottleneck |
|-------|--------------|------------|
| Phase 1 | ~105-210s | 3 models x 7 attributes x 5-10s per Ollama query |
| Phase 2 | ~35s | 7 attributes x 5s per MedGemma query |
| Phase 3 | ~2s | 1 text-only query |
| **Total** | **~2.5-4 min per image** | Phase 1 dominates |

---

## Differences from Master

| Aspect | Master (`pipeline.py`) | vlm2 (`pipeline.py`) |
|--------|----------------------|---------------------|
| Phase 1 models | MedGemma only (7 direct queries) | 3 Ollama VLMs x 7 CoT queries |
| Phase 1 prompting | Direct questions ("Answer with one word") | Chain-of-thought (multi-step visual analysis) |
| Phase 1 output | Single-model cleaned observations | Raw per-model labels (3 independent views) |
| Phase 2 role | Compare observations vs image | Clinical judgment + concordance/discordance with Phase 1 |
| Phase 2 output | Prose description paragraph | Per-attribute: label + concordant/discordant + clinical opinion |
| Phase 3 input | Rule-based MEL/BKL + fallback text prompt | Phase 1 raw + Phase 2 judgments + 6-class definitions |
| Phase 3 method | Hardcoded rules + LLM fallback (4 classes) | Pure LLM classification (6 classes) |
| Image resolution | 224x224 | 448x448 |
| Dependencies | llama-cpp-python only | llama-cpp-python + ollama |
| Context window | 2048 | 4096 |

---

## TODO — Testing

- [ ] Start Docker with GPU: `cd docker && docker compose run --rm derm-mcp bash`
- [ ] Install ollama Python package: `pip install ollama`
- [ ] Verify Ollama server is reachable: `python3 -c "import ollama; print(ollama.list())"`
- [ ] Pull Phase 1 models: `ollama pull llama3.2-vision:11b && ollama pull qwen2.5vl:7b && ollama pull llava:13b`
- [ ] Run benchmark test: `python3 /home/project/code/test_vlm2_pipeline.py`
- [ ] Check results: `cat /home/project/data/vlm2_pipeline_summary.txt`
- [ ] Test MedGemma-only fallback — stop Ollama, rerun benchmark, verify single-model mode completes

---

## Branch

- **Branch**: `vlm2` (from `master`)
- **Parent**: `master` — stable single-model pipeline
- **Related**: `vlm-trial` — multi-VLM attribute benchmarks (groundwork for this branch)
