#!/usr/bin/env python3
"""
Ollama CoT Resolution Test — Llama, Qwen, LLaVA-OneVision
==========================================================
Same CoT prompts as the main benchmark but with image resolution
increased from 224x224 to 1120x1120 (max native tile size for
Llama 3.2 Vision; Qwen2.5-VL supports dynamic resolution even
higher; LLaVA internally uses 336px but benefits from sharper input).

Tests whether higher-resolution input improves attribute extraction.

Run inside Docker:
    python3 /home/project/code/test_ollama_cot_resolution.py
"""

import os
import re
import csv
import sys
import time
import base64
import traceback
from datetime import datetime

import pandas as pd
from PIL import Image, ImageEnhance

try:
    import ollama
except ImportError:
    print("ERROR: ollama python package not installed. Run: pip install ollama")
    sys.exit(1)

# ── Paths (Docker layout) ────────────────────────────────────────────────────
PROJECT_ROOT = "/home/project"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
CSV_PATH = os.path.join(PROJECT_ROOT, "code", "DermsGemms.csv")
RESULTS_CSV = os.path.join(DATA_DIR, "ollama_cot_resolution_test_results.csv")
SUMMARY_TXT = os.path.join(DATA_DIR, "ollama_cot_resolution_test_summary.txt")

# ── Resolution ───────────────────────────────────────────────────────────────
# Max native among the 3 models: Llama 3.2 Vision tiles at 560px (up to 1120),
# Qwen2.5-VL supports dynamic resolution (>2048px), LLaVA uses 336px internally.
# Using 1120x1120 — the highest that at least 2 of 3 models can fully utilise.
IMAGE_RESOLUTION = 1120

# ── Model Registry (3 models only) ──────────────────────────────────────────
MODEL_REGISTRY = [
    {
        "name": "Llama-3.2-Vision",
        "ollama_model": "llama3.2-vision:11b",
        "temperature": 0.7,
        "max_tokens": 512,
    },
    {
        "name": "Qwen2.5-VL",
        "ollama_model": "qwen2.5vl:7b",
        "temperature": 0.7,
        "max_tokens": 512,
    },
    {
        "name": "LLaVA-OneVision",
        "ollama_model": "llava:13b",
        "temperature": 0.7,
        "max_tokens": 512,
    },
]

# ── CoT Attribute Queries ────────────────────────────────────────────────────

SYSTEM_PREFIX = (
    "You are a computer vision expert specializing in pattern recognition "
    "and spatial analysis of close-up surface images. "
    "Analyze the image strictly based on visual patterns you observe. "
    "Do not assume any medical or domain context.\n\n"
)

COT_QUERIES = {
    "Pigment_Network": {
        "prompt": SYSTEM_PREFIX + (
            "Analyze the line-network pattern in this image step by step.\n\n"
            "Step 1 — Grid Detection: Identify if there is a recurring lattice or "
            "honeycomb pattern of intersecting dark lines across the surface.\n\n"
            "Step 2 — Mesh Consistency: Are the holes in the grid uniform in size, "
            "or do they vary significantly (meshes becoming larger or distorted)?\n\n"
            "Step 3 — Line Morphology: Describe the lines of the grid. Are they thin "
            "and delicate, or thick, dark, and smudged?\n\n"
            "After completing all steps, state your final answer on a new line as:\n"
            "LABEL: <one of: absent, typical, atypical>"
        ),
        "labels": ["absent", "typical", "atypical"],
    },
    "Dots_Globules": {
        "prompt": SYSTEM_PREFIX + (
            "Analyze circular point features in this image step by step.\n\n"
            "Step 1 — Primitive Count: Locate all small, circumscribed circular "
            "primitives within the region of interest.\n\n"
            "Step 2 — Size Sorting: Differentiate between points (pinprick size) "
            "and globes (larger, distinct circles).\n\n"
            "Step 3 — Spatial Map: Are these circles concentrated in the center of "
            "the region, or scattered randomly at the periphery/edge?\n\n"
            "After completing all steps, state your final answer on a new line as:\n"
            "LABEL: <one of: absent, regular, irregular>"
        ),
        "labels": ["absent", "regular", "irregular"],
    },
    "Vascular_Structures": {
        "prompt": SYSTEM_PREFIX + (
            "Analyze linear red features in this image step by step.\n\n"
            "Step 1 — Color Isolation: Filter for red or pinkish tones within the "
            "region. Identify if these appear as dots, curves, or branches.\n\n"
            "Step 2 — Path Analysis: Describe the shape of the red features. Are they "
            "C-shaped (comma-like), tree-like (branching), or twisted loops?\n\n"
            "Step 3 — Distribution: Are the red structures confined to a specific "
            "zone or distributed throughout the region?\n\n"
            "After completing all steps, state your final answer on a new line as:\n"
            "LABEL: <one of: absent, comma, arborizing, dotted, polymorphous>"
        ),
        "labels": ["absent", "comma", "arborizing", "dotted", "polymorphous"],
    },
    "Blue_White_Structures": {
        "prompt": SYSTEM_PREFIX + (
            "Analyze blue and white color regions in this image step by step.\n\n"
            "Step 1 — Spectral Scan: Identify areas with a steel-blue or gray-blue "
            "hue anywhere in the image.\n\n"
            "Step 2 — Opacity Check: Does this area look like it has a frosted glass "
            "or ground glass white film overlaying the blue pigment?\n\n"
            "Step 3 — Border Definition: Is the blue area well-defined with sharp "
            "edges, or does it have blurry, ill-defined margins?\n\n"
            "After completing all steps, state your final answer on a new line as:\n"
            "LABEL: <one of: absent, blue_white_veil, regression_peppering>"
        ),
        "labels": ["absent", "blue_white_veil", "regression_peppering"],
    },
    "Streaks": {
        "prompt": SYSTEM_PREFIX + (
            "Analyze radial projections in this image step by step.\n\n"
            "Step 1 — Edge Detection: Examine the outermost boundary of the dark "
            "region in the image.\n\n"
            "Step 2 — Directional Mapping: Identify any linear projections or "
            "spokes that emanate outward from the center like a starburst pattern.\n\n"
            "Step 3 — Symmetry: Are these projections appearing around the entire "
            "circumference (symmetric/starburst), or only in one focal area "
            "(asymmetric/focal)?\n\n"
            "After completing all steps, state your final answer on a new line as:\n"
            "LABEL: <one of: absent, regular, irregular>"
        ),
        "labels": ["absent", "regular", "irregular"],
    },
    "Milia_Cysts": {
        "prompt": SYSTEM_PREFIX + (
            "Analyze bright sphere features in this image step by step.\n\n"
            "Step 1 — Contrast Peak: Identify small, distinct ivory or bright white "
            "circular structures within the region.\n\n"
            "Step 2 — Shape Check: Are they perfectly round small globes (starry), "
            "or larger and cloudy/opaque clusters?\n\n"
            "Step 3 — Artifact Filter: Differentiate from surface reflections or "
            "bubbles which usually have a specular (brightest point) highlight. "
            "True bright spheres are embedded within the surface texture.\n\n"
            "After completing all steps, state your final answer on a new line as:\n"
            "LABEL: <one of: absent, starry, cloudy>"
        ),
        "labels": ["absent", "starry", "cloudy"],
    },
    "Blotches": {
        "prompt": SYSTEM_PREFIX + (
            "Analyze structureless dark masses in this image step by step.\n\n"
            "Step 1 — Texture Density: Locate any dark area where the underlying "
            "pattern (like a grid or dots) is completely obscured by a solid mass "
            "of dark pigment.\n\n"
            "Step 2 — Light Transmission: Can you see through the dark area to any "
            "underlying structure, or is it 100% opaque and structureless?\n\n"
            "After completing all steps, state your final answer on a new line as:\n"
            "LABEL: <one of: absent, regular, irregular>"
        ),
        "labels": ["absent", "regular", "irregular"],
    },
}

# ── CoT Label → Benchmark Label Mapping ──────────────────────────────────────

COT_TO_BENCHMARK = {
    "Pigment_Network": {
        "typical": "present_uniform",
        "atypical": "present_irregular",
        "absent": "absent",
    },
    "Dots_Globules": {
        "regular": "present_central",
        "irregular": "present_peripheral",
        "absent": "absent",
    },
    "Vascular_Structures": {
        "comma": "other_vascular",
        "arborizing": "arborizing",
        "dotted": "dotted_glomerular",
        "polymorphous": "other_vascular",
        "absent": "absent",
    },
    "Blue_White_Structures": {
        "blue_white_veil": "blue_white_veil",
        "regression_peppering": "present_unspecified",
        "absent": "absent",
    },
    "Streaks": {
        "regular": "present",
        "irregular": "present",
        "absent": "absent",
    },
    "Milia_Cysts": {
        "starry": "milia_only",
        "cloudy": "comedo_only",
        "absent": "absent",
    },
    "Blotches": {
        "regular": "symmetric",
        "irregular": "asymmetric",
        "absent": "absent",
    },
}


def map_to_benchmark_label(attr_key, cot_label):
    """Convert a CoT prompt label to the original benchmark label for CSV output."""
    mapping = COT_TO_BENCHMARK.get(attr_key, {})
    return mapping.get(cot_label, cot_label)


# ── Stratified Sample ────────────────────────────────────────────────────────
SAMPLE_CLASSES = ["mel", "nv", "bcc", "bkl", "akiec", "df"]
N_PER_CLASS = 5


def select_sample_images():
    """Select up to N_PER_CLASS images per class from DermsGemms.csv."""
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip() for c in df.columns]
    sample_rows = []
    for dx in SAMPLE_CLASSES:
        subset = df[df["Dx"] == dx].sort_values("Image").head(N_PER_CLASS)
        n_available = len(df[df["Dx"] == dx])
        n_selected = len(subset)
        if n_selected < N_PER_CLASS:
            print(f"  Note: {dx} has only {n_available} image(s), using all of them")
        for _, row in subset.iterrows():
            sample_rows.append({
                "image_id": row["Image"],
                "dx": row["Dx"],
                "gt_attributes": row.get("Lesion attributes", ""),
            })
    return sample_rows


# ── Image Preparation (HIGH RESOLUTION) ─────────────────────────────────────

def prepare_image_base64(filepath):
    """Resize to IMAGE_RESOLUTIONxIMAGE_RESOLUTION, enhance contrast, return base64-encoded JPEG."""
    import io
    with Image.open(filepath) as img:
        img = img.convert("RGB").resize(
            (IMAGE_RESOLUTION, IMAGE_RESOLUTION), Image.Resampling.LANCZOS
        )
        img = ImageEnhance.Contrast(img).enhance(1.4)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Model Pull ────────────────────────────────────────────────────────────────

def ensure_model_available(ollama_model):
    """Check if model is locally available; pull if missing. Returns True if ready."""
    try:
        models = ollama.list()
        local_names = []
        for m in models.get("models", []):
            local_names.append(m.get("name", ""))
            local_names.append(m.get("name", "").split(":")[0])
        model_base = ollama_model.split(":")[0]
        if ollama_model in local_names or model_base in local_names:
            return True
    except Exception as e:
        print(f"    Warning: could not list models ({e}), attempting pull anyway...")

    print(f"    Model '{ollama_model}' not found locally. Pulling...")
    try:
        current_digest = ""
        for progress in ollama.pull(ollama_model, stream=True):
            digest = progress.get("digest", "")
            status = progress.get("status", "")
            total = progress.get("total") or 0
            completed = progress.get("completed") or 0
            if digest and digest != current_digest:
                current_digest = digest
                print(f"\n    [{status}]", end="", flush=True)
            elif total > 0:
                pct = 100.0 * completed / total
                print(f"\r    [{status}] {pct:.0f}%", end="", flush=True)
            else:
                print(f"\r    [{status}]", end="", flush=True)
        print()
        print(f"    Pull complete: {ollama_model}")
        return True
    except Exception as e:
        print(f"    Pull failed for '{ollama_model}': {e}")
        return False


# ── Ollama Query ──────────────────────────────────────────────────────────────

def query_ollama(ollama_model, image_b64, prompt, temperature=0.7, max_tokens=512):
    """Send a vision query to Ollama and return the response text."""
    response = ollama.chat(
        model=ollama_model,
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [image_b64],
        }],
        options={
            "temperature": temperature,
            "num_predict": max_tokens,
            "repeat_penalty": 1.5,
        },
    )
    return response["message"]["content"].strip()


# ── CoT Response Cleaning ────────────────────────────────────────────────────

def extract_cot_label(raw, attr_key):
    """Extract the final label from a CoT response."""
    valid_labels = COT_QUERIES[attr_key]["labels"]
    text = raw.strip()

    # Primary: find explicit LABEL: line
    label_match = re.search(r"LABEL:\s*(.+)", text, re.IGNORECASE)
    if label_match:
        label_text = label_match.group(1).strip().lower()
        label_text = re.sub(r"[*\"`'_\[\](){}]", "", label_text).strip()
        for lbl in valid_labels:
            if lbl == label_text:
                return lbl
        for lbl in valid_labels:
            if lbl in label_text:
                return lbl

    # Fallback: scan last 3 lines for any valid label keyword
    last_lines = "\n".join(text.split("\n")[-3:]).lower()
    last_lines = re.sub(r"[*\"`'_\[\](){}]", "", last_lines)
    for lbl in valid_labels:
        if lbl in last_lines:
            return lbl

    # Attribute-specific keyword fallback on full text
    lower = text.lower()
    return _keyword_fallback(lower, attr_key, valid_labels)


def _keyword_fallback(text, attr_key, valid_labels):
    """Last-resort keyword matching when LABEL: line is missing."""

    if attr_key == "Pigment_Network":
        if "atypical" in text or "irregular" in text or "distort" in text:
            return "atypical"
        if "typical" in text or "uniform" in text or "regular" in text or "delicate" in text:
            return "typical"
        if "absent" in text or "no grid" in text or "no lattice" in text or "no network" in text:
            return "absent"
        if "honeycomb" in text or "lattice" in text or "grid" in text or "network" in text:
            return "typical"

    elif attr_key == "Dots_Globules":
        if "irregular" in text or "scatter" in text or "peripher" in text or "random" in text:
            return "irregular"
        if "regular" in text or "central" in text or "homogen" in text or "uniform" in text:
            return "regular"
        if "absent" in text or "no dot" in text or "no glob" in text or "none" in text:
            return "absent"
        if "dot" in text or "glob" in text or "circle" in text or "point" in text:
            return "regular"

    elif attr_key == "Vascular_Structures":
        if "arboriz" in text or "branch" in text or "tree" in text:
            return "arborizing"
        if "comma" in text or "c-shape" in text or "curved" in text:
            return "comma"
        if "dotted" in text or "glomerul" in text or "punctate" in text or "loop" in text:
            return "dotted"
        if "polymorphous" in text or "multiple" in text or "mixed" in text:
            return "polymorphous"
        if "absent" in text or "no vessel" in text or "no vascular" in text or "none" in text:
            return "absent"
        if "vessel" in text or "vascular" in text or "red" in text:
            return "dotted"

    elif attr_key == "Blue_White_Structures":
        if "regression" in text or "pepper" in text or "granular" in text:
            return "regression_peppering"
        if "veil" in text or ("blue" in text and "white" in text):
            return "blue_white_veil"
        if "absent" in text or "no blue" in text or "none" in text:
            return "absent"
        if "blue" in text or "steel" in text or "gray" in text:
            return "blue_white_veil"

    elif attr_key == "Streaks":
        if "irregular" in text or "focal" in text or "asymmetr" in text:
            return "irregular"
        if "regular" in text or "starburst" in text or "symmetr" in text or "circumferen" in text:
            return "regular"
        if "absent" in text or "no streak" in text or "no projection" in text or "none" in text:
            return "absent"
        if "streak" in text or "radial" in text or "spoke" in text or "pseudopod" in text:
            return "irregular"

    elif attr_key == "Milia_Cysts":
        if "cloudy" in text or "opaque" in text or "large" in text or "cluster" in text:
            return "cloudy"
        if "starry" in text or "small" in text or "distinct" in text or "round" in text:
            return "starry"
        if "absent" in text or "no milia" in text or "none" in text or "no bright" in text:
            return "absent"
        if "milia" in text or "ivory" in text or "white" in text or "bead" in text:
            return "starry"

    elif attr_key == "Blotches":
        if "irregular" in text or "asymmetr" in text or "eccentric" in text:
            return "irregular"
        if "regular" in text or "symmetr" in text or "central" in text or "uniform" in text:
            return "regular"
        if "absent" in text or "no blotch" in text or "none" in text or "no dark" in text:
            return "absent"
        if "blotch" in text or "opaque" in text or "structureless" in text or "solid" in text:
            return "irregular"

    return "unclear"


# ── Benchmark Runner ──────────────────────────────────────────────────────────

def run_benchmark():
    print("=" * 70)
    print("Ollama CoT RESOLUTION TEST (1120x1120)")
    print(f"Models: {', '.join(m['name'] for m in MODEL_REGISTRY)}")
    print(f"Resolution: {IMAGE_RESOLUTION}x{IMAGE_RESOLUTION} (up from 224x224)")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # Verify Ollama is reachable
    try:
        ollama.list()
        print("Ollama server: connected")
    except Exception as e:
        print(f"ERROR: Cannot connect to Ollama server: {e}")
        print("Make sure Ollama is running (ollama serve)")
        sys.exit(1)

    # Select sample images
    sample = select_sample_images()
    class_counts = {}
    for s in sample:
        class_counts[s["dx"]] = class_counts.get(s["dx"], 0) + 1
    class_desc = ", ".join(f"{dx.upper()}={n}" for dx, n in class_counts.items())
    print(f"\nSample: {len(sample)} images ({class_desc})")
    for s in sample:
        print(f"  {s['image_id']} ({s['dx']})")

    # Verify images exist
    missing_images = []
    for s in sample:
        img_path = os.path.join(IMAGE_DIR, f"{s['image_id']}.jpg")
        if not os.path.exists(img_path):
            missing_images.append(s["image_id"])
    if missing_images:
        print(f"\nWARNING: {len(missing_images)} images not found:")
        for m in missing_images:
            print(f"  {m}")

    # Results accumulator
    all_results = []
    model_status = {}
    model_timings = {}  # Track per-model total time

    try:
        for cfg in MODEL_REGISTRY:
            model_name = cfg["name"]
            ollama_model = cfg["ollama_model"]
            print(f"\n{'─' * 60}")
            print(f"Model: {model_name} ({ollama_model})")
            print(f"{'─' * 60}")

            if not ensure_model_available(ollama_model):
                model_status[model_name] = f"SKIPPED — pull failed for {ollama_model}"
                continue

            model_status[model_name] = "OK"
            model_t0 = time.time()

            for img_idx, s in enumerate(sample):
                image_id = s["image_id"]
                img_path = os.path.join(IMAGE_DIR, f"{image_id}.jpg")

                if not os.path.exists(img_path):
                    print(f"  [{img_idx+1}/{len(sample)}] {image_id} — MISSING, skipping")
                    row = {
                        "model": model_name,
                        "image_id": image_id,
                        "dx": s["dx"],
                        "gt_attributes": s["gt_attributes"],
                        "resolution": f"{IMAGE_RESOLUTION}x{IMAGE_RESOLUTION}",
                    }
                    for attr_key in COT_QUERIES:
                        row[f"{attr_key}_raw"] = "IMAGE_MISSING"
                        row[f"{attr_key}_cot_label"] = "IMAGE_MISSING"
                        row[f"{attr_key}_label"] = "IMAGE_MISSING"
                    all_results.append(row)
                    continue

                try:
                    image_b64 = prepare_image_base64(img_path)
                except Exception as e:
                    print(f"  [{img_idx+1}/{len(sample)}] {image_id} — image prep failed: {e}")
                    row = {
                        "model": model_name,
                        "image_id": image_id,
                        "dx": s["dx"],
                        "gt_attributes": s["gt_attributes"],
                        "resolution": f"{IMAGE_RESOLUTION}x{IMAGE_RESOLUTION}",
                    }
                    for attr_key in COT_QUERIES:
                        row[f"{attr_key}_raw"] = f"PREP_ERROR: {e}"
                        row[f"{attr_key}_cot_label"] = "ERROR"
                        row[f"{attr_key}_label"] = "ERROR"
                    all_results.append(row)
                    continue

                row = {
                    "model": model_name,
                    "image_id": image_id,
                    "dx": s["dx"],
                    "gt_attributes": s["gt_attributes"],
                    "resolution": f"{IMAGE_RESOLUTION}x{IMAGE_RESOLUTION}",
                }

                for attr_key, attr_cfg in COT_QUERIES.items():
                    try:
                        t0 = time.time()
                        raw_answer = query_ollama(
                            ollama_model,
                            image_b64,
                            attr_cfg["prompt"],
                            temperature=cfg["temperature"],
                            max_tokens=cfg["max_tokens"],
                        )
                        elapsed = time.time() - t0
                        cot_label = extract_cot_label(raw_answer, attr_key)
                        benchmark_label = map_to_benchmark_label(attr_key, cot_label)
                        row[f"{attr_key}_raw"] = raw_answer
                        row[f"{attr_key}_cot_label"] = cot_label
                        row[f"{attr_key}_label"] = benchmark_label
                        print(f"    {attr_key}: {cot_label} -> {benchmark_label} ({elapsed:.1f}s)")

                    except Exception as e:
                        print(f"    ERROR on {image_id}/{attr_key}: {e}")
                        row[f"{attr_key}_raw"] = f"ERROR: {e}"
                        row[f"{attr_key}_cot_label"] = "ERROR"
                        row[f"{attr_key}_label"] = "ERROR"

                all_results.append(row)
                label_summary = " ".join(
                    f"{k[:2]}={row.get(f'{k}_label', '?')}"
                    for k in COT_QUERIES
                )
                print(f"  [{img_idx+1}/{len(sample)}] {image_id} ({s['dx']}): {label_summary}")

            model_elapsed = time.time() - model_t0
            model_timings[model_name] = model_elapsed
            print(f"  {model_name} total time: {model_elapsed:.1f}s")

            # Delete model after getting results to free disk/GPU
            print(f"  Deleting {ollama_model} to free resources...")
            try:
                ollama.delete(ollama_model)
                print(f"  Deleted {ollama_model}.")
            except Exception as e:
                print(f"  Warning: could not delete {ollama_model}: {e}")

    except KeyboardInterrupt:
        print("\n\nKeyboardInterrupt — saving partial results...")

    # ── Save Results ──────────────────────────────────────────────────────────

    if not all_results:
        print("\nNo results to save.")
        return

    fieldnames = ["model", "image_id", "dx", "gt_attributes", "resolution"]
    for attr_key in COT_QUERIES:
        fieldnames.extend([f"{attr_key}_raw", f"{attr_key}_cot_label", f"{attr_key}_label"])

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)
    print(f"\nResults saved: {RESULTS_CSV} ({len(all_results)} rows)")

    # ── Build Summary ─────────────────────────────────────────────────────────
    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("Ollama CoT RESOLUTION TEST — Summary")
    summary_lines.append(f"Generated: {datetime.now().isoformat()}")
    summary_lines.append(f"Resolution: {IMAGE_RESOLUTION}x{IMAGE_RESOLUTION} (baseline was 224x224)")
    summary_lines.append(f"JPEG quality: 95 (baseline was 90)")
    summary_lines.append(f"Contrast enhancement: 1.4x")
    class_counts = {}
    for s in sample:
        class_counts[s["dx"]] = class_counts.get(s["dx"], 0) + 1
    class_desc = ", ".join(f"{dx.upper()}={n}" for dx, n in class_counts.items())
    summary_lines.append(f"Images: {len(sample)} ({class_desc})")
    summary_lines.append(f"Attributes (CoT): {', '.join(COT_QUERIES.keys())}")
    summary_lines.append("=" * 70)

    # Model status & timing table
    summary_lines.append("\n## Model Status & Timing")
    summary_lines.append(f"{'Model':<25} {'Ollama Tag':<25} {'Status':<10} {'Time'}")
    summary_lines.append("-" * 75)
    for cfg in MODEL_REGISTRY:
        name = cfg["name"]
        tag = cfg["ollama_model"]
        status = model_status.get(name, "NOT RUN")
        timing = model_timings.get(name)
        time_str = f"{timing:.1f}s" if timing else "—"
        summary_lines.append(f"{name:<25} {tag:<25} {status:<10} {time_str}")

    # Per-model distribution of each attribute
    df_results = pd.DataFrame(all_results)
    active_models = list(df_results["model"].unique())

    for model_name in active_models:
        mdf = df_results[df_results["model"] == model_name]
        summary_lines.append(f"\n## {model_name} — CoT Label Distributions")
        summary_lines.append(f"  (n={len(mdf)} images)")

        for attr_key in COT_QUERIES:
            col = f"{attr_key}_label"
            if col in mdf.columns:
                dist = mdf[col].value_counts()
                valid = COT_QUERIES[attr_key]["labels"]
                summary_lines.append(f"\n  {attr_key} (valid: {', '.join(valid)}):")
                for val, count in dist.items():
                    pct = 100.0 * count / len(mdf)
                    marker = " " if val in valid else " ?"
                    summary_lines.append(f"   {marker}{val:<25} {count:>3} ({pct:5.1f}%)")

        # Stickiness check
        summary_lines.append(f"\n  Stickiness Check:")
        for attr_key in COT_QUERIES:
            col = f"{attr_key}_label"
            if col in mdf.columns:
                unique_vals = mdf[col].nunique()
                is_sticky = unique_vals <= 1
                summary_lines.append(
                    f"    {attr_key}: {unique_vals} unique value(s)"
                    f"{' ** STICKY' if is_sticky else ' - varied'}"
                )

    # Cross-model agreement
    if len(active_models) >= 2:
        summary_lines.append(f"\n## Cross-Model Agreement (CoT @ {IMAGE_RESOLUTION}px)")
        for attr_key in COT_QUERIES:
            col = f"{attr_key}_label"
            if col not in df_results.columns:
                continue
            summary_lines.append(f"\n  {attr_key}:")
            image_ids = df_results["image_id"].unique()
            agree_count = 0
            total_count = 0
            for img_id in image_ids:
                vals = df_results[df_results["image_id"] == img_id][col].dropna().tolist()
                vals = [v for v in vals if v not in ("ERROR", "IMAGE_MISSING", "unclear")]
                if len(vals) >= 2:
                    total_count += 1
                    if len(set(vals)) == 1:
                        agree_count += 1
            if total_count > 0:
                pct = 100.0 * agree_count / total_count
                summary_lines.append(
                    f"    Agreement: {agree_count}/{total_count} images ({pct:.1f}%)"
                )
            else:
                summary_lines.append("    Not enough data for comparison")

    # CoT quality: % of responses with explicit LABEL: line
    summary_lines.append(f"\n## CoT Quality — Explicit LABEL: Rate")
    for model_name in active_models:
        mdf = df_results[df_results["model"] == model_name]
        total_queries = 0
        explicit_labels = 0
        for attr_key in COT_QUERIES:
            raw_col = f"{attr_key}_raw"
            if raw_col not in mdf.columns:
                continue
            for raw_val in mdf[raw_col]:
                if raw_val in ("IMAGE_MISSING", "ERROR") or str(raw_val).startswith("PREP_ERROR"):
                    continue
                total_queries += 1
                if re.search(r"LABEL:", str(raw_val), re.IGNORECASE):
                    explicit_labels += 1
        if total_queries > 0:
            pct = 100.0 * explicit_labels / total_queries
            summary_lines.append(
                f"  {model_name}: {explicit_labels}/{total_queries} ({pct:.1f}%)"
            )

    # Resolution comparison note
    summary_lines.append(f"\n## Resolution Test Notes")
    summary_lines.append(f"  This test uses {IMAGE_RESOLUTION}x{IMAGE_RESOLUTION} input images")
    summary_lines.append(f"  (5x the baseline 224x224 resolution = 25x more pixels).")
    summary_lines.append(f"  Compare results with ollama_cot_benchmark_summary_round3.txt")
    summary_lines.append(f"  to evaluate whether higher resolution improves attribute extraction.")
    summary_lines.append(f"")
    summary_lines.append(f"  Native vision encoder resolutions:")
    summary_lines.append(f"    Llama 3.2 Vision:  560x560 per tile (up to 1120x1120 with 4 tiles)")
    summary_lines.append(f"    Qwen2.5-VL:        Dynamic resolution (up to ~2048px)")
    summary_lines.append(f"    LLaVA (13B):       CLIP ViT-L/14 at 336x336")

    summary_text = "\n".join(summary_lines) + "\n"
    with open(SUMMARY_TXT, "w") as f:
        f.write(summary_text)
    print(f"Summary saved: {SUMMARY_TXT}")
    print("\n" + summary_text)


if __name__ == "__main__":
    run_benchmark()
