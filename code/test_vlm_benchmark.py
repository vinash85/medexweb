#!/usr/bin/env python3
"""
Multi-VLM Dermoscopy Attribute Benchmark
=========================================
Benchmarks vision-language models on 8 dermoscopy attributes across a
stratified sample from all 6 lesion classes (MEL, NV, BCC, BKL, AKIEC, DF).
Up to 5 images per class (fewer when the dataset has fewer).

Run inside Docker:
    python3 /home/project/code/test_vlm_benchmark.py
"""

import os
import re
import gc
import sys
import csv
import time
import uuid
import base64
import hashlib
import traceback
from datetime import datetime

import pandas as pd
from PIL import Image, ImageEnhance

# ── Paths (Docker layout) ────────────────────────────────────────────────────
PROJECT_ROOT = "/home/project"
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
CSV_PATH = os.path.join(PROJECT_ROOT, "code", "DermsGemms.csv")
RESULTS_CSV = os.path.join(DATA_DIR, "vlm_benchmark_results.csv")
SUMMARY_TXT = os.path.join(DATA_DIR, "vlm_benchmark_summary.txt")

# ── Model Registry ───────────────────────────────────────────────────────────
MODEL_REGISTRY = [
    {
        "name": "MedGemma",
        "model_path": os.path.join(MODEL_DIR, "medgemma.gguf"),
        "mmproj_path": os.path.join(MODEL_DIR, "medgemma-mmproj.gguf"),
        "handler_type": "medgemma",
        "n_ctx": 2048,
        "temperature": 0.7,
        "max_tokens": 32,
        "repeat_penalty": 1.5,
    },
    {
        "name": "PaliGemma",
        "model_path": os.path.join(MODEL_DIR, "paligemma-3b-mix-224-gguf",
                                   "paligemma-3b-mix-224-text-model-q8_0.gguf"),
        "mmproj_path": os.path.join(MODEL_DIR, "paligemma-3b-mix-224-gguf",
                                    "paligemma-3b-mix-224-mmproj-f16.gguf"),
        "handler_type": "paligemma",
        "n_ctx": 2048,
        "temperature": 0.7,
        "max_tokens": 32,
        "repeat_penalty": 1.5,
    },
    {
        "name": "Llama-3.2-Vision",
        "model_path": os.path.join(MODEL_DIR, "Llama-3.2-11B-Vision-Instruct-GGUF",
                                   "Llama-3.2-11B-Vision-Instruct.Q8_0.gguf"),
        "mmproj_path": os.path.join(MODEL_DIR, "Llama-3.2-11B-Vision-Instruct-GGUF",
                                    "Llama-3.2-11B-Vision-Instruct-mmproj.f16.gguf"),
        "handler_type": "llama3vision",
        "n_ctx": 2048,
        "temperature": 0.7,
        "max_tokens": 32,
        "repeat_penalty": 1.5,
    },
    {
        "name": "Moondream2",
        "model_path": os.path.join(MODEL_DIR, "moondream2-20250414-GGUF",
                                   "moondream2-text-model-f16_ct-vicuna.gguf"),
        "mmproj_path": os.path.join(MODEL_DIR, "moondream2-20250414-GGUF",
                                    "moondream2-mmproj-f16-20250414.gguf"),
        "handler_type": "moondream",
        "n_ctx": 2048,
        "temperature": 0.7,
        "max_tokens": 32,
        "repeat_penalty": 1.5,
    },
    {
        "name": "InternVL2",
        "model_path": os.path.join(MODEL_DIR, "InternVL2_5-1B-GGUF",
                                   "InternVL2_5-1B-Q8_0.gguf"),
        "mmproj_path": os.path.join(MODEL_DIR, "InternVL2_5-1B-GGUF",
                                    "mmproj-InternVL2_5-1B-Q8_0.gguf"),
        "handler_type": "internvl",
        "n_ctx": 2048,
        "temperature": 0.7,
        "max_tokens": 32,
        "repeat_penalty": 1.5,
    },
]

# ── Attribute Queries ─────────────────────────────────────────────────────────
ATTRIBUTE_QUERIES = {
    "Pigment_Network": (
        "Does the lesion contain a net-like grid of dark lines (pigment network)? "
        "If yes, is the grid uniform/regular or irregular/atypical? "
        "Answer with one of: absent, present_uniform, present_irregular, present_unspecified."
    ),
    "Dots_Globules": (
        "Identify solid circular spots (dots or globules) in the lesion. "
        "If present, are they mostly central or peripheral? "
        "Answer with one of: absent, present_central, present_peripheral, present_unspecified."
    ),
    "Streaks": (
        "Look at the lesion border. Are there radial lines or pseudopods (streaks) "
        "extending outward from the edge? "
        "Answer with one of: absent, present."
    ),
    "Milia_Cysts": (
        "Are there tiny, bright white or yellow circular 'beads' (milia-like cysts) "
        "or dark comedo-like openings visible in this lesion? "
        "Answer with one of: absent, milia_only, comedo_only, milia+comedo."
    ),
    "Vascular_Structures": (
        "Examine the lesion for blood vessel patterns. Are there arborizing (tree-like branching) "
        "vessels, dotted/glomerular vessels, or other prominent vascular structures? "
        "Answer with one of: absent, arborizing, dotted_glomerular, other_vascular."
    ),
    "Blue_White_Structures": (
        "Does the lesion contain blue-white veil (structureless blue-white area), "
        "shiny white lines/blotches/strands, or white clods? "
        "Answer with one of: absent, blue_white_veil, shiny_white, white_clods, present_unspecified."
    ),
    "Scale_Surface": (
        "Does the lesion surface show scale, crust, or textured surface features "
        "such as gyri/ridges (brain-like pattern) or a moth-eaten border? "
        "Answer with one of: absent, scale_crust, gyri_ridges, moth_eaten, present_unspecified."
    ),
    "Blotches": (
        "Are there dark structureless areas (blotches) within the lesion? "
        "If present, are they symmetric/regular or asymmetric/irregular? "
        "Answer with one of: absent, symmetric, asymmetric, present_unspecified."
    ),
}

# ── Stratified Sample (up to 5 per class, all 6 lesion types) ────────────────
SAMPLE_CLASSES = ["mel", "nv", "bcc", "bkl", "akiec", "df"]
N_PER_CLASS = 5


def select_sample_images():
    """Select up to N_PER_CLASS images per class from DermsGemms.csv.

    Takes all available images for classes with fewer than N_PER_CLASS samples.
    """
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


# ── Image Preparation ─────────────────────────────────────────────────────────

def prepare_image(filepath):
    """Resize to 224x224, enhance contrast, save to temp file."""
    temp_path = os.path.join(DATA_DIR, f"vlm_bench_{uuid.uuid4().hex[:6]}.jpg")
    with Image.open(filepath) as img:
        img = img.convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
        img = ImageEnhance.Contrast(img).enhance(1.4)
        img.save(temp_path, quality=90)
    return temp_path


def encode_image_base64(image_path):
    """Read image and return a data:image/jpeg;base64 URI."""
    with open(image_path, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# ── Model Loading ─────────────────────────────────────────────────────────────

def load_model(config):
    """Factory: create handler + Llama instance based on handler_type."""
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Llava15ChatHandler

    handler_type = config["handler_type"]
    model_path = config["model_path"]
    mmproj_path = config["mmproj_path"]
    n_ctx = config["n_ctx"]

    handler = None
    chat_format = None

    if handler_type == "medgemma":
        # Import the project's custom handler
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))
        from pipeline import MedGemmaChatHandler
        handler = MedGemmaChatHandler(clip_model_path=mmproj_path, verbose=False)

    elif handler_type == "paligemma":
        # Try PaliGemmaChatHandler, fallback to Llava15ChatHandler
        try:
            from llama_cpp.llama_chat_format import PaliGemmaChatHandler
            handler = PaliGemmaChatHandler(clip_model_path=mmproj_path, verbose=False)
        except (ImportError, AttributeError):
            print(f"  [INFO] PaliGemmaChatHandler not available, using Llava15ChatHandler")
            handler = Llava15ChatHandler(clip_model_path=mmproj_path, verbose=False)

    elif handler_type == "llama3vision":
        # Uses built-in chat_format string, no custom handler
        chat_format = "llama-3-vision"
        # Still need the mmproj loaded via a handler for image embedding
        handler = Llava15ChatHandler(clip_model_path=mmproj_path, verbose=False)

    elif handler_type == "moondream":
        try:
            from llama_cpp.llama_chat_format import MoondreamChatHandler
            handler = MoondreamChatHandler(clip_model_path=mmproj_path, verbose=False)
        except (ImportError, AttributeError):
            print(f"  [INFO] MoondreamChatHandler not available, using Llava15ChatHandler")
            handler = Llava15ChatHandler(clip_model_path=mmproj_path, verbose=False)

    elif handler_type == "internvl":
        handler = Llava15ChatHandler(clip_model_path=mmproj_path, verbose=False)

    else:
        raise ValueError(f"Unknown handler_type: {handler_type}")

    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=-1,
        chat_handler=handler,
        verbose=False,
    )
    return llm, handler


# ── Response Cleaning ─────────────────────────────────────────────────────────

def clean_attribute_response(raw, attr_key):
    """Extract categorical value from raw VLM output via keyword matching."""
    text = raw.lower().strip()
    text = text.replace("*", "").replace('"', "").replace("'", "")
    text = text.split("\n")[0].strip()
    text = re.sub(r"\s+", " ", text)

    if attr_key == "Pigment_Network":
        if "irregular" in text or "atypical" in text:
            return "present_irregular"
        if "uniform" in text or "regular" in text or "typical" in text:
            return "present_uniform"
        if "present" in text or "yes" in text or "net" in text or "grid" in text:
            return "present_unspecified"
        if "absent" in text or "no " in text or "none" in text or "not " in text:
            return "absent"
        return "unclear"

    elif attr_key == "Dots_Globules":
        has_central = "central" in text or "center" in text
        has_peripheral = "peripher" in text or "border" in text or "edge" in text
        if has_central and not has_peripheral:
            return "present_central"
        if has_peripheral and not has_central:
            return "present_peripheral"
        if "present" in text or "yes" in text or "dot" in text or "globul" in text:
            return "present_unspecified"
        if "absent" in text or "no " in text or "none" in text or "not " in text:
            return "absent"
        return "unclear"

    elif attr_key == "Streaks":
        if "present" in text or "yes" in text or "streak" in text or "pseudopod" in text or "radial" in text:
            return "present"
        if "absent" in text or "no " in text or "none" in text or "not " in text:
            return "absent"
        return "unclear"

    elif attr_key == "Milia_Cysts":
        has_milia = "milia" in text or "bead" in text or "white" in text or "bright" in text
        has_comedo = "comedo" in text or "opening" in text or "dark" in text
        if has_milia and has_comedo:
            return "milia+comedo"
        if has_milia:
            return "milia_only"
        if has_comedo:
            return "comedo_only"
        if "absent" in text or "no " in text or "none" in text or "not " in text:
            return "absent"
        return "unclear"

    elif attr_key == "Vascular_Structures":
        if "arboriz" in text or "branch" in text or "tree" in text:
            return "arborizing"
        if "dotted" in text or "glomerul" in text or "punctate" in text:
            return "dotted_glomerular"
        if "vessel" in text or "vascular" in text or "capillar" in text or "telangi" in text:
            return "other_vascular"
        if "absent" in text or "no " in text or "none" in text or "not " in text:
            return "absent"
        return "unclear"

    elif attr_key == "Blue_White_Structures":
        if "blue" in text and ("white" in text or "veil" in text):
            return "blue_white_veil"
        if "shiny" in text or ("white" in text and ("line" in text or "strand" in text or "blotch" in text)):
            return "shiny_white"
        if "white" in text and "clod" in text:
            return "white_clods"
        if "present" in text or "yes" in text or "white" in text or "blue" in text:
            return "present_unspecified"
        if "absent" in text or "no " in text or "none" in text or "not " in text:
            return "absent"
        return "unclear"

    elif attr_key == "Scale_Surface":
        if "scale" in text or "crust" in text or "kerat" in text:
            return "scale_crust"
        if "gyri" in text or "ridge" in text or "brain" in text or "cerebriform" in text:
            return "gyri_ridges"
        if "moth" in text or "eaten" in text:
            return "moth_eaten"
        if "present" in text or "yes" in text or "rough" in text or "texture" in text:
            return "present_unspecified"
        if "absent" in text or "no " in text or "none" in text or "not " in text or "smooth" in text:
            return "absent"
        return "unclear"

    elif attr_key == "Blotches":
        if "asymmetri" in text or "irregular" in text or "atypical" in text:
            return "asymmetric"
        if "symmetri" in text or "regular" in text or "uniform" in text:
            return "symmetric"
        if "present" in text or "yes" in text or "blotch" in text or "structureless" in text:
            return "present_unspecified"
        if "absent" in text or "no " in text or "none" in text or "not " in text:
            return "absent"
        return "unclear"

    return "unclear"


# ── Benchmark Runner ──────────────────────────────────────────────────────────

def run_benchmark():
    print("=" * 70)
    print("Multi-VLM Dermoscopy Attribute Benchmark")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

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
    model_status = {}  # name -> "OK" | "SKIPPED" | "ERROR: ..."
    temp_files = []

    try:
        for cfg in MODEL_REGISTRY:
            model_name = cfg["name"]
            print(f"\n{'─' * 60}")
            print(f"Model: {model_name}")
            print(f"{'─' * 60}")

            # Check model files exist
            if not os.path.exists(cfg["model_path"]):
                msg = f"Model file not found: {cfg['model_path']}"
                print(f"  SKIP: {msg}")
                model_status[model_name] = f"SKIPPED — {msg}"
                continue
            if not os.path.exists(cfg["mmproj_path"]):
                msg = f"mmproj file not found: {cfg['mmproj_path']}"
                print(f"  SKIP: {msg}")
                model_status[model_name] = f"SKIPPED — {msg}"
                continue

            # Load model
            try:
                print(f"  Loading {model_name}...")
                t0 = time.time()
                llm, handler = load_model(cfg)
                load_time = time.time() - t0
                print(f"  Loaded in {load_time:.1f}s")
                model_status[model_name] = "OK"
            except Exception as e:
                msg = f"Load failed: {e}"
                print(f"  ERROR: {msg}")
                traceback.print_exc()
                model_status[model_name] = f"ERROR — {msg}"
                continue

            # Run queries on each image
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
                    }
                    for attr_key in ATTRIBUTE_QUERIES:
                        row[f"{attr_key}_raw"] = "IMAGE_MISSING"
                        row[f"{attr_key}_clean"] = "IMAGE_MISSING"
                    all_results.append(row)
                    continue

                # Prepare image
                try:
                    proc_path = prepare_image(img_path)
                    temp_files.append(proc_path)
                    image_uri = encode_image_base64(proc_path)
                except Exception as e:
                    print(f"  [{img_idx+1}/{len(sample)}] {image_id} — image prep failed: {e}")
                    row = {
                        "model": model_name,
                        "image_id": image_id,
                        "dx": s["dx"],
                        "gt_attributes": s["gt_attributes"],
                    }
                    for attr_key in ATTRIBUTE_QUERIES:
                        row[f"{attr_key}_raw"] = f"PREP_ERROR: {e}"
                        row[f"{attr_key}_clean"] = "ERROR"
                    all_results.append(row)
                    continue

                row = {
                    "model": model_name,
                    "image_id": image_id,
                    "dx": s["dx"],
                    "gt_attributes": s["gt_attributes"],
                }

                for attr_key, query_text in ATTRIBUTE_QUERIES.items():
                    try:
                        # Reset KV cache by resetting the model state
                        llm.reset()

                        res = llm.create_chat_completion(
                            messages=[{
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": image_uri}},
                                    {"type": "text", "text": query_text},
                                ]
                            }],
                            max_tokens=cfg["max_tokens"],
                            temperature=cfg["temperature"],
                            repeat_penalty=cfg["repeat_penalty"],
                        )
                        raw_answer = res["choices"][0]["message"]["content"].strip()
                        cleaned = clean_attribute_response(raw_answer, attr_key)
                        row[f"{attr_key}_raw"] = raw_answer
                        row[f"{attr_key}_clean"] = cleaned

                    except Exception as e:
                        print(f"    ERROR on {image_id}/{attr_key}: {e}")
                        row[f"{attr_key}_raw"] = f"ERROR: {e}"
                        row[f"{attr_key}_clean"] = "ERROR"

                all_results.append(row)
                attr_summary = " ".join(
                    f"{k[:2]}={row.get(f'{k}_clean', '?')}"
                    for k in ATTRIBUTE_QUERIES
                )
                print(f"  [{img_idx+1}/{len(sample)}] {image_id}: {attr_summary}")

            # Unload model
            print(f"  Unloading {model_name}...")
            del llm
            del handler
            gc.collect()
            time.sleep(2)
            print(f"  Unloaded.")

    except KeyboardInterrupt:
        print("\n\nKeyboardInterrupt — saving partial results...")

    # ── Save Results ──────────────────────────────────────────────────────────

    # Clean up temp files
    for tf in temp_files:
        try:
            if os.path.exists(tf):
                os.remove(tf)
        except OSError:
            pass

    if not all_results:
        print("\nNo results to save.")
        return

    # Build CSV
    fieldnames = ["model", "image_id", "dx", "gt_attributes"]
    for attr_key in ATTRIBUTE_QUERIES:
        fieldnames.extend([f"{attr_key}_raw", f"{attr_key}_clean"])

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
    summary_lines.append("Multi-VLM Dermoscopy Attribute Benchmark — Summary")
    summary_lines.append(f"Generated: {datetime.now().isoformat()}")
    class_counts = {}
    for s in sample:
        class_counts[s["dx"]] = class_counts.get(s["dx"], 0) + 1
    class_desc = ", ".join(f"{dx.upper()}={n}" for dx, n in class_counts.items())
    summary_lines.append(f"Images: {len(sample)} ({class_desc})")
    summary_lines.append(f"Attributes: {', '.join(ATTRIBUTE_QUERIES.keys())}")
    summary_lines.append("=" * 70)

    # Model status table
    summary_lines.append("\n## Model Status")
    summary_lines.append(f"{'Model':<20} {'Status'}")
    summary_lines.append("-" * 50)
    for cfg in MODEL_REGISTRY:
        name = cfg["name"]
        status = model_status.get(name, "NOT RUN")
        summary_lines.append(f"{name:<20} {status}")

    # Per-model distribution of each attribute
    df_results = pd.DataFrame(all_results)
    active_models = [m for m in df_results["model"].unique()]

    for model_name in active_models:
        mdf = df_results[df_results["model"] == model_name]
        summary_lines.append(f"\n## {model_name} — Attribute Distributions")
        summary_lines.append(f"  (n={len(mdf)} images)")

        for attr_key in ATTRIBUTE_QUERIES:
            col = f"{attr_key}_clean"
            if col in mdf.columns:
                dist = mdf[col].value_counts()
                summary_lines.append(f"\n  {attr_key}:")
                for val, count in dist.items():
                    pct = 100.0 * count / len(mdf)
                    summary_lines.append(f"    {val:<25} {count:>3} ({pct:5.1f}%)")

        # Stickiness check: does the model give the same answer for every image?
        summary_lines.append(f"\n  Stickiness Check:")
        for attr_key in ATTRIBUTE_QUERIES:
            col = f"{attr_key}_clean"
            if col in mdf.columns:
                unique_vals = mdf[col].nunique()
                is_sticky = unique_vals <= 1
                summary_lines.append(
                    f"    {attr_key}: {unique_vals} unique value(s)"
                    f"{' ⚠ STICKY — same answer for all images' if is_sticky else ' ✓ varied'}"
                )

    # Cross-model agreement (only if 2+ models ran)
    if len(active_models) >= 2:
        summary_lines.append(f"\n## Cross-Model Agreement")
        for attr_key in ATTRIBUTE_QUERIES:
            col = f"{attr_key}_clean"
            if col not in df_results.columns:
                continue
            summary_lines.append(f"\n  {attr_key}:")
            # For each image, check if all models agree
            image_ids = df_results["image_id"].unique()
            agree_count = 0
            total_count = 0
            for img_id in image_ids:
                vals = df_results[df_results["image_id"] == img_id][col].dropna().tolist()
                vals = [v for v in vals if v not in ("ERROR", "IMAGE_MISSING")]
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

    summary_text = "\n".join(summary_lines) + "\n"
    with open(SUMMARY_TXT, "w") as f:
        f.write(summary_text)
    print(f"Summary saved: {SUMMARY_TXT}")
    print("\n" + summary_text)


if __name__ == "__main__":
    run_benchmark()
