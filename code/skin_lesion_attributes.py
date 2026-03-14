#!/usr/bin/env python3
"""
Skin Lesion — Visual Attribute Extraction with MedGemma
=======================================================
For each dermoscopic image, asks MedGemma to describe visual attributes
WITHOUT revealing the diagnosis class. The Dx class is read from the
annotation CSV (code/DermsGemms.csv) and used only for grouping output.

Images: data/images/ (flat folder, ISIC_*.jpg)
Annotations: code/DermsGemms.csv (Image, Dx, Lesion attributes)

Output: one JSON per Dx class in data/skin_lesion_attributes/

Run inside Docker:
    python3 /home/project/code/skin_lesion_attributes.py
"""

import os
import sys
import gc
import json
import uuid
import time
import base64
import argparse
import traceback
from datetime import datetime

import pandas as pd
from PIL import Image, ImageEnhance
from tqdm import tqdm

# ── Paths (Docker layout) ────────────────────────────────────────────────────
PROJECT_ROOT = "/home/project"
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "images")
CSV_PATH = os.path.join(PROJECT_ROOT, "code", "DermsGemms.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "skin_lesion_attributes")
OUTPUT_TXT = os.path.join(OUTPUT_DIR, "summary.txt")

# ── MedGemma Model Paths ─────────────────────────────────────────────────────
MEDGEMMA_MODEL = os.path.join(MODEL_DIR, "medgemma.gguf")
MEDGEMMA_MMPROJ = os.path.join(MODEL_DIR, "medgemma-mmproj.gguf")

# ── Dx class labels ──────────────────────────────────────────────────────────
DX_LABELS = {
    "nv": "Melanocytic Nevus",
    "mel": "Melanoma",
    "bkl": "Benign Keratosis-like Lesion",
    "bcc": "Basal Cell Carcinoma",
    "akiec": "Actinic Keratosis / Intraepithelial Carcinoma",
    "df": "Dermatofibroma",
}

# ── Prompt Template ───────────────────────────────────────────────────────────
ATTRIBUTE_PROMPT = (
    "This is a dermoscopic image of a skin lesion. "
    "The confirmed diagnosis is: {diagnosis}.\n\n"
    "Describe the visual attributes in this image that are consistent with "
    "and support this diagnosis. Focus on:\n"
    "- Pigment network (regular, irregular, atypical, fading, reticular)\n"
    "- Globules and dots (distribution, color, regularity)\n"
    "- Streaks and pseudopods (presence, symmetry)\n"
    "- Blotches (color, distribution, symmetry)\n"
    "- Vascular structures (dotted vessels, linear vessels, arborizing, hairpin)\n"
    "- Structureless areas (color, homogeneity)\n"
    "- Blue-white structures (veil, shiny white lines/streaks)\n"
    "- Regression structures (white scar-like areas, peppering)\n"
    "- Border characteristics (sharp, fading, regular, irregular)\n"
    "- Overall symmetry and color distribution\n\n"
    "IMPORTANT: Only describe attributes you can clearly identify in this image. "
    "Do NOT mention the diagnosis name or classify the lesion in your description. "
    "If you are uncertain about any attribute, do NOT mention it. "
    "Be concise and factual. Do not speculate."
)


def prepare_image(filepath):
    """Resize to 224x224, enhance contrast, save to temp file."""
    temp_path = os.path.join(
        os.path.dirname(filepath),
        f"skin_proc_{uuid.uuid4().hex[:6]}.jpg",
    )
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


def load_medgemma():
    """Load MedGemma with custom chat handler."""
    from llama_cpp import Llama

    sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))
    from pipeline import MedGemmaChatHandler

    print("[INFO] Loading MedGemma vision model...")
    t0 = time.time()
    handler = MedGemmaChatHandler(clip_model_path=MEDGEMMA_MMPROJ, verbose=False)
    llm = Llama(
        model_path=MEDGEMMA_MODEL,
        n_ctx=2048,
        n_gpu_layers=-1,
        chat_handler=handler,
        verbose=False,
    )
    print(f"[INFO] MedGemma loaded in {time.time() - t0:.1f}s")
    return llm


def clean_response(text):
    """Strip special tokens and clean up model output."""
    special_tokens = [
        "<start_of_turn>", "<end_of_turn>", "<eos>", "<pad>", "\u2581",
    ]
    for token in special_tokens:
        text = text.replace(token, "")
    lines = text.strip().split("\n")
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        alpha = sum(1 for c in line if c.isalpha())
        if len(line) > 5 and alpha < len(line) * 0.4:
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def load_annotations():
    """Load DermsGemms.csv and return DataFrame with image->dx mapping."""
    df = pd.read_csv(CSV_PATH)
    # Strip whitespace from column names and values
    df.columns = [c.strip() for c in df.columns]
    df["Image"] = df["Image"].str.strip()
    df["Dx"] = df["Dx"].str.strip()
    # Drop duplicates (e.g. ISIC_0027683 appears twice)
    df = df.drop_duplicates(subset="Image", keep="first")
    return df


def collect_images(annotations, limit_per_class=None):
    """Match annotation rows to image files in IMAGE_DIR.

    Args:
        annotations: DataFrame with Image and Dx columns.
        limit_per_class: If set, cap images per Dx class to this number.
    """
    images = []
    class_counts = {dx: 0 for dx in DX_LABELS}

    for _, row in annotations.iterrows():
        image_id = row["Image"]
        dx = row["Dx"]
        if dx not in DX_LABELS:
            continue

        # Find the image file
        fname = f"{image_id}.jpg"
        filepath = os.path.join(IMAGE_DIR, fname)
        if not os.path.exists(filepath):
            continue

        if limit_per_class and class_counts[dx] >= limit_per_class:
            continue

        images.append({
            "filepath": filepath,
            "filename": fname,
            "image_id": image_id,
            "dx": dx,
            "dx_label": DX_LABELS[dx],
            "lesion_attributes": row.get("Lesion attributes", ""),
        })
        class_counts[dx] += 1

    return images


def load_existing_results():
    """Load previously saved JSON results for resume support."""
    existing = {}
    for dx in DX_LABELS:
        json_path = os.path.join(OUTPUT_DIR, f"skin_lesion_attributes_{dx}.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                existing[dx] = data.get("images", {})
            except (json.JSONDecodeError, KeyError):
                existing[dx] = {}
        else:
            existing[dx] = {}
    return existing


def run_extraction(limit_per_class=None):
    print("=" * 70)
    print("Skin Lesion — Visual Attribute Extraction (MedGemma)")
    print(f"Started: {datetime.now().isoformat()}")
    if limit_per_class:
        print(f"Limit: {limit_per_class} images per Dx class")
    print("=" * 70)

    # Load annotations
    annotations = load_annotations()
    print(f"\nAnnotations loaded: {len(annotations)} entries")

    # Collect images
    images = collect_images(annotations, limit_per_class=limit_per_class)
    print(f"Total images matched: {len(images)}")
    for dx, label in DX_LABELS.items():
        count = sum(1 for img in images if img["dx"] == dx)
        if count > 0:
            print(f"  {dx} ({label}): {count}")

    if not images:
        print("No images found. Exiting.")
        return

    # Check model files
    if not os.path.exists(MEDGEMMA_MODEL):
        print(f"ERROR: Model not found: {MEDGEMMA_MODEL}")
        return
    if not os.path.exists(MEDGEMMA_MMPROJ):
        print(f"ERROR: mmproj not found: {MEDGEMMA_MMPROJ}")
        return

    # Load existing results for resume
    all_existing = load_existing_results()
    target_filenames = {(img["dx"], img["filename"]) for img in images}
    results_by_class = {dx: {} for dx in DX_LABELS}
    for dx in DX_LABELS:
        for fname, result in all_existing.get(dx, {}).items():
            if (dx, fname) in target_filenames:
                results_by_class[dx][fname] = result
    already_done = sum(
        1 for dx in results_by_class
        for r in results_by_class[dx].values()
        if r.get("status") == "ok"
    )
    if already_done > 0:
        print(f"\n[RESUME] Found {already_done} previously completed images — skipping them.")

    # Filter to only images that need processing
    to_process = [
        img for img in images
        if img["filename"] not in results_by_class.get(img["dx"], {})
        or results_by_class[img["dx"]][img["filename"]].get("status") != "ok"
    ]
    print(f"Images to process: {len(to_process)}")

    if not to_process:
        print("All images already processed. Nothing to do.")
        return

    # Load model
    llm = load_medgemma()

    counters = {"success": 0, "error": 0, "skipped": already_done}
    total = len(images)
    image_times = []

    pbar = tqdm(
        to_process,
        initial=already_done,
        total=total,
        desc="Processing",
        unit="img",
        dynamic_ncols=True,
    )

    try:
        for img_info in pbar:
            filepath = img_info["filepath"]
            filename = img_info["filename"]
            dx = img_info["dx"]

            pbar.set_postfix_str(f"{dx}/{filename[:20]}")
            t_start = time.time()

            proc_path = None
            try:
                proc_path = prepare_image(filepath)
                image_uri = encode_image_base64(proc_path)

                # Query MedGemma with diagnosis
                prompt_text = ATTRIBUTE_PROMPT.format(diagnosis=img_info["dx_label"])
                llm.reset()
                res = llm.create_chat_completion(
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_uri}},
                            {"type": "text", "text": prompt_text},
                        ],
                    }],
                    max_tokens=256,
                    temperature=0.5,
                    repeat_penalty=1.5,
                )
                raw_response = res["choices"][0]["message"]["content"].strip()
                cleaned = clean_response(raw_response)

                elapsed = time.time() - t_start
                image_times.append(elapsed)

                results_by_class[dx][filename] = {
                    "dx": dx,
                    "dx_label": img_info["dx_label"],
                    "gt_lesion_attributes": img_info["lesion_attributes"],
                    "attributes": cleaned,
                    "raw_response": raw_response,
                    "status": "ok",
                    "time_seconds": round(elapsed, 1),
                }
                counters["success"] += 1

                avg_t = sum(image_times) / len(image_times)
                remaining = len(to_process) - (counters["success"] + counters["error"])
                eta_min = (avg_t * remaining) / 60
                pbar.set_postfix_str(
                    f"{elapsed:.1f}s | avg {avg_t:.1f}s/img | ETA {eta_min:.0f}min"
                )

            except Exception as e:
                elapsed = time.time() - t_start
                tqdm.write(f"  ERROR ({dx}/{filename}): {e}")
                traceback.print_exc()
                results_by_class[dx][filename] = {
                    "dx": dx,
                    "dx_label": img_info["dx_label"],
                    "gt_lesion_attributes": img_info["lesion_attributes"],
                    "attributes": "",
                    "raw_response": f"ERROR: {e}",
                    "status": "error",
                    "time_seconds": round(elapsed, 1),
                }
                counters["error"] += 1

            finally:
                if proc_path and os.path.exists(proc_path):
                    os.remove(proc_path)

    except KeyboardInterrupt:
        print("\n\nInterrupted — saving partial results...")
    finally:
        pbar.close()

    if image_times:
        avg_t = sum(image_times) / len(image_times)
        print(f"\n[TIMING] {len(image_times)} images | avg {avg_t:.1f}s/img | total {sum(image_times)/60:.1f}min")

    # Unload model
    print("\n[INFO] Unloading MedGemma...")
    del llm
    gc.collect()

    # ── Save one JSON per Dx class ────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for dx, file_results in results_by_class.items():
        if not file_results:
            continue
        json_path = os.path.join(OUTPUT_DIR, f"skin_lesion_attributes_{dx}.json")
        output = {
            "metadata": {
                "model": "MedGemma",
                "timestamp": datetime.now().isoformat(),
                "dx": dx,
                "dx_label": DX_LABELS[dx],
                "total_images": len(file_results),
                "successful": sum(1 for r in file_results.values() if r["status"] == "ok"),
                "errors": sum(1 for r in file_results.values() if r["status"] == "error"),
            },
            "images": file_results,
        }
        with open(json_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Saved: {json_path} ({len(file_results)} images)")

    # ── Save text summary ─────────────────────────────────────────────────────
    total_ok = sum(1 for dx in results_by_class for r in results_by_class[dx].values() if r.get("status") == "ok")
    total_err = sum(1 for dx in results_by_class for r in results_by_class[dx].values() if r.get("status") == "error")
    summary_lines = [
        "=" * 70,
        "Skin Lesion — Visual Attribute Extraction Summary",
        f"Generated: {datetime.now().isoformat()}",
        f"Model: MedGemma",
        f"Total images: {total}  |  Success: {total_ok}  |  Errors: {total_err}",
        f"  (this run: {counters['success']} new, {counters['skipped']} resumed)",
        "=" * 70,
    ]

    for dx, label in DX_LABELS.items():
        file_results = results_by_class[dx]
        ok_results = {f: r for f, r in file_results.items() if r["status"] == "ok"}
        if not ok_results:
            continue
        summary_lines.append(f"\n{'─' * 60}")
        summary_lines.append(f"  {dx} — {label} ({len(ok_results)} images)")
        summary_lines.append(f"{'─' * 60}")
        for fname, r in ok_results.items():
            summary_lines.append(f"\n  [{fname}]  GT: {r.get('gt_lesion_attributes', '')}")
            for line in r["attributes"].split("\n"):
                summary_lines.append(f"    {line}")

    summary_text = "\n".join(summary_lines) + "\n"
    with open(OUTPUT_TXT, "w") as f:
        f.write(summary_text)
    print(f"Summary saved: {OUTPUT_TXT}")
    print(f"\nDone. {total_ok}/{total} images completed ({counters['success']} new this run, {counters['skipped']} resumed).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skin Lesion Attribute Extraction with MedGemma")
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max images per Dx class (default: 0 = all)",
    )
    args = parser.parse_args()
    limit = args.limit if args.limit > 0 else None
    run_extraction(limit_per_class=limit)
