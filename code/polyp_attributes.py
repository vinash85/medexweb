#!/usr/bin/env python3
"""
Polyp JNET Classification — Visual Attribute Extraction with MedGemma
=====================================================================
For each endoscopic polyp image, asks MedGemma to describe visual attributes
WITHOUT revealing the JNET class. The JNET class is parsed from the filename
and used only for grouping output.

Folder structure: all images are in a single flat directory (data/polyp/).
Filenames encode JNET class: e.g. 23001_1_Tubular_LGD_JNet_2A.jpg

Output: one JSON per JNET class in data/polyp_attributes/

Run inside Docker:
    python3 /home/project/code/polyp_attributes.py
"""

import os
import sys
import gc
import json
import uuid
import re
import time
import base64
import argparse
import traceback
from datetime import datetime

from PIL import Image, ImageEnhance
from tqdm import tqdm

# ── Paths (Docker layout) ────────────────────────────────────────────────────
PROJECT_ROOT = "/home/project"
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
POLYP_IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "polyp")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "polyp_attributes")
OUTPUT_TXT = os.path.join(OUTPUT_DIR, "summary.txt")

# ── MedGemma Model Paths ─────────────────────────────────────────────────────
MEDGEMMA_MODEL = os.path.join(MODEL_DIR, "medgemma.gguf")
MEDGEMMA_MMPROJ = os.path.join(MODEL_DIR, "medgemma-mmproj.gguf")

# ── JNET class regex: match JNet_1, JNet_2A, JNet_2B, JNet_3 ────────────────
JNET_RE = re.compile(r"JNet_(1|2A|2B|3)", re.IGNORECASE)

# ── JNET classes ──────────────────────────────────────────────────────────────
JNET_CLASSES = ["1", "2A", "2B", "3"]

# ── JNET class descriptions for prompt ────────────────────────────────────────
JNET_DESCRIPTIONS = {
    "1": "JNET Type 1",
    "2A": "JNET Type 2A",
    "2B": "JNET Type 2B",
    "3": "JNET Type 3",
}

# ── Prompt Template ──────────────────────────────────────────────────────────
ATTRIBUTE_PROMPT = (
    "This is a narrow-band imaging (NBI) endoscopic image of a colorectal polyp. "
    "The confirmed diagnosis is: {diagnosis}.\n\n"
    "Describe the visual attributes in this image that are consistent with "
    "and support this diagnosis. Focus on:\n"
    "- Surface pattern (pit pattern, regularity, shape of surface structures)\n"
    "- Vascular pattern (vessel visibility, vessel caliber, regularity, arrangement)\n"
    "- Color and brightness (mucosal color relative to surrounding tissue)\n"
    "- Borders and demarcation (clarity of polyp margins)\n"
    "- Surface texture (smooth, granular, villous, lobulated, ulcerated)\n"
    "- Overall morphology (sessile, pedunculated, flat, depressed)\n\n"
    "IMPORTANT: Only describe attributes you can clearly identify in this image. "
    "Do NOT mention JNET classification or Kudo pit pattern classification. "
    "If you are uncertain about any attribute, do NOT mention it. "
    "Be concise and factual. Do not speculate."
)


def prepare_image(filepath):
    """Resize to 224x224, enhance contrast, save to temp file."""
    temp_path = os.path.join(
        os.path.dirname(filepath),
        f"polyp_proc_{uuid.uuid4().hex[:6]}.jpg",
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


def parse_jnet_class(filename):
    """Extract JNET class (1, 2A, 2B, 3) from filename. Returns None if not found."""
    match = JNET_RE.search(filename)
    if match:
        return match.group(1).upper()
    return None


def collect_images(limit_per_class=None):
    """Scan flat polyp directory and return list of (filepath, filename, jnet_class).

    Args:
        limit_per_class: If set, cap images per JNET class to this number.
    """
    images = []
    class_counts = {c: 0 for c in JNET_CLASSES}

    for fname in sorted(os.listdir(POLYP_IMAGE_DIR)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        jnet = parse_jnet_class(fname)
        if jnet is None:
            continue
        if jnet not in JNET_CLASSES:
            continue
        if limit_per_class and class_counts[jnet] >= limit_per_class:
            continue
        images.append({
            "filepath": os.path.join(POLYP_IMAGE_DIR, fname),
            "filename": fname,
            "jnet_class": jnet,
        })
        class_counts[jnet] += 1

    return images


def load_existing_results():
    """Load previously saved JSON results for resume support."""
    existing = {}
    for jnet in JNET_CLASSES:
        safe_name = f"jnet_{jnet.lower()}"
        json_path = os.path.join(OUTPUT_DIR, f"polyp_attributes_{safe_name}.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                existing[jnet] = data.get("images", {})
            except (json.JSONDecodeError, KeyError):
                existing[jnet] = {}
        else:
            existing[jnet] = {}
    return existing


def run_extraction(limit_per_class=None):
    print("=" * 70)
    print("Polyp JNET — Visual Attribute Extraction (MedGemma)")
    print(f"Started: {datetime.now().isoformat()}")
    if limit_per_class:
        print(f"Limit: {limit_per_class} images per JNET class")
    print("=" * 70)

    # Collect all images (respecting limit)
    images = collect_images(limit_per_class=limit_per_class)
    print(f"\nTotal images: {len(images)}")
    for jnet in JNET_CLASSES:
        count = sum(1 for img in images if img["jnet_class"] == jnet)
        print(f"  JNet {jnet}: {count}")

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
    target_filenames = {(img["jnet_class"], img["filename"]) for img in images}
    results_by_class = {jnet: {} for jnet in JNET_CLASSES}
    for jnet in JNET_CLASSES:
        for fname, result in all_existing.get(jnet, {}).items():
            if (jnet, fname) in target_filenames:
                results_by_class[jnet][fname] = result
    already_done = sum(
        1 for jnet in results_by_class
        for r in results_by_class[jnet].values()
        if r.get("status") == "ok"
    )
    if already_done > 0:
        print(f"\n[RESUME] Found {already_done} previously completed images — skipping them.")

    # Filter to only images that need processing
    to_process = [
        img for img in images
        if img["filename"] not in results_by_class.get(img["jnet_class"], {})
        or results_by_class[img["jnet_class"]][img["filename"]].get("status") != "ok"
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
            jnet_class = img_info["jnet_class"]

            pbar.set_postfix_str(f"JNet_{jnet_class}/{filename[:20]}")
            t_start = time.time()

            proc_path = None
            try:
                # Prepare image
                proc_path = prepare_image(filepath)
                image_uri = encode_image_base64(proc_path)

                # Query MedGemma with JNET diagnosis
                diagnosis = JNET_DESCRIPTIONS[jnet_class]
                prompt_text = ATTRIBUTE_PROMPT.format(diagnosis=diagnosis)
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

                results_by_class[jnet_class][filename] = {
                    "jnet_class": jnet_class,
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
                tqdm.write(f"  ERROR (JNet_{jnet_class}/{filename}): {e}")
                traceback.print_exc()
                results_by_class[jnet_class][filename] = {
                    "jnet_class": jnet_class,
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

    # Print timing summary
    if image_times:
        avg_t = sum(image_times) / len(image_times)
        print(f"\n[TIMING] {len(image_times)} images | avg {avg_t:.1f}s/img | total {sum(image_times)/60:.1f}min")

    # Unload model
    print("\n[INFO] Unloading MedGemma...")
    del llm
    gc.collect()

    # ── Save one JSON per JNET class ──────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for jnet_class, file_results in results_by_class.items():
        if not file_results:
            continue
        safe_name = f"jnet_{jnet_class.lower()}"
        json_path = os.path.join(OUTPUT_DIR, f"polyp_attributes_{safe_name}.json")
        output = {
            "metadata": {
                "model": "MedGemma",
                "timestamp": datetime.now().isoformat(),
                "jnet_class": jnet_class,
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
    total_ok = sum(1 for jnet in results_by_class for r in results_by_class[jnet].values() if r.get("status") == "ok")
    total_err = sum(1 for jnet in results_by_class for r in results_by_class[jnet].values() if r.get("status") == "error")
    summary_lines = [
        "=" * 70,
        "Polyp JNET — Visual Attribute Extraction Summary",
        f"Generated: {datetime.now().isoformat()}",
        f"Model: MedGemma",
        f"Total images: {total}  |  Success: {total_ok}  |  Errors: {total_err}",
        f"  (this run: {counters['success']} new, {counters['skipped']} resumed)",
        "=" * 70,
    ]

    for jnet_class in JNET_CLASSES:
        file_results = results_by_class[jnet_class]
        ok_results = {f: r for f, r in file_results.items() if r["status"] == "ok"}
        summary_lines.append(f"\n{'─' * 60}")
        summary_lines.append(f"  JNet {jnet_class} ({len(ok_results)} images)")
        summary_lines.append(f"{'─' * 60}")
        for fname, r in ok_results.items():
            summary_lines.append(f"\n  [{fname}]")
            for line in r["attributes"].split("\n"):
                summary_lines.append(f"    {line}")

    summary_text = "\n".join(summary_lines) + "\n"
    with open(OUTPUT_TXT, "w") as f:
        f.write(summary_text)
    print(f"Summary saved: {OUTPUT_TXT}")
    print(f"\nDone. {total_ok}/{total} images completed ({counters['success']} new this run, {counters['skipped']} resumed).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Polyp JNET Attribute Extraction with MedGemma")
    parser.add_argument(
        "--limit", type=int, default=100,
        help="Max images per JNET class (default: 100, use 0 for all)",
    )
    args = parser.parse_args()
    limit = args.limit if args.limit > 0 else None
    run_extraction(limit_per_class=limit)
