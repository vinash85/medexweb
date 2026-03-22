#!/usr/bin/env python3
"""
vlm2 Pipeline Test — 25 images (5 per class, or max available)
==============================================================
Runs the full 3-phase multi-VLM consensus pipeline on a stratified sample
and saves all phase outputs per image to CSV.

Run inside Docker:
    python3 /home/project/code/test_vlm2_pipeline.py
"""

import os
import sys
import csv
import json
import time
import traceback
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.join("/home/project", "code"))
from pipeline import (
    DermPipeline, ATTRIBUTE_KEYS, VALID_CODES,
    DATA_DIR, ATTR_TO_UI_KEY,
)

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = "/home/project"
CSV_PATH = os.path.join(PROJECT_ROOT, "code", "DermsGemms.csv")
IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "images")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_CSV = os.path.join(OUTPUT_DIR, "vlm2_pipeline_results.csv")
SUMMARY_TXT = os.path.join(OUTPUT_DIR, "vlm2_pipeline_summary.txt")

# ── Sample Config ────────────────────────────────────────────────────────────
SAMPLE_CLASSES = ["mel", "nv", "bcc", "bkl", "akiec", "df"]
N_PER_CLASS = 5


def select_sample_images():
    """Select up to N_PER_CLASS images per class."""
    df = pd.read_csv(CSV_PATH, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df = df.drop_duplicates(subset=["Image"], keep="first")
    sample = []
    for dx in SAMPLE_CLASSES:
        subset = df[df["Dx"] == dx].sort_values("Image").head(N_PER_CLASS)
        n_avail = len(df[df["Dx"] == dx])
        if len(subset) < N_PER_CLASS:
            print(f"  Note: {dx} has only {n_avail} image(s), using all")
        for _, row in subset.iterrows():
            sample.append({
                "image_id": row["Image"],
                "dx": row["Dx"],
                "gt_attributes": row.get("Lesion attributes", ""),
            })
    return sample


def run_test():
    started = datetime.now()
    print("=" * 70)
    print("vlm2 Pipeline Test")
    print(f"Started: {started.isoformat()}")
    print("=" * 70)

    # Select sample
    sample = select_sample_images()
    class_counts = {}
    for s in sample:
        class_counts[s["dx"]] = class_counts.get(s["dx"], 0) + 1
    class_desc = ", ".join(f"{dx.upper()}={n}" for dx, n in class_counts.items())
    print(f"\nSample: {len(sample)} images ({class_desc})")
    for s in sample:
        print(f"  {s['image_id']} ({s['dx']})")

    # Check missing images
    missing = [s["image_id"] for s in sample
               if not os.path.exists(os.path.join(IMAGE_DIR, f"{s['image_id']}.jpg"))]
    if missing:
        print(f"\nWARNING: {len(missing)} images not found: {missing}")

    # Load pipeline
    print("\nLoading pipeline...")
    t0 = time.time()
    pipeline = DermPipeline()
    print(f"Pipeline loaded in {time.time() - t0:.1f}s\n")

    # Run pipeline on each image
    all_results = []

    try:
        for idx, s in enumerate(sample):
            image_id = s["image_id"]
            fname = f"{image_id}.jpg"
            img_path = os.path.join(IMAGE_DIR, fname)

            print(f"\n{'─' * 60}")
            print(f"[{idx+1}/{len(sample)}] {image_id} (gt: {s['dx'].upper()})")
            print(f"{'─' * 60}")

            if not os.path.exists(img_path):
                print(f"  SKIP: image not found")
                all_results.append({
                    "image_id": image_id, "dx": s["dx"],
                    "gt_attributes": s["gt_attributes"],
                    "ai_code": "MISSING", "match": "",
                    "error": "image not found",
                })
                continue

            row = {
                "image_id": image_id,
                "dx": s["dx"],
                "gt_attributes": s["gt_attributes"],
            }

            try:
                t_start = time.time()
                result = pipeline.process(fname)
                elapsed = time.time() - t_start

                row["ai_code"] = result["ai_code"]
                row["match"] = "YES" if result["ai_code"] == s["dx"].upper() else "NO"
                row["elapsed_s"] = f"{elapsed:.1f}"
                row["error"] = ""

                # Phase 1: per-attribute concordance
                # Extract from final_implication report
                for attr_key in ATTRIBUTE_KEYS:
                    ui_key = ATTR_TO_UI_KEY.get(attr_key, attr_key)
                    obs_val = result.get("raw_obs", {}).get(ui_key, "")
                    row[f"p1_{attr_key}"] = obs_val

                # Phase 2: adj_desc (summary of MedGemma review)
                row["p2_summary"] = result.get("adj_desc", "")

                # Phase 3: full report
                row["p3_report"] = result.get("final_implication", "")

                status = "MATCH" if row["match"] == "YES" else "MISS"
                print(f"  Result: {result['ai_code']} (gt: {s['dx'].upper()}) "
                      f"[{status}] in {elapsed:.1f}s")

            except Exception as e:
                print(f"  ERROR: {e}")
                traceback.print_exc()
                row["ai_code"] = "ERROR"
                row["match"] = ""
                row["elapsed_s"] = ""
                row["error"] = str(e)

            all_results.append(row)

    except KeyboardInterrupt:
        print("\n\nInterrupted — saving partial results...")

    # ── Save CSV ─────────────────────────────────────────────────────────────
    if not all_results:
        print("\nNo results to save.")
        return

    fieldnames = [
        "image_id", "dx", "gt_attributes", "ai_code", "match", "elapsed_s", "error",
    ]
    for attr_key in ATTRIBUTE_KEYS:
        fieldnames.append(f"p1_{attr_key}")
    fieldnames.extend(["p2_summary", "p3_report"])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)
    print(f"\nResults saved: {RESULTS_CSV} ({len(all_results)} rows)")

    # ── Summary ──────────────────────────────────────────────────────────────
    finished = datetime.now()
    duration = finished - started

    df_res = pd.DataFrame(all_results)
    valid = df_res[df_res["ai_code"].isin(VALID_CODES)]
    n_match = len(valid[valid["match"] == "YES"])
    n_total = len(valid)
    accuracy = 100.0 * n_match / n_total if n_total > 0 else 0

    lines = [
        "=" * 70,
        "vlm2 Pipeline Test — Summary",
        f"Started:  {started.isoformat()}",
        f"Finished: {finished.isoformat()}",
        f"Duration: {duration}",
        "=" * 70,
        "",
        f"Images tested: {len(all_results)}",
        f"Overall accuracy: {n_match}/{n_total} ({accuracy:.1f}%)",
        "",
        "## Per-Class Results",
        f"{'Class':<8} {'Total':>6} {'Correct':>8} {'Accuracy':>9}  Predictions",
        "-" * 70,
    ]

    for dx in SAMPLE_CLASSES:
        cls_df = valid[valid["dx"] == dx]
        if len(cls_df) == 0:
            lines.append(f"{dx.upper():<8} {'0':>6}")
            continue
        cls_match = len(cls_df[cls_df["match"] == "YES"])
        cls_total = len(cls_df)
        cls_acc = 100.0 * cls_match / cls_total if cls_total > 0 else 0
        pred_counts = cls_df["ai_code"].value_counts().to_dict()
        pred_str = ", ".join(f"{k}={v}" for k, v in sorted(pred_counts.items()))
        lines.append(
            f"{dx.upper():<8} {cls_total:>6} {cls_match:>8} {cls_acc:>8.1f}%  {pred_str}"
        )

    # Confusion summary
    lines.append("")
    lines.append("## Confusion Matrix")
    lines.append(f"{'Predicted →':<10} " + " ".join(f"{c:>6}" for c in VALID_CODES))
    lines.append("-" * 60)
    for gt in SAMPLE_CLASSES:
        gt_upper = gt.upper()
        cls_df = valid[valid["dx"] == gt]
        counts = []
        for pred in VALID_CODES:
            counts.append(str(len(cls_df[cls_df["ai_code"] == pred])))
        lines.append(f"{gt_upper:<10} " + " ".join(f"{c:>6}" for c in counts))

    # Timing
    timed = df_res[df_res["elapsed_s"] != ""]
    if len(timed) > 0:
        times = timed["elapsed_s"].astype(float)
        lines.append("")
        lines.append("## Timing")
        lines.append(f"  Mean:   {times.mean():.1f}s per image")
        lines.append(f"  Median: {times.median():.1f}s")
        lines.append(f"  Min:    {times.min():.1f}s")
        lines.append(f"  Max:    {times.max():.1f}s")

    # Errors
    errors = df_res[df_res["error"] != ""]
    if len(errors) > 0:
        lines.append("")
        lines.append(f"## Errors ({len(errors)})")
        for _, row in errors.iterrows():
            lines.append(f"  {row['image_id']}: {row['error']}")

    summary_text = "\n".join(lines) + "\n"
    with open(SUMMARY_TXT, "w") as f:
        f.write(summary_text)
    print(f"Summary saved: {SUMMARY_TXT}")
    print("\n" + summary_text)


if __name__ == "__main__":
    run_test()
