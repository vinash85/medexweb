"""
Phase 3 Classification Test — Text-only MedGemma classification
Two-pass design: Pass 1 loads visual model for Phase 1 + Phase 2 on all images,
Pass 2 unloads visual model, loads text-only model for Phase 3 classification.

Run inside Docker:
    python3 /home/project/code/test_phase3.py
"""
import os
import gc
import re
import sys
import csv
import time
from llama_cpp import Llama

from pipeline import (
    SPECIALIST_PATH, SPECIALIST_PROJ, MEDGEMMA_PATH, DATA_DIR,
    VALID_CODES, REFUSAL_PHRASES, SPECIAL_TOKENS,
    run_specialist_queries, run_phase2_description, run_phase3_classification,
    clean_phase2_output, DermPipeline,
)


def check_classification_flags(code, raw_output):
    """Validate Phase 3 classification output. Returns list of flag strings."""
    flags = []

    # 1. Valid code
    if code not in VALID_CODES:
        flags.append("invalid_code")

    lower = raw_output.lower()

    # 2. Refusal phrases in raw output
    for phrase in REFUSAL_PHRASES:
        if phrase in lower:
            flags.append("refusal")
            break

    # 3. Special token leakage
    for token in SPECIAL_TOKENS:
        if token in raw_output:
            flags.append("special_token_leak")
            break

    # 4. Thinking mode leakage
    for marker in ["<think>", "<reasoning>", "<thought>"]:
        if marker in lower:
            flags.append("thinking_leak")
            break

    # 5. Verbose output (classification should be very short)
    if len(raw_output) > 50:
        flags.append("verbose_output")

    # 6. Multiple codes (model confused)
    codes_found = [c for c in VALID_CODES if c in raw_output.upper()]
    if len(codes_found) > 1:
        flags.append("multiple_codes")

    return flags


def check_cleaning_flags(original_desc, cleaned_desc):
    """Validate that Phase 2 cleaning worked properly. Returns list of flag strings."""
    flags = []

    # 1. Cleaning didn't empty a non-empty input
    if original_desc and original_desc.strip() and not cleaned_desc:
        flags.append("cleaning_emptied")

    lower = cleaned_desc.lower()

    # 2. No refusal phrases survived cleaning
    for phrase in REFUSAL_PHRASES:
        if phrase in lower:
            flags.append("refusal_survived")
            break

    # 3. No special tokens survived cleaning
    for token in SPECIAL_TOKENS:
        if token in cleaned_desc:
            flags.append("special_token_survived")
            break

    return flags


def run_phase3_test():
    all_images = sorted([
        f for f in os.listdir(DATA_DIR)
        if f.lower().endswith(".jpg") and ".proc." not in f and "vlm_in_" not in f
    ])
    total = len(all_images)
    if total == 0:
        print(f"FATAL: No images found in {DATA_DIR}")
        sys.exit(1)

    print("=" * 70)
    print(f"PHASE 3 CLASSIFICATION TEST — {total} images")
    print(f"Model: {MEDGEMMA_PATH}")
    print(f"Projector: {SPECIALIST_PROJ}")
    print("=" * 70)

    for path in [SPECIALIST_PATH, SPECIALIST_PROJ]:
        if not os.path.exists(path):
            print(f"FATAL: Model file not found: {path}")
            sys.exit(1)

    pipeline = DermPipeline()

    # ===== PASS 1: Visual model — Phase 1 + Phase 2 for all images =====
    print("\n[PASS 1] Loading visual MedGemma (Phase 1 + Phase 2)...")
    t0 = time.time()
    vlm = Llama(
        model_path=SPECIALIST_PATH,
        clip_model_path=SPECIALIST_PROJ,
        n_ctx=2048,
        n_gpu_layers=-1,
        chat_format="medgemma-direct",
        verbose=False,
    )
    print(f"[PASS 1] Loaded in {time.time() - t0:.1f}s\n")

    # Store intermediate results for Pass 2
    pass1_results = []
    errors = 0

    for img_idx, img_file in enumerate(all_images, 1):
        print(f"\n[PASS 1] [{img_idx}/{total}] {img_file}")

        proc_path = pipeline._prepare_image(img_file)
        image_uri = pipeline._encode_image_base64(proc_path)

        # Phase 1: Specialist queries
        try:
            raw_obs = run_specialist_queries(vlm, image_uri)
        except Exception as e:
            print(f"  Phase 1 ERROR: {e}")
            errors += 1
            if os.path.exists(proc_path):
                os.remove(proc_path)
            continue

        print(f"  Phase 1: Arch={raw_obs['Architecture']} | Net={raw_obs['Network']} | "
              f"Str={raw_obs['Structures']} | Col={raw_obs['Colors']}")

        # Staging
        phase2_inputs = pipeline.prepare_phase2_inputs(raw_obs, image_uri)

        # Phase 2: Visual description
        try:
            adj_desc = run_phase2_description(vlm, phase2_inputs)
        except Exception as e:
            print(f"  Phase 2 ERROR: {e}")
            errors += 1
            if os.path.exists(proc_path):
                os.remove(proc_path)
            continue

        desc_preview = adj_desc[:60] + "..." if len(adj_desc) > 60 else adj_desc
        print(f"  Phase 2 ({len(adj_desc)} chars): {desc_preview}")

        pass1_results.append({
            "filename": img_file,
            "raw_obs": raw_obs,
            "adj_desc": adj_desc,
        })

        if os.path.exists(proc_path):
            os.remove(proc_path)

    # Unload visual model
    print(f"\n[PASS 1] Complete — {len(pass1_results)} images processed, {errors} errors")
    vlm.reset()
    del vlm
    gc.collect()
    time.sleep(0.5)

    # ===== PASS 2: Text-only model — Phase 3 classification =====
    print("\n[PASS 2] Loading text-only MedGemma (Phase 3 classification)...")
    t0 = time.time()
    text_model = Llama(
        model_path=MEDGEMMA_PATH,
        n_ctx=2048,
        n_gpu_layers=-1,
        chat_format="medgemma-direct",
        verbose=False,
    )
    print(f"[PASS 2] Loaded in {time.time() - t0:.1f}s\n")

    all_results = []
    class_clean = 0
    clean_clean = 0
    code_counts = {c: 0 for c in VALID_CODES}
    code_counts["FALLBACK"] = 0

    for idx, p1 in enumerate(pass1_results, 1):
        img_file = p1["filename"]
        raw_obs = p1["raw_obs"]
        adj_desc = p1["adj_desc"]

        print(f"\n[PASS 2] [{idx}/{len(pass1_results)}] {img_file}")

        # Stage Phase 3 inputs (includes cleaning)
        phase3_inputs = pipeline.prepare_phase3_inputs(raw_obs, adj_desc)
        cleaned_desc = phase3_inputs["cleaned_description"]

        # Cleaning validation
        cleaning_flags = check_cleaning_flags(adj_desc, cleaned_desc)
        if not cleaning_flags:
            clean_clean += 1

        # Phase 3: Classification
        try:
            code, raw_output = run_phase3_classification(text_model, phase3_inputs)
        except Exception as e:
            print(f"  Phase 3 ERROR: {e}")
            code, raw_output = "NV", f"ERROR: {e}"

        # Classification validation
        class_flags = check_classification_flags(code, raw_output)
        if not class_flags:
            class_clean += 1

        # Track distribution
        if code in code_counts:
            code_counts[code] += 1
        else:
            code_counts["FALLBACK"] += 1

        all_flags = cleaning_flags + class_flags
        flag_str = ", ".join(all_flags) if all_flags else "clean"

        print(f"  Cleaned desc: {len(cleaned_desc)} chars (from {len(adj_desc)})")
        print(f"  Classification: {code} (raw: {raw_output[:60]})")
        print(f"  Status: {flag_str}")

        if cleaning_flags:
            print(f"  [CLEANING FLAGS] {', '.join(cleaning_flags)}")
        if class_flags:
            print(f"  [CLASS FLAGS] {', '.join(class_flags)}")

        all_results.append({
            "filename": img_file,
            "Architecture": raw_obs["Architecture"],
            "Network": raw_obs["Network"],
            "Structures": raw_obs["Structures"],
            "Colors": raw_obs["Colors"],
            "adj_desc": adj_desc,
            "desc_length": len(adj_desc),
            "cleaned_desc_length": len(cleaned_desc),
            "classification": code,
            "raw_class_output": raw_output,
            "cleaning_flags": ", ".join(cleaning_flags) if cleaning_flags else "",
            "class_flags": ", ".join(class_flags) if class_flags else "",
            "all_flags": flag_str,
        })

    # Unload text model
    text_model.reset()
    del text_model
    gc.collect()

    # ===== SUMMARY =====
    print("\n" + "=" * 70)
    print("PHASE 3 CLASSIFICATION SUMMARY")
    print("=" * 70)
    print(f"{'Image':<22s} {'Arch':<10s} {'Net':<5s} {'Struct':<13s} {'Colors':<13s} "
          f"{'Len':>4s} {'CLn':>4s} {'Code':<5s} {'Flags'}")
    print("-" * 120)
    for r in all_results:
        print(f"{r['filename']:<22s} {r['Architecture'][:8]:<10s} {r['Network'][:3]:<5s} "
              f"{r['Structures'][:11]:<13s} {r['Colors'][:11]:<13s} "
              f"{r['desc_length']:>4d} {r['cleaned_desc_length']:>4d} {r['classification']:<5s} "
              f"{r['all_flags']}")

    # --- TOTALS ---
    n = len(all_results)
    print("\n" + "=" * 70)
    print("TOTALS")
    print("=" * 70)
    print(f"  Images processed:      {n}")
    print(f"  Phase 1+2 errors:      {errors}")
    print(f"  Classification clean:  {class_clean}")
    print(f"  Cleaning clean:        {clean_clean}")
    if n > 0:
        overall_clean = sum(1 for r in all_results if r["all_flags"] == "clean")
        print(f"  Overall clean:         {overall_clean}")
        print(f"  Clean rate:            {100 * overall_clean / n:.1f}%")

    # --- CLASSIFICATION DISTRIBUTION ---
    print("\n" + "=" * 70)
    print("CLASSIFICATION DISTRIBUTION")
    print("=" * 70)
    max_count = max(code_counts.values()) if code_counts else 1
    for code_name in VALID_CODES + ["FALLBACK"]:
        count = code_counts[code_name]
        bar_len = int(40 * count / max_count) if max_count > 0 else 0
        bar = "#" * bar_len
        pct = 100 * count / n if n > 0 else 0
        print(f"  {code_name:<8s} {count:>3d} ({pct:5.1f}%)  {bar}")

    # Stickiness check — flag if all predictions are the same code
    unique_codes = set(r["classification"] for r in all_results)
    if len(unique_codes) == 1 and n > 1:
        print(f"\n  WARNING: STICKINESS DETECTED — all {n} images classified as {unique_codes.pop()}")
    elif len(unique_codes) == 2 and n > 10:
        print(f"\n  NOTE: Only 2 unique codes used: {unique_codes}")
    else:
        print(f"\n  Diversity: {len(unique_codes)} unique codes — OK")

    # --- CSV OUTPUT ---
    out_path = "/home/project/data/phase3_results.csv"
    fieldnames = [
        "filename", "Architecture", "Network", "Structures", "Colors",
        "adj_desc", "desc_length", "cleaned_desc_length",
        "classification", "raw_class_output",
        "cleaning_flags", "class_flags", "all_flags",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nResults saved to {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_phase3_test()
