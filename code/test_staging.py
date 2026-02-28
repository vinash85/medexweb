"""
Staging Input Verification — Validate (context, image, prompt) triplets
Runs Phase 1 specialist on all images via pipeline.run_specialist_queries(),
then calls prepare_phase2_inputs() for each, validating the staged triplet.

Run inside Docker:
    python3 /home/project/code/test_staging.py
"""
import os
import gc
import sys
import csv
import time
from llama_cpp import Llama

# Import all Phase 1 logic and staging functions from pipeline
from pipeline import (
    SPECIALIST_PATH, SPECIALIST_PROJ, DATA_DIR, OBS_QUERIES,
    run_specialist_queries, build_specialist_context, DermPipeline,
)

EXPECTED_KEYS = list(OBS_QUERIES.keys())


def run_staging_test():
    all_images = sorted([
        f for f in os.listdir(DATA_DIR)
        if f.lower().endswith(".jpg") and ".proc." not in f and "vlm_in_" not in f
    ])
    total = len(all_images)
    if total == 0:
        print(f"FATAL: No images found in {DATA_DIR}")
        sys.exit(1)

    print("=" * 70)
    print(f"STAGING INPUT TEST — {total} images")
    print(f"Model: {SPECIALIST_PATH}")
    print(f"Projector: {SPECIALIST_PROJ}")
    print("=" * 70)

    for path in [SPECIALIST_PATH, SPECIALIST_PROJ]:
        if not os.path.exists(path):
            print(f"FATAL: Model file not found: {path}")
            sys.exit(1)

    print("[LOADING] Specialist VLM...")
    t0 = time.time()
    vlm = Llama(
        model_path=SPECIALIST_PATH,
        clip_model_path=SPECIALIST_PROJ,
        n_ctx=2048,
        n_gpu_layers=-1,
        chat_format="medgemma-direct",
        verbose=False,
    )
    print(f"[LOADED] in {time.time() - t0:.1f}s\n")

    pipeline = DermPipeline()
    all_results = []
    errors = 0
    validation_failures = 0

    for img_idx, img_file in enumerate(all_images, 1):
        print(f"\n[{img_idx}/{total}] {img_file}")

        # Use pipeline's image preparation and encoding
        proc_path = pipeline._prepare_image(img_file)
        image_uri = pipeline._encode_image_base64(proc_path)

        # Run Phase 1 using shared function from pipeline
        try:
            raw_obs = run_specialist_queries(vlm, image_uri)
        except Exception as e:
            print(f"  ERROR: {e}")
            errors += 1
            if os.path.exists(proc_path):
                os.remove(proc_path)
            continue

        print(f"  Arch: {raw_obs['Architecture']} | Net: {raw_obs['Network']} | "
              f"Str: {raw_obs['Structures']} | Col: {raw_obs['Colors']}")

        # --- Stage the triplet ---
        phase2 = pipeline.prepare_phase2_inputs(raw_obs, image_uri)

        # --- Validate the triplet ---
        issues = []

        # 1. Context contains all 4 attributes
        # Map keys to the labels used in build_specialist_context()
        context_labels = {
            "Architecture": "Shape (Architecture)",
            "Network": "Pigment network",
            "Structures": "Structures",
            "Colors": "Colors",
        }
        for attr in EXPECTED_KEYS:
            label = context_labels[attr]
            if label not in phase2["context"]:
                issues.append(f"context missing '{attr}'")

        # 2. Image URI is a valid base64 data URI
        uri_valid = phase2["image_uri"].startswith("data:image/jpeg;base64,")
        if not uri_valid:
            issues.append("image_uri not a valid data URI")

        # 3. Prompt includes context + comparison review request
        if phase2["context"] not in phase2["prompt"]:
            issues.append("prompt does not include context")
        if "compare" not in phase2["prompt"].lower():
            issues.append("prompt missing comparison request")

        # 4. raw_obs has all 4 keys with non-empty values
        for attr in EXPECTED_KEYS:
            if attr not in phase2["raw_obs"] or not phase2["raw_obs"][attr]:
                issues.append(f"raw_obs['{attr}'] missing or empty")

        if issues:
            validation_failures += 1
            print(f"  VALIDATION FAIL: {'; '.join(issues)}")
        else:
            print(f"  VALIDATION OK: context={len(phase2['context'])} chars, "
                  f"prompt={len(phase2['prompt'])} chars, image_uri=valid")

        all_results.append({
            "filename": img_file,
            "Architecture": raw_obs["Architecture"],
            "Network": raw_obs["Network"],
            "Structures": raw_obs["Structures"],
            "Colors": raw_obs["Colors"],
            "context": phase2["context"],
            "prompt_length": len(phase2["prompt"]),
            "image_uri_valid": uri_valid,
        })

        if os.path.exists(proc_path):
            os.remove(proc_path)

    # --- Cleanup model ---
    vlm.reset()
    del vlm
    gc.collect()

    # --- SUMMARY TABLE ---
    print("\n" + "=" * 70)
    print("STAGING SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Image':<22s} {'Arch':<15s} {'Network':<8s} {'Structures':<22s} {'Colors':<20s} {'URI':5s} {'Prompt'}")
    print("-" * 100)
    for r in all_results:
        print(f"{r['filename']:<22s} {r['Architecture'][:13]:<15s} {r['Network'][:6]:<8s} "
              f"{r['Structures'][:20]:<22s} {r['Colors'][:18]:<20s} "
              f"{'OK' if r['image_uri_valid'] else 'BAD':5s} {r['prompt_length']} chars")

    # --- RESULTS ---
    print("\n" + "=" * 70)
    print(f"TOTAL: {total} images | ERRORS: {errors} | VALIDATION FAILURES: {validation_failures}")
    if validation_failures == 0 and errors == 0:
        print("ALL TRIPLETS VALIDATED SUCCESSFULLY")
    print("=" * 70)

    # --- DUMP RESULTS TO CSV ---
    out_path = "/home/project/data/staging_results.csv"
    fieldnames = ["filename", "Architecture", "Network", "Structures", "Colors",
                  "context", "prompt_length", "image_uri_valid"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_staging_test()
