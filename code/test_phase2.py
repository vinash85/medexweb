"""
Phase 2 Visual Test — Visual MedGemma description generation
Runs Phase 1 specialist queries, stages the triplet, then sends image + context + prompt
to visual MedGemma for clinical description generation.

Run inside Docker:
    python3 /home/project/code/test_phase2.py
"""
import os
import gc
import re
import sys
import csv
import time
from llama_cpp import Llama

from pipeline import (
    SPECIALIST_PATH, SPECIALIST_PROJ, DATA_DIR,
    run_specialist_queries, run_phase2_description, DermPipeline,
)

CLASSIFICATION_CODES = ["MEL", "NV", "BCC", "BKL"]
REFUSAL_PHRASES = ["i cannot", "i'm sorry", "as an ai", "not able to", "inappropriate"]
SPECIAL_TOKENS = ["<start_of_turn>", "<end_of_turn>", "<eos>", "<pad>", "\u2581"]
THINKING_MARKERS = ["<think>", "<reasoning>", "<thought>"]


def check_flags(adj_desc):
    """Run validation checks on a Phase 2 description. Returns list of flag strings."""
    flags = []
    lower = adj_desc.lower()

    # 1. Non-empty
    if not adj_desc:
        flags.append("empty")
        return flags

    # 2. Meaningful length
    if len(adj_desc) < 10:
        flags.append("too_short")

    # 3. Classification leakage
    for code in CLASSIFICATION_CODES:
        if code in adj_desc.upper():
            flags.append("classification_leakage")
            break

    # 4. Safety guardrail detection
    for phrase in REFUSAL_PHRASES:
        if phrase in lower:
            flags.append("guardrail")
            break

    # 5. Gibberish / token artifact detection
    # 5a. Repeated token patterns (same 3+ char substring repeated 3+ times)
    for length in range(3, min(len(adj_desc) // 3 + 1, 20)):
        for start in range(len(adj_desc) - length * 3 + 1):
            substr = adj_desc[start:start + length]
            if substr * 3 in adj_desc:
                flags.append("gibberish_repeat")
                break
        if "gibberish_repeat" in flags:
            break

    # 5b. Special token leakage
    for token in SPECIAL_TOKENS:
        if token in adj_desc:
            flags.append("special_token_leak")
            break

    # 5c. Excessive non-ASCII characters
    non_ascii = sum(1 for c in adj_desc if ord(c) > 127)
    if non_ascii > len(adj_desc) * 0.3:
        flags.append("excessive_non_ascii")

    # 5d. Very high ratio of non-alphabetic characters
    alpha = sum(1 for c in adj_desc if c.isalpha())
    if len(adj_desc) > 5 and alpha < len(adj_desc) * 0.4:
        flags.append("low_alpha_ratio")

    # 6. Thinking mode leakage
    for marker in THINKING_MARKERS:
        if marker in lower:
            flags.append("thinking_leak")
            break

    return flags


def run_phase2_test():
    all_images = sorted([
        f for f in os.listdir(DATA_DIR)
        if f.lower().endswith(".jpg") and ".proc." not in f and "vlm_in_" not in f
    ])
    total = len(all_images)
    if total == 0:
        print(f"FATAL: No images found in {DATA_DIR}")
        sys.exit(1)

    print("=" * 70)
    print(f"PHASE 2 VISUAL TEST — {total} images")
    print(f"Model: {SPECIALIST_PATH}")
    print(f"Projector: {SPECIALIST_PROJ}")
    print("=" * 70)

    for path in [SPECIALIST_PATH, SPECIALIST_PROJ]:
        if not os.path.exists(path):
            print(f"FATAL: Model file not found: {path}")
            sys.exit(1)

    # Single visual model for both Phase 1 and Phase 2
    print("[LOADING] Visual MedGemma (Phase 1 + Phase 2)...")
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
    guardrail_hits = 0
    gibberish_hits = 0
    clean_count = 0

    for img_idx, img_file in enumerate(all_images, 1):
        print(f"\n[{img_idx}/{total}] {img_file}")

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

        # Validation
        flags = check_flags(adj_desc)
        flag_str = ", ".join(flags) if flags else "clean"

        if not flags:
            clean_count += 1

        # Diagnostic output for flagged images
        has_guardrail = "guardrail" in flags
        has_gibberish = any(f in flags for f in [
            "gibberish_repeat", "special_token_leak", "excessive_non_ascii", "low_alpha_ratio"
        ])
        has_thinking = "thinking_leak" in flags

        if has_guardrail:
            guardrail_hits += 1
        if has_gibberish:
            gibberish_hits += 1

        if flags:
            print(f"  FLAGS: {flag_str}")
            if has_guardrail:
                print(f"  [GUARDRAIL] Full output: {adj_desc}")
                print(f"  [GUARDRAIL] Prompt sent: {phase2_inputs['prompt'][:200]}...")
            if has_gibberish:
                print(f"  [GIBBERISH] Full output: {adj_desc}")
                print(f"  [GIBBERISH] Prompt sent: {phase2_inputs['prompt'][:200]}...")
            if has_thinking:
                print(f"  [THINKING LEAK] Full output: {adj_desc}")
            if any(f in flags for f in ["empty", "too_short", "classification_leakage"]):
                print(f"  [DIAGNOSTIC] Full output: '{adj_desc}'")
                print(f"  [DIAGNOSTIC] raw_obs: {raw_obs}")

        desc_preview = adj_desc[:60] + "..." if len(adj_desc) > 60 else adj_desc
        print(f"  Description ({len(adj_desc)} chars): {desc_preview}")
        print(f"  Status: {flag_str}")

        all_results.append({
            "filename": img_file,
            "Architecture": raw_obs["Architecture"],
            "Network": raw_obs["Network"],
            "Structures": raw_obs["Structures"],
            "Colors": raw_obs["Colors"],
            "adj_desc": adj_desc,
            "desc_length": len(adj_desc),
            "flags": flag_str,
        })

        if os.path.exists(proc_path):
            os.remove(proc_path)

    # --- Cleanup model ---
    vlm.reset()
    del vlm
    gc.collect()

    # --- SUMMARY TABLE ---
    print("\n" + "=" * 70)
    print("PHASE 2 SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Image':<22s} {'Arch':<12s} {'Net':<6s} {'Struct':<15s} {'Colors':<15s} {'Len':>4s} {'Flags':<20s} Description")
    print("-" * 130)
    for r in all_results:
        desc_short = r["adj_desc"][:40] + "..." if len(r["adj_desc"]) > 40 else r["adj_desc"]
        print(f"{r['filename']:<22s} {r['Architecture'][:10]:<12s} {r['Network'][:4]:<6s} "
              f"{r['Structures'][:13]:<15s} {r['Colors'][:13]:<15s} {r['desc_length']:>4d} "
              f"{r['flags'][:18]:<20s} {desc_short}")

    # --- TOTALS ---
    print("\n" + "=" * 70)
    print("TOTALS")
    print("=" * 70)
    print(f"  Images processed: {len(all_results)}")
    print(f"  Errors:           {errors}")
    print(f"  Clean outputs:    {clean_count}")
    print(f"  Guardrail hits:   {guardrail_hits}")
    print(f"  Gibberish hits:   {gibberish_hits}")
    if len(all_results) > 0:
        print(f"  Clean rate:       {100 * clean_count / len(all_results):.1f}%")

    # --- CSV OUTPUT ---
    out_path = "/home/project/data/phase2_results.csv"
    fieldnames = ["filename", "Architecture", "Network", "Structures", "Colors",
                  "adj_desc", "desc_length", "flags"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nResults saved to {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_phase2_test()
