"""
Phase 1 FULL Test — PaliGemma Specialist (Vision Model)
Hybrid prompts: Architecture from Run 3, rest from Run 1.
Runs ALL images with distribution analysis and stickiness check.

Run inside Docker:
    python3 /home/project/code/test_phase1.py
"""
import os
import gc
import re
import sys
import csv
import base64
import time
from PIL import Image, ImageEnhance
from llama_cpp import Llama
from llama_cpp.llama_chat_format import (
    register_chat_format,
    ChatFormatterResponse,
)

# --- Register a raw pass-through chat handler for PaliGemma base ---
@register_chat_format("paligemma-raw")
def paligemma_raw_handler(messages, **kwargs):
    prompt = ""
    for msg in messages:
        if isinstance(msg["content"], list):
            for part in msg["content"]:
                if part["type"] == "text":
                    prompt += part["text"]
        elif isinstance(msg["content"], str):
            prompt += msg["content"]
    return ChatFormatterResponse(prompt=prompt + "\n")

# --- MedGemma direct-answer handler (bypasses thinking mode) ---
@register_chat_format("medgemma-direct")
def medgemma_direct_handler(messages, **kwargs):
    prompt = ""
    for msg in messages:
        role = msg.get("role", "user")
        prompt += f"<start_of_turn>{role}\n"
        if isinstance(msg["content"], list):
            for part in msg["content"]:
                if part["type"] == "text":
                    prompt += part["text"]
        elif isinstance(msg["content"], str):
            prompt += msg["content"]
        prompt += "<end_of_turn>\n"
    # Force model to start with "Answer:" to skip thinking mode
    prompt += "<start_of_turn>model\nAnswer:"
    return ChatFormatterResponse(prompt=prompt)

# --- CONFIG ---
MODEL_DIR = "/home/project/models"
DATA_DIR = "/home/project/data/images"
# MedGemma multimodal for Phase 1 comparison
SPECIALIST_PATH = os.path.join(MODEL_DIR, "medgemma.gguf")
SPECIALIST_PROJ = os.path.join(MODEL_DIR, "medgemma-mmproj.gguf")

EXPECTED_KEYS = ["Architecture", "Network", "Structures", "Colors"]


def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def prepare_image(filepath: str) -> str:
    temp_path = filepath + ".proc.jpg"
    with Image.open(filepath) as img:
        img = img.convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
        img = ImageEnhance.Contrast(img).enhance(1.4)
        img.save(temp_path, quality=90)
    return temp_path




def clean_response(val, key=None):
    """Strip markdown artifacts and trailing explanations from model output."""
    # Remove markdown bold markers
    val = val.replace("*", "")
    # Take only the first line (drop trailing explanations after blank lines)
    val = val.split("\n")[0].strip()
    # Remove trailing periods
    val = val.rstrip(".")
    # Collapse extra whitespace
    val = re.sub(r"\s+", " ", val).strip()

    # --- Per-key normalization ---
    if key == "Structures":
        # Extract only the structure keywords we care about
        found = [s for s in ["dots", "globules", "streaks"] if s in val]
        val = ", ".join(found) if found else "none"

    if key == "Colors":
        # Extract only recognized color names
        known_colors = ["red", "white", "blue", "black", "brown", "yellow", "pink", "orange", "gray", "tan"]
        found = [c for c in known_colors if c in val]
        val = ", ".join(found) if found else "unknown"

    return val


def run_full_test():
    all_images = sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith(".jpg") and ".proc." not in f])
    total = len(all_images)
    if total == 0:
        print(f"FATAL: No images found in {DATA_DIR}")
        sys.exit(1)

    print("=" * 70)
    print(f"PHASE 1 FULL TEST (MedGemma) — {total} images")
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

    # PaliGemma format (answer en mode) — not supported by MedGemma
    # obs_queries = {
    #     "Architecture": "answer en What is the shape of this image? round, oval, or irregular?",
    #     "Network": "answer en Does this lesion have a pigment network? yes or no",
    #     "Structures": "answer en Describe structures: dots, globules, or haze. Say clear if none",
    #     "Colors": "answer en What colors are in this lesion?",
    # }

    # MedGemma format — direct clinical questions
    obs_queries = {
        "Architecture": "What is the shape of this lesion? Answer with one word: round, oval, or irregular.",
        "Network": "Does this lesion have a pigment network? Answer with one word: yes or no.",
        "Structures": "What structures are visible in this lesion? Answer briefly: dots, globules, streaks, or none.",
        "Colors": "What colors are present in this lesion? Reply with only color names separated by commas, nothing else.",
    }

    # Collect all results for summary table
    all_results = []
    errors = 0

    for img_idx, img_file in enumerate(all_images, 1):
        print(f"[{img_idx}/{total}] {img_file}", end=" ", flush=True)

        img_path = os.path.join(DATA_DIR, img_file)
        proc_path = prepare_image(img_path)
        image_uri = encode_image_base64(proc_path)

        row = {"Image": img_file}
        vlm.reset()  # Flush KV cache before each new image
        for key, q in obs_queries.items():
            try:
                res = vlm.create_chat_completion(
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_uri}},
                            {"type": "text", "text": q},
                        ],
                    }],
                    max_tokens=32,
                    temperature=0.7,
                    repeat_penalty=1.5,
                )
                val = res["choices"][0]["message"]["content"].strip().lower()
                row[key] = clean_response(val, key)
            except Exception as e:
                row[key] = f"ERR"
                errors += 1

        print(f"| Arch: {row['Architecture'][:15]:15s} | Net: {row['Network'][:5]:5s} | Str: {row['Structures'][:20]:20s} | Col: {row['Colors']}")
        all_results.append(row)

        if os.path.exists(proc_path):
            os.remove(proc_path)

    # --- Cleanup model ---
    vlm.reset()
    del vlm
    gc.collect()

    # --- SUMMARY TABLE ---
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Image':<22s} {'Arch':<15s} {'Network':<8s} {'Structures':<22s} {'Colors'}")
    print("-" * 90)
    for r in all_results:
        print(f"{r['Image']:<22s} {r['Architecture'][:13]:<15s} {r['Network'][:6]:<8s} {r['Structures'][:20]:<22s} {r['Colors']}")

    # --- DISTRIBUTION COUNTS ---
    print("\n" + "=" * 70)
    print("DISTRIBUTION ANALYSIS")
    print("=" * 70)
    for key in EXPECTED_KEYS:
        counts = {}
        for r in all_results:
            v = r.get(key, "MISSING")
            counts[v] = counts.get(v, 0) + 1
        sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
        print(f"\n  {key}:")
        for val, cnt in sorted_counts:
            pct = 100 * cnt / total
            bar = "#" * int(pct / 2)
            print(f"    {val[:50]:<52s} {cnt:3d} ({pct:5.1f}%) {bar}")

    # --- STICKINESS CHECK ---
    print("\n" + "=" * 70)
    print("STICKINESS CHECK")
    print("=" * 70)
    for key in EXPECTED_KEYS:
        vals = [r[key] for r in all_results]
        unique = len(set(vals))
        if unique == 1:
            print(f"  {key}: STUCK — all {total} images returned '{vals[0]}'")
        else:
            print(f"  {key}: OK — {unique} distinct values across {total} images")

    if errors:
        print(f"\n  ERRORS: {errors} query failures")

    # --- DUMP RESULTS TO CSV ---
    out_path = "/home/project/data/phase1_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Image"] + EXPECTED_KEYS)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nResults saved to {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_full_test()
