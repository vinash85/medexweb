"""
Color Prompt Validation Test — runs ONLY the Colors query on sample images.

Tests what the model actually returns with the new dermoscopic color prompt
before we trust color-based rules (pink→BCC, blue→MEL, color_count≥3→MEL).

Run inside Docker:
    python3 /home/project/code/test_colors.py
"""
import os
import sys
import csv
import time

from pipeline import (
    DATA_DIR, OBS_QUERIES, clean_response, DermPipeline,
)


def run_colors_test():
    # Load ground truth for diagnosis labels
    gt = {}
    csv_path = os.path.join(os.path.dirname(__file__), "DermsGemms.csv")
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            gt[row["Image"].strip()] = row["Dx"].strip().upper()

    all_images = sorted([
        f for f in os.listdir(DATA_DIR)
        if f.lower().endswith(".jpg") and ".proc." not in f and "vlm_in_" not in f
    ])

    if not all_images:
        print(f"FATAL: No images in {DATA_DIR}")
        sys.exit(1)

    # Sample: pick up to 5 per class
    by_class = {"MEL": [], "NV": [], "BCC": [], "BKL": []}
    for img in all_images:
        name = img.replace(".jpg", "")
        dx = gt.get(name, "?")
        if dx in by_class and len(by_class[dx]) < 5:
            by_class[dx].append(img)

    sample = []
    for dx, imgs in by_class.items():
        for img in imgs:
            sample.append((img, dx))

    print("=" * 70)
    print(f"COLOR PROMPT VALIDATION — {len(sample)} images")
    print(f"Prompt: {OBS_QUERIES['Colors']}")
    print("=" * 70)

    pipeline = DermPipeline()

    results = []
    for i, (img_file, dx) in enumerate(sample, 1):
        print(f"\n[{i}/{len(sample)}] {img_file} (GT: {dx})")

        proc_path = pipeline._prepare_image(img_file)
        image_uri = pipeline._encode_image_base64(proc_path)

        try:
            pipeline._vlm.reset()
            res = pipeline._vlm.create_chat_completion(
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_uri}},
                        {"type": "text", "text": OBS_QUERIES["Colors"]}
                    ]
                }],
                max_tokens=32,
                temperature=0.7,
                repeat_penalty=1.5,
            )
            raw_val = res["choices"][0]["message"]["content"].strip().lower()
            cleaned = clean_response(raw_val, "Colors")
            color_list = [c.strip() for c in cleaned.split(",")]

            print(f"  RAW:     {raw_val}")
            print(f"  CLEANED: {cleaned}")
            print(f"  COUNT:   {len(color_list)}")
            print(f"  has_red: {'red' in raw_val}")
            print(f"  has_pink: {'pink' in cleaned}")
            print(f"  has_blue: {'blue' in cleaned}")
            print(f"  has_black: {'black' in cleaned}")

            results.append({
                "image": img_file, "dx": dx,
                "raw": raw_val, "cleaned": cleaned,
                "count": len(color_list),
                "has_red": "red" in raw_val,
                "has_pink": "pink" in cleaned,
                "has_blue": "blue" in cleaned,
                "has_black": "black" in cleaned,
            })
        except Exception as e:
            print(f"  ERROR: {e}")
        finally:
            if os.path.exists(proc_path):
                os.remove(proc_path)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY BY CLASS")
    print("=" * 70)
    for dx in ["MEL", "NV", "BCC", "BKL"]:
        class_results = [r for r in results if r["dx"] == dx]
        if not class_results:
            continue
        n = len(class_results)
        avg_count = sum(r["count"] for r in class_results) / n
        red_pct = sum(1 for r in class_results if r["has_red"]) / n * 100
        pink_pct = sum(1 for r in class_results if r["has_pink"]) / n * 100
        blue_pct = sum(1 for r in class_results if r["has_blue"]) / n * 100
        black_pct = sum(1 for r in class_results if r["has_black"]) / n * 100
        print(f"\n{dx} (n={n}):")
        print(f"  Avg colors: {avg_count:.1f}")
        print(f"  Red in raw: {red_pct:.0f}%")
        print(f"  Pink:  {pink_pct:.0f}%")
        print(f"  Blue:  {blue_pct:.0f}%")
        print(f"  Black: {black_pct:.0f}%")
        for r in class_results:
            print(f"    {r['image']}: {r['cleaned']}")


if __name__ == "__main__":
    run_colors_test()
