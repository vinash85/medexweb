"""
Phase 1 Attribute Validation Test — runs ALL specialist queries on sample images.

Tests what the model actually returns for every Phase 1 attribute across classes.
Produces per-image CSV + auto-generated attributes.md summary.

Run inside Docker:
    python3 /home/project/code/test_attributes.py
"""
import os
import sys
import csv
import time
from collections import Counter

from pipeline import (
    DATA_DIR, OBS_QUERIES, clean_response, DermPipeline,
)

TARGET_CLASSES = ["MEL", "NV", "BCC", "BKL"]
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
CSV_OUT = os.path.join(BASE_DIR, "data", "attribute_test_results.csv")
MD_OUT = os.path.join(BASE_DIR, "attributes.md")


def load_ground_truth():
    gt = {}
    lesion_attrs = {}
    csv_path = os.path.join(os.path.dirname(__file__), "DermsGemms.csv")
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            name = row["Image"].strip()
            gt[name] = row["Dx"].strip().upper()
            lesion_attrs[name] = row.get(" Lesion attributes", row.get("Lesion attributes", "")).strip()
    return gt, lesion_attrs


def pick_sample(gt):
    all_images = sorted([
        f for f in os.listdir(DATA_DIR)
        if f.lower().endswith(".jpg") and ".proc." not in f and "vlm_in_" not in f
    ])
    if not all_images:
        print(f"FATAL: No images in {DATA_DIR}")
        sys.exit(1)

    by_class = {c: [] for c in TARGET_CLASSES}
    for img in all_images:
        name = img.replace(".jpg", "")
        dx = gt.get(name, "?")
        if dx in by_class and len(by_class[dx]) < 5:
            by_class[dx].append(img)

    sample = []
    for dx in TARGET_CLASSES:
        for img in by_class[dx]:
            sample.append((img, dx))
    return sample


def run_queries(pipeline, sample, lesion_attrs):
    query_keys = list(OBS_QUERIES.keys())
    results = []

    for i, (img_file, dx) in enumerate(sample, 1):
        name = img_file.replace(".jpg", "")
        lesion_attr = lesion_attrs.get(name, "")
        print(f"\n[{i}/{len(sample)}] {img_file} (GT: {dx}, Attr: {lesion_attr})")

        proc_path = pipeline._prepare_image(img_file)
        image_uri = pipeline._encode_image_base64(proc_path)

        row = {"image": name, "dx": dx, "lesion_attr": lesion_attr}

        for key in query_keys:
            try:
                pipeline._vlm.reset()
                q = OBS_QUERIES[key]
                res = pipeline._vlm.create_chat_completion(
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_uri}},
                            {"type": "text", "text": q}
                        ]
                    }],
                    max_tokens=32,
                    temperature=0.7,
                    repeat_penalty=1.5,
                )
                raw_val = res["choices"][0]["message"]["content"].strip().lower()
                cleaned = clean_response(raw_val, key)
                row[f"{key}_raw"] = raw_val
                row[key] = cleaned
                print(f"  {key}: {raw_val} -> {cleaned}")
            except Exception as e:
                row[f"{key}_raw"] = f"ERROR: {e}"
                row[key] = "ERROR"
                print(f"  {key}: ERROR {e}")

        if os.path.exists(proc_path):
            os.remove(proc_path)

        results.append(row)

    return results


def save_csv(results, query_keys):
    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    fieldnames = ["image", "dx", "lesion_attr"] + query_keys + [f"{k}_raw" for k in query_keys]
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in results:
            w.writerow(row)
    print(f"\nCSV saved: {CSV_OUT}")


def generate_md(results, query_keys):
    lines = []
    lines.append("# Phase 1 Attribute Test Results")
    lines.append("")
    lines.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Images tested**: {len(results)} (5 per class)")
    lines.append(f"**Queries**: {', '.join(query_keys)}")
    lines.append("")

    # Prompts used
    lines.append("## Prompts")
    lines.append("")
    for key in query_keys:
        prompt = OBS_QUERIES[key]
        if isinstance(prompt, tuple):
            prompt = " ".join(prompt)
        lines.append(f"**{key}**: {prompt}")
        lines.append("")

    # Per-image table
    lines.append("## Per-Image Results")
    lines.append("")
    header = "| Image | GT | Lesion Attr | " + " | ".join(query_keys) + " |"
    sep = "|" + "---|" * (3 + len(query_keys))
    lines.append(header)
    lines.append(sep)
    for r in results:
        attr_short = r["lesion_attr"][:25] + "..." if len(r["lesion_attr"]) > 25 else r["lesion_attr"]
        cols = [r["image"], r["dx"], attr_short]
        for key in query_keys:
            cols.append(r.get(key, "?"))
        lines.append("| " + " | ".join(cols) + " |")
    lines.append("")

    # Per-attribute summary
    lines.append("## Attribute Distribution by Class")
    lines.append("")

    for key in query_keys:
        lines.append(f"### {key}")
        lines.append("")

        # Collect all unique values across classes
        all_vals = set()
        class_counters = {}
        for dx in TARGET_CLASSES:
            class_rows = [r for r in results if r["dx"] == dx]
            counter = Counter(r[key] for r in class_rows)
            class_counters[dx] = (counter, len(class_rows))
            all_vals.update(counter.keys())

        # For multi-value fields (Colors, Structures, Vessels), also do component-level analysis
        multi_value_keys = ["Colors", "Structures", "Vessels"]
        is_multi = key in multi_value_keys

        # Value distribution table
        sorted_vals = sorted(all_vals)
        header = "| Value | " + " | ".join(f"{dx} (n={class_counters[dx][1]})" for dx in TARGET_CLASSES) + " |"
        sep = "|" + "---|" * (1 + len(TARGET_CLASSES))
        lines.append(header)
        lines.append(sep)
        for val in sorted_vals:
            cols = [val]
            for dx in TARGET_CLASSES:
                counter, n = class_counters[dx]
                cnt = counter.get(val, 0)
                pct = cnt / n * 100 if n > 0 else 0
                if pct >= 50:
                    cols.append(f"**{cnt}/{n} ({pct:.0f}%)**")
                else:
                    cols.append(f"{cnt}/{n} ({pct:.0f}%)")
            lines.append("| " + " | ".join(cols) + " |")
        lines.append("")

        # Component-level analysis for multi-value fields
        if is_multi:
            lines.append(f"#### {key} — Individual Component Analysis")
            lines.append("")

            # Extract individual components
            all_components = set()
            class_component_counts = {}
            for dx in TARGET_CLASSES:
                class_rows = [r for r in results if r["dx"] == dx]
                comp_counter = Counter()
                for r in class_rows:
                    val = r.get(key, "")
                    if val and val != "none" and val != "ERROR":
                        for part in val.split(","):
                            part = part.strip()
                            if part:
                                comp_counter[part] += 1
                                all_components.add(part)
                class_component_counts[dx] = (comp_counter, len(class_rows))

            if all_components:
                sorted_comps = sorted(all_components)
                header = "| Component | " + " | ".join(f"{dx} (n={class_component_counts[dx][1]})" for dx in TARGET_CLASSES) + " |"
                sep = "|" + "---|" * (1 + len(TARGET_CLASSES))
                lines.append(header)
                lines.append(sep)
                for comp in sorted_comps:
                    cols = [comp]
                    for dx in TARGET_CLASSES:
                        counter, n = class_component_counts[dx]
                        cnt = counter.get(comp, 0)
                        pct = cnt / n * 100 if n > 0 else 0
                        if pct >= 50:
                            cols.append(f"**{cnt}/{n} ({pct:.0f}%)**")
                        else:
                            cols.append(f"{cnt}/{n} ({pct:.0f}%)")
                    lines.append("| " + " | ".join(cols) + " |")
                lines.append("")

    # Raw vs Cleaned comparison
    lines.append("## Raw vs Cleaned Comparison")
    lines.append("")
    lines.append("Cases where raw output differs significantly from cleaned output:")
    lines.append("")
    diff_count = 0
    for r in results:
        for key in query_keys:
            raw = r.get(f"{key}_raw", "")
            cleaned = r.get(key, "")
            if raw != cleaned and "ERROR" not in raw:
                lines.append(f"- **{r['image']}** ({r['dx']}) `{key}`: `{raw}` -> `{cleaned}`")
                diff_count += 1
    if diff_count == 0:
        lines.append("None — all raw outputs matched cleaned outputs.")
    lines.append("")

    md_content = "\n".join(lines)
    with open(MD_OUT, "w") as f:
        f.write(md_content)
    print(f"Markdown saved: {MD_OUT}")


def run_attributes_test():
    gt, lesion_attrs = load_ground_truth()
    sample = pick_sample(gt)
    query_keys = list(OBS_QUERIES.keys())

    total = len(sample)
    print("=" * 70)
    print(f"PHASE 1 ATTRIBUTE VALIDATION — {total} images x {len(query_keys)} queries")
    print("=" * 70)

    pipeline = DermPipeline()

    t0 = time.time()
    results = run_queries(pipeline, sample, lesion_attrs)
    elapsed = time.time() - t0

    print(f"\nCompleted in {elapsed:.1f}s ({elapsed/total:.1f}s per image)")

    save_csv(results, query_keys)
    generate_md(results, query_keys)

    print(f"\nOutputs:")
    print(f"  CSV: {CSV_OUT}")
    print(f"  MD:  {MD_OUT}")


if __name__ == "__main__":
    run_attributes_test()
