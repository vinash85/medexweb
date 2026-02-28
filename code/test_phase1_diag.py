"""
Diagnostic: Try multiple prompt formats to find what PaliGemma responds to.
Run inside Docker: python3 /home/project/code/test_phase1_diag.py
"""
import os, base64, time
from PIL import Image, ImageEnhance
from llama_cpp import Llama

MODEL_DIR = "/home/project/models"
DATA_DIR = "/home/project/data/images"
SPECIALIST_PATH = os.path.join(MODEL_DIR, "specialist.gguf")
SPECIALIST_PROJ = os.path.join(MODEL_DIR, "specialist-mmproj.gguf")

# Pick first available image
img_file = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".jpg")])[0]
img_path = os.path.join(DATA_DIR, img_file)

# Preprocess
temp = img_path + ".diag.jpg"
with Image.open(img_path) as img:
    img = img.convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
    ImageEnhance.Contrast(img).enhance(1.4).save(temp, quality=90)

with open(temp, "rb") as f:
    uri = f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}"

# Test matrix: (chat_format, prompt_text)
tests = [
    # Default handler (llava-style) with various prompts
    (None, "describe en"),
    (None, "caption en"),
    (None, "answer en symmetric or asymmetric?"),
    (None, "What do you see?"),
    # chatml handler
    ("chatml", "describe en"),
    ("chatml", "answer en What colors are visible?"),
    # gemma handler
    ("gemma", "describe en"),
]

print(f"Image: {img_file}")
print("=" * 70)

for fmt, prompt in tests:
    label = fmt or "default"
    print(f"\n[{label}] prompt: '{prompt}'")

    try:
        kwargs = dict(
            model_path=SPECIALIST_PATH,
            clip_model_path=SPECIALIST_PROJ,
            n_ctx=1024,
            n_gpu_layers=-1,
            verbose=False
        )
        if fmt:
            kwargs["chat_format"] = fmt

        vlm = Llama(**kwargs)
        t0 = time.time()
        res = vlm.create_chat_completion(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": uri}},
                    {"type": "text", "text": prompt}
                ]
            }],
            max_tokens=64,
            temperature=0.3,
        )
        elapsed = time.time() - t0
        content = res["choices"][0]["message"]["content"]
        reason = res["choices"][0].get("finish_reason", "?")
        print(f"  response: '{content}'")
        print(f"  finish: {reason} | time: {elapsed:.1f}s")

        del vlm

    except Exception as e:
        print(f"  ERROR: {e}")

os.remove(temp)
print("\n" + "=" * 70)
print("DONE — check which format produced meaningful output")
