import os
import gc
import re
import json
import uuid
import time
import base64
from PIL import Image, ImageEnhance, ImageFilter
from llama_cpp import Llama
from llama_cpp.llama_chat_format import register_chat_format, ChatFormatterResponse

# Raw pass-through handler for PaliGemma base — no chat template wrapping
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

# --- CONFIGURATION ---
MODEL_DIR = "/home/project/models"
DATA_DIR = "/home/project/data/images"
# Phase 1 Specialist: MedGemma multimodal (testing bias reduction)
SPECIALIST_PATH = os.path.join(MODEL_DIR, "medgemma.gguf")
SPECIALIST_PROJ = os.path.join(MODEL_DIR, "medgemma-mmproj.gguf")
MEDGEMMA_PATH = os.path.join(MODEL_DIR, "medgemma.gguf")

# MedGemma format — direct clinical questions
OBS_QUERIES = {
    "Architecture": "What is the shape of this lesion? Answer with one word: round, oval, or irregular.",
    "Network": "Does this lesion have a pigment network? Answer with one word: yes or no.",
    "Structures": "What structures are visible in this lesion? Answer briefly: dots, globules, streaks, or none.",
    "Colors": "What colors are present in this lesion? Reply with only color names separated by commas, nothing else.",
}

def clean_response(val, key=None):
    """Strip markdown artifacts and trailing explanations from model output."""
    val = val.replace("*", "")
    val = val.split("\n")[0].strip()
    val = val.rstrip(".")
    val = re.sub(r"\s+", " ", val).strip()

    if key == "Structures":
        found = [s for s in ["dots", "globules", "streaks"] if s in val]
        val = ", ".join(found) if found else "none"

    if key == "Colors":
        known_colors = ["red", "white", "blue", "black", "brown", "yellow", "pink", "orange", "gray", "tan"]
        found = [c for c in known_colors if c in val]
        val = ", ".join(found) if found else "unknown"

    return val


def build_specialist_context(raw_obs):
    """Build structured clinical context from specialist observations."""
    return (
        f"Specialist observations:\n"
        f"- Shape (Architecture): {raw_obs['Architecture']}\n"
        f"- Pigment network: {raw_obs['Network']}\n"
        f"- Structures: {raw_obs['Structures']}\n"
        f"- Colors: {raw_obs['Colors']}"
    )


def run_specialist_queries(vlm, image_uri):
    """Run Phase 1 specialist queries on a single image and return cleaned observations."""
    raw_obs = {}
    vlm.reset()
    for key, q in OBS_QUERIES.items():
        res = vlm.create_chat_completion(
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
        val = res["choices"][0]["message"]["content"].strip().lower()
        raw_obs[key] = clean_response(val, key)
        print(f"[BACKEND] Specialist ({key}): {val}")
    return raw_obs


class DermPipeline:
    def _encode_image_base64(self, image_path: str) -> str:
        """Read image file and return a data URI for llama-cpp-python chat API."""
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def prepare_phase2_inputs(self, raw_obs, image_uri):
        """Stage input triplet (context, image, prompt) for visual MedGemma Phase 2."""
        context = build_specialist_context(raw_obs)
        prompt = (
            f"{context}\n\n"
            "Based on the specialist observations above and the dermoscopic image, "
            "provide a one-sentence clinical description of this lesion."
        )
        return {
            "context": context,
            "image_uri": image_uri,
            "prompt": prompt,
            "raw_obs": raw_obs,
        }

    def _prepare_image(self, filename: str) -> str:
        original_path = os.path.join(DATA_DIR, filename)
        # Unique input filename forces the CLIP projector to bypass cached embeddings
        temp_path = os.path.join(DATA_DIR, f"vlm_in_{uuid.uuid4().hex[:6]}.jpg")
        with Image.open(original_path) as img:
            img = img.convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
            # Standardizing contrast helps PaliGemma see edges better
            img = ImageEnhance.Contrast(img).enhance(1.4)
            img.save(temp_path, quality=90)
        return temp_path

    def process(self, filename: str):
        processed_path = self._prepare_image(filename)
        print(f"\n[BACKEND] >>> STARTING FRESH ANALYSIS: {filename}")
        
        try:
            # PHASE 1: Specialist (The Vision Model - MedGemma multimodal)
            # Re-initializing here is heavy, but essential to clear internal KV-cache
            vlm = Llama(
                model_path=SPECIALIST_PATH,
                clip_model_path=SPECIALIST_PROJ,
                n_ctx=2048,
                n_gpu_layers=-1,
                chat_format="medgemma-direct",
                verbose=False
            )
            
            # Encode the processed image as base64 data URI for the chat API
            image_uri = self._encode_image_base64(processed_path)
            raw_obs = run_specialist_queries(vlm, image_uri)
            
            # HARD RESET: Clean up memory immediately
            vlm.reset()
            del vlm
            gc.collect()
            time.sleep(0.5) # Brief pause for GPU VRAM deallocation

            # STAGING: Prepare input triplet for future visual Phase 2
            phase2_inputs = self.prepare_phase2_inputs(raw_obs, image_uri)
            print(f"[BACKEND] Staged Phase 2 inputs: context={len(phase2_inputs['context'])} chars, "
                  f"prompt={len(phase2_inputs['prompt'])} chars, "
                  f"image_uri={'valid' if phase2_inputs['image_uri'].startswith('data:image/') else 'INVALID'}")

            # PHASE 2: MedGemma (The Reasoning Model)
            reasoner = Llama(model_path=MEDGEMMA_PATH, n_ctx=2048, n_gpu_layers=-1, verbose=False)
            
            case_id = uuid.uuid4().hex[:4].upper()
            desc_prompt = (
                f"<start_of_turn>user\n"
                f"DATA: {raw_obs['Architecture']} shape, network: {raw_obs['Network']}, features: {raw_obs['Structures']}.\n"
                "In one sentence, describe the clinical appearance.<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )
            desc_res = reasoner.create_completion(prompt=desc_prompt, max_tokens=100)
            adj_desc = desc_res["choices"][0]["text"].strip()
            print(f"[BACKEND] Adjudicator Description: {adj_desc}")

            class_prompt = f"<start_of_turn>user\nDescription: {adj_desc}\nCode: MEL, NV, BCC, or BKL.<end_of_turn>\n<start_of_turn>model\nCODE: "
            class_res = reasoner.create_completion(prompt=class_prompt, max_tokens=10)
            raw_code = class_res["choices"][0]["text"].strip().upper()

            # PHASE 3: LOGIC GATE & OVERRIDE
            final_code = "UNKNOWN"
            for code in ["MEL", "NV", "BCC", "BKL"]:
                if code in raw_code: final_code = code; break
            
            # Strict Color Check
            color_text = raw_obs['Colors']
            # We filter out colors only if the model clearly detected them
            detected_colors = [c for c in ["red", "white", "blue", "black", "brown"] if c in color_text]
            
            # Risk Analysis
            has_blue_red = any(c in detected_colors for c in ["blue", "red"])
            print(f"[BACKEND] Logic Check - Colors: {detected_colors} | Risk: {'HIGH' if has_blue_red else 'NORMAL'}")

            # Only override if the Specialist actually found High Risk colors
            if has_blue_red or len(detected_colors) >= 4:
                final_code = "MEL"
                print("[BACKEND] Result: High Risk Overrule to MEL")
            elif final_code == "UNKNOWN":
                final_code = "NV"

            del reasoner
            gc.collect()

            report = (
                f"--- DERMATOLOGY AI ADJUDICATION ---\n"
                f"FINAL CLASSIFICATION: {final_code}\n\n"
                f"CLINICAL SUMMARY:\n{adj_desc}\n\n"
                f"SPECIALIST DATA:\n"
                f"- Architecture: {raw_obs['Architecture']}\n"
                f"- Colors: {raw_obs['Colors']}\n"
                f"------------------------------------"
            )
            return {"ai_code": final_code, "final_implication": report}
            
        finally:
            if os.path.exists(processed_path):
                os.remove(processed_path)
