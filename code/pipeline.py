import os
import gc
import re
import uuid
import time
import base64
from PIL import Image, ImageEnhance
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

# Phase 3 classification constants
VALID_CODES = ["MEL", "NV", "BCC", "BKL"]
REFUSAL_PHRASES = ["i cannot", "i'm sorry", "as an ai", "not able to", "inappropriate"]
SPECIAL_TOKENS = ["<start_of_turn>", "<end_of_turn>", "<eos>", "<pad>", "\u2581"]

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
        known_colors = ["white", "blue", "black", "brown", "yellow", "pink", "orange", "gray", "tan"]
        found = [c for c in known_colors if c in val]
        val = ", ".join(found) if found else "brown"

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


def clean_phase2_output(adj_desc):
    """Sentence-level cleaning of Phase 2 output.

    Removes sentences containing refusal phrases, gibberish (repeated patterns,
    special tokens, excessive non-ASCII, low alpha ratio), and normalizes whitespace.
    """
    if not adj_desc or not adj_desc.strip():
        return ""

    # Split into sentences (period/exclamation/question followed by space or end)
    sentences = re.split(r'(?<=[.!?])\s+', adj_desc.strip())
    cleaned = []

    for sent in sentences:
        lower = sent.lower()

        # Skip sentences with refusal phrases
        if any(phrase in lower for phrase in REFUSAL_PHRASES):
            continue

        # Skip sentences with special token leakage
        if any(token in sent for token in SPECIAL_TOKENS):
            continue

        # Skip sentences with repeated patterns (same 3+ char substring repeated 3+ times)
        has_repeat = False
        for length in range(3, min(len(sent) // 3 + 1, 20)):
            for start in range(len(sent) - length * 3 + 1):
                substr = sent[start:start + length]
                if substr * 3 in sent:
                    has_repeat = True
                    break
            if has_repeat:
                break
        if has_repeat:
            continue

        # Skip sentences with excessive non-ASCII
        non_ascii = sum(1 for c in sent if ord(c) > 127)
        if len(sent) > 5 and non_ascii > len(sent) * 0.3:
            continue

        # Skip sentences with very low alpha ratio
        alpha = sum(1 for c in sent if c.isalpha())
        if len(sent) > 5 and alpha < len(sent) * 0.4:
            continue

        cleaned.append(sent)

    result = " ".join(cleaned)
    # Normalize whitespace
    result = re.sub(r"\s+", " ", result).strip()
    # Scrub "red" color references — ubiquitous dermoscopic artifact that inflates color counts
    result = re.sub(r'\bred[- ]brown\b', 'brown', result, flags=re.IGNORECASE)
    result = re.sub(r'\breddish\b', 'brownish', result, flags=re.IGNORECASE)
    result = re.sub(r'\bred\b', 'brown', result, flags=re.IGNORECASE)
    return result


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
        cleaned = clean_response(val, key)
        raw_obs[key] = cleaned
        print(f"[BACKEND] Specialist ({key}): {val} -> {cleaned}")
    return raw_obs


def run_phase2_description(vlm, phase2_inputs):
    """Run Phase 2: visual MedGemma generates clinical description from image + context."""
    vlm.reset()
    res = vlm.create_chat_completion(
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": phase2_inputs["image_uri"]}},
                {"type": "text", "text": phase2_inputs["prompt"]}
            ]
        }],
        max_tokens=256,
        temperature=0.7,
        repeat_penalty=1.5,
    )
    return res["choices"][0]["message"]["content"].strip()


def run_phase3_classification(text_model, phase3_inputs):
    """Run Phase 3: rule-based MEL/NV + LLM-based BCC/BKL detection."""
    raw_obs = phase3_inputs.get("raw_obs", {})
    arch = raw_obs.get("Architecture", "").lower()
    net = raw_obs.get("Network", "").lower()

    # Rule-based MEL: requires irregular shape AND absent pigment network
    is_mel = "irregular" in arch and "no" in net

    # Ask LLM only for BCC/BKL detection from clinical description
    text_model.reset()
    res = text_model.create_chat_completion(
        messages=[{
            "role": "user",
            "content": phase3_inputs["prompt"],
        }],
        max_tokens=16,
        temperature=0.1,
    )
    raw_output = res["choices"][0]["message"]["content"].strip().upper()

    # BCC/BKL from LLM takes priority over rule-based MEL/NV
    for code in ["BCC", "BKL"]:
        if code in raw_output:
            return code, raw_output

    # Otherwise use rule-based MEL/NV
    if is_mel:
        return "MEL", f"RULE:irregular+no_network (LLM:{raw_output})"
    return "NV", raw_output


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
            "Review the dermoscopic image and compare each observation above (shape, pigment "
            "network, structures, colors) with what is visible. For each, state whether it "
            "is consistent or inconsistent with the image in one sentence. "
            "Then note any additional clearly visible dermoscopic features not mentioned above, "
            "such as border definition, pattern regularity, or symmetry. "
            "Be brief and factual. Only describe features you can clearly see. "
            "Do not speculate about diagnoses or features that may or may not be present. "
            "Do not mention red or reddish tones — these are common dermoscopic artifacts. "
            "Write in plain prose paragraphs only. Do not use tables or summary grids."
        )
        return {
            "context": context,
            "image_uri": image_uri,
            "prompt": prompt,
            "raw_obs": raw_obs,
        }

    def prepare_phase3_inputs(self, raw_obs, adj_desc):
        """Stage inputs for text-only Phase 3 classification.

        Cleans Phase 2 output, combines with Phase 1 context, and builds
        classification prompt.
        """
        cleaned_desc = clean_phase2_output(adj_desc)
        context = build_specialist_context(raw_obs)
        prompt = (
            f"{context}\n\n"
            f"Clinical description:\n{cleaned_desc}\n\n"
            "Does the clinical description indicate BCC or BKL?\n"
            "- BCC: arborizing vessels, blue-grey ovoid nests, shiny white structures, "
            "or ulceration\n"
            "- BKL: comedo-like openings, milia-like cysts, or moth-eaten border\n"
            "- NV: none of the above features\n\n"
            "Respond with exactly one code: BCC, BKL, or NV."
        )
        return {
            "context": context,
            "cleaned_description": cleaned_desc,
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

            # PHASE 2: Visual MedGemma (description generation)
            reasoner = Llama(
                model_path=MEDGEMMA_PATH,
                clip_model_path=SPECIALIST_PROJ,
                n_ctx=2048,
                n_gpu_layers=-1,
                chat_format="medgemma-direct",
                verbose=False
            )
            adj_desc = run_phase2_description(reasoner, phase2_inputs)
            print(f"[BACKEND] Phase 2 Description: {adj_desc}")

            del reasoner
            gc.collect()
            time.sleep(0.5)

            # STAGING: Prepare Phase 3 inputs (cleans Phase 2 output)
            phase3_inputs = self.prepare_phase3_inputs(raw_obs, adj_desc)
            print(f"[BACKEND] Staged Phase 3 inputs: cleaned_desc={len(phase3_inputs['cleaned_description'])} chars")

            # PHASE 3: Text-only MedGemma classification (no CLIP projector)
            text_model = Llama(
                model_path=MEDGEMMA_PATH,
                n_ctx=2048,
                n_gpu_layers=-1,
                chat_format="medgemma-direct",
                verbose=False,
            )
            code, raw_class = run_phase3_classification(text_model, phase3_inputs)
            print(f"[BACKEND] Phase 3 Classification: {code} (raw: {raw_class})")

            del text_model
            gc.collect()

            report = (
                f"--- DERMATOLOGY AI ADJUDICATION ---\n"
                f"FINAL CLASSIFICATION: {code}\n\n"
                f"CLINICAL SUMMARY:\n{adj_desc}\n\n"
                f"SPECIALIST DATA:\n"
                f"- Architecture: {raw_obs['Architecture']}\n"
                f"- Colors: {raw_obs['Colors']}\n"
                f"------------------------------------"
            )
            return {
                "ai_code": code,
                "adj_desc": adj_desc,
                "raw_obs": raw_obs,
                "final_implication": report,
            }
            
        finally:
            if os.path.exists(processed_path):
                os.remove(processed_path)
