import os
import re
import uuid
import base64
import hashlib
from PIL import Image, ImageEnhance
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

# --- MedGemma vision handler — subclasses Llava15 for proper CLIP embedding injection ---
class MedGemmaChatHandler(Llava15ChatHandler):
    # Gemma 3 turn-based template; image URLs are rendered into text then
    # replaced with mtmd media markers by the parent __call__.
    DEFAULT_SYSTEM_MESSAGE = None
    CHAT_FORMAT = (
        "{% for message in messages %}"
        "{% if message.role == 'user' %}"
        "<start_of_turn>user\n"
        "{% if message.content is string %}"
        "{{ message.content }}"
        "{% else %}"
        "{% for content in message.content %}"
        "{% if content.type == 'image_url' and content.image_url is string %}"
        "{{ content.image_url }}"
        "{% elif content.type == 'image_url' and content.image_url is mapping %}"
        "{{ content.image_url.url }}"
        "{% elif content.type == 'text' %}"
        "{{ content.text }}"
        "{% endif %}"
        "{% endfor %}"
        "{% endif %}"
        "<end_of_turn>\n"
        "{% endif %}"
        "{% if message.role == 'assistant' and message.content is not none %}"
        "<start_of_turn>model\n"
        "{{ message.content }}<end_of_turn>\n"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "<start_of_turn>model\n"
        "Answer:"
        "{% endif %}"
    )

# --- CONFIGURATION ---
MODEL_DIR = "/home/project/models"
DATA_DIR = "/home/project/data/images"
# Phase 1 Specialist: MedGemma multimodal (testing bias reduction)
SPECIALIST_PATH = os.path.join(MODEL_DIR, "medgemma.gguf")
SPECIALIST_PROJ = os.path.join(MODEL_DIR, "medgemma-mmproj.gguf")
MEDGEMMA_PATH = os.path.join(MODEL_DIR, "medgemma.gguf")

# MedGemma format — direct clinical questions (one per CLIP call)
OBS_QUERIES = {
    "Architecture": "What is the shape of this lesion? Answer with one word: round, oval, or irregular.",
    "Network": "Does this lesion have a pigment network? Answer with one word: yes or no.",
    "Structures": "What structures are visible in this lesion? Answer briefly: dots, globules, streaks, or none.",
    "Colors": (
        "What colors are present in this skin lesion? Check for: "
        "light brown, dark brown, black, blue-gray, white, pink. "
        "List only colors clearly visible, separated by commas."
    ),
    "Vessels": "What type of blood vessels are visible in this lesion? Answer briefly: arborizing, dotted, linear, or none.",
    "Regression": "Are regression structures or blue-white veil visible in this lesion? Answer with one word: yes or no.",
    "Keratosis": (
        "Look for small dark round plugs (comedo-like openings) or tiny bright white "
        "dots/cysts (milia-like cysts) anywhere in this lesion. Are any present? "
        "Answer with one word: yes or no."
    ),
    "Symmetry": "Is this lesion symmetric or asymmetric? Answer with one word: symmetric or asymmetric.",
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

    if key == "Vessels":
        found = [v for v in ["arborizing", "dotted", "linear"] if v in val]
        val = ", ".join(found) if found else "none"

    if key in ("Regression", "Keratosis"):
        val = "yes" if "yes" in val else "no"

    if key == "Symmetry":
        val = "asymmetric" if "asymmetric" in val else "symmetric"

    return val


def build_specialist_context(raw_obs):
    """Build structured clinical context from specialist observations."""
    lines = [
        f"Specialist observations:",
        f"- Shape (Architecture): {raw_obs['Architecture']}",
        f"- Pigment network: {raw_obs['Network']}",
        f"- Structures: {raw_obs['Structures']}",
        f"- Colors: {raw_obs['Colors']}",
    ]
    if "Vessels" in raw_obs:
        lines.append(f"- Vessels: {raw_obs['Vessels']}")
    if "Regression" in raw_obs:
        lines.append(f"- Regression/blue-white veil: {raw_obs['Regression']}")
    if "Keratosis" in raw_obs:
        lines.append(f"- Comedo-like openings/milia-like cysts: {raw_obs['Keratosis']}")
    if "Symmetry" in raw_obs:
        lines.append(f"- Symmetry: {raw_obs['Symmetry']}")
    return "\n".join(lines)


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
    for key, q in OBS_QUERIES.items():
        vlm.reset()
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
        temperature=0.5,
        repeat_penalty=1.5,
    )
    return res["choices"][0]["message"]["content"].strip()


def run_phase3_classification(vlm, phase3_inputs):
    """Run Phase 3: rule-based MEL/BCC/BKL + LLM fallback.

    Uses the same vision model for text-only prompts — the handler template
    handles `message.content is string` correctly (no CLIP processing).
    """
    raw_obs = phase3_inputs.get("raw_obs", {})
    arch = raw_obs.get("Architecture", "").lower()
    net = raw_obs.get("Network", "").lower()
    vessels = raw_obs.get("Vessels", "none").lower()
    regression = raw_obs.get("Regression", "no").lower()
    keratosis = raw_obs.get("Keratosis", "no").lower()
    symmetry = raw_obs.get("Symmetry", "symmetric").lower()
    colors = raw_obs.get("Colors", "brown").lower()
    color_list = [c.strip() for c in colors.split(",")]
    color_count = len(color_list)

    # 1. BKL: keratosis features detected
    if "yes" in keratosis:
        return "BKL", "RULE:keratosis_features"

    # 2a. BCC: arborizing vessels detected (now allowed back through filter)
    if "arborizing" in vessels:
        return "BCC", "RULE:arborizing_vessels"

    # 2b. BCC: pink color (vascular component, strong BCC signal)
    if "pink" in colors and "irregular" not in arch:
        return "BCC", "RULE:pink_vascular"

    # 3. MEL: require asymmetric + at least 2 other indicators
    #    (Previously: irregular + no_network — fired on 50/99!)
    mel_score = 0
    if "irregular" in arch:
        mel_score += 1
    if "no" in net:
        mel_score += 1
    if "yes" in regression:
        mel_score += 1
    if "asymmetric" in symmetry:
        mel_score += 1
    if "blue" in colors:
        mel_score += 1
    if color_count >= 3:
        mel_score += 1
    if "asymmetric" in symmetry and mel_score >= 3:
        return "MEL", f"RULE:mel_score_{mel_score}"

    # 4. LLM fallback — now includes MEL option
    vlm.reset()
    res = vlm.create_chat_completion(
        messages=[{
            "role": "user",
            "content": phase3_inputs["prompt"],
        }],
        max_tokens=16,
        temperature=0.2,
    )
    raw_output = res["choices"][0]["message"]["content"].strip().upper()

    for code in ["MEL", "BCC", "BKL"]:
        if code in raw_output:
            return code, raw_output

    return "NV", raw_output


class DermPipeline:
    def __init__(self):
        """Load the vision model once — reused across all images and phases."""
        print("[BACKEND] Loading MedGemma vision model (one-time)...")
        self._handler = MedGemmaChatHandler(clip_model_path=SPECIALIST_PROJ, verbose=False)
        self._vlm = Llama(
            model_path=SPECIALIST_PATH,
            n_ctx=2048,
            n_gpu_layers=-1,
            chat_handler=self._handler,
            verbose=False
        )
        print("[BACKEND] MedGemma vision model loaded and ready.")

    def _encode_image_base64(self, image_path: str) -> str:
        """Read image file and return a data URI for llama-cpp-python chat API."""
        with open(image_path, "rb") as f:
            raw = f.read()
        md5 = hashlib.md5(raw).hexdigest()
        print(f"[BACKEND] Image encoded: path={image_path}, size={len(raw)} bytes, md5={md5}")
        b64 = base64.b64encode(raw).decode("utf-8")
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
            "such as border definition, pattern regularity, symmetry, "
            "comedo-like openings (dark plugs), milia-like cysts (white dots), "
            "fissures, ridges, clod patterns, "
            "arborizing vessels, ulceration, or shiny white structures. "
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
        color_list = [c.strip() for c in raw_obs.get("Colors", "brown").split(",")]
        color_note = f"Color diversity: {len(color_list)} color(s) detected ({raw_obs.get('Colors', 'brown')})\n"
        prompt = (
            f"{context}\n\n"
            f"Clinical description:\n{cleaned_desc}\n\n"
            f"{color_note}\n"
            "Based on the clinical description, classify this lesion.\n"
            "- MEL: asymmetric shape, multiple colors, irregular dots/globules, "
            "regression structures, or blue-white veil\n"
            "- BCC: arborizing vessels, blue-grey ovoid nests, shiny white structures, "
            "or ulceration\n"
            "- BKL: comedo-like openings, milia-like cysts, moth-eaten border, "
            "clod pattern, fissures and ridges, or cerebriform pattern\n"
            "- NV: none of the above features\n\n"
            "Respond with exactly one code: MEL, BCC, BKL, or NV."
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
            # Encode the processed image as base64 data URI for the chat API
            image_uri = self._encode_image_base64(processed_path)

            # PHASE 1: Combined specialist observations (single CLIP-encoded call)
            raw_obs = run_specialist_queries(self._vlm, image_uri)

            # STAGING: Prepare input triplet for Phase 2
            phase2_inputs = self.prepare_phase2_inputs(raw_obs, image_uri)
            print(f"[BACKEND] Staged Phase 2 inputs: context={len(phase2_inputs['context'])} chars, "
                  f"prompt={len(phase2_inputs['prompt'])} chars, "
                  f"image_uri={'valid' if phase2_inputs['image_uri'].startswith('data:image/') else 'INVALID'}")

            # PHASE 2: Visual MedGemma description (reuse same model)
            adj_desc = run_phase2_description(self._vlm, phase2_inputs)
            print(f"[BACKEND] Phase 2 Description: {adj_desc}")

            # STAGING: Prepare Phase 3 inputs (cleans Phase 2 output)
            phase3_inputs = self.prepare_phase3_inputs(raw_obs, adj_desc)
            print(f"[BACKEND] Staged Phase 3 inputs: cleaned_desc={len(phase3_inputs['cleaned_description'])} chars")

            # PHASE 3: Classification (reuse vision model for text-only prompt)
            code, raw_class = run_phase3_classification(self._vlm, phase3_inputs)
            print(f"[BACKEND] Phase 3 Classification: {code} (raw: {raw_class})")

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
