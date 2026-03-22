import os
import re
import io
import uuid
import base64
import hashlib
from PIL import Image, ImageEnhance
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

try:
    import ollama as _ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

# =============================================================================
# MedGemma Vision Handler
# =============================================================================

class MedGemmaChatHandler(Llava15ChatHandler):
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

# =============================================================================
# Configuration
# =============================================================================

MODEL_DIR = "/home/project/models"
DATA_DIR = "/home/project/data/images"
SPECIALIST_PATH = os.path.join(MODEL_DIR, "medgemma.gguf")
SPECIALIST_PROJ = os.path.join(MODEL_DIR, "medgemma-mmproj.gguf")

VALID_CODES = ["MEL", "NV", "BCC", "BKL", "AKIEC", "DF"]
REFUSAL_PHRASES = ["i cannot", "i'm sorry", "as an ai", "not able to", "inappropriate"]
SPECIAL_TOKENS = ["<start_of_turn>", "<end_of_turn>", "<eos>", "<pad>", "\u2581"]

# Phase 1: Ollama model registry
OLLAMA_MODELS = [
    {"name": "Llama-3.2-Vision", "ollama_model": "llama3.2-vision:11b",
     "temperature": 0.7, "max_tokens": 512},
    {"name": "Qwen2.5-VL", "ollama_model": "qwen2.5vl:7b",
     "temperature": 0.7, "max_tokens": 512},
    {"name": "LLaVA-OneVision", "ollama_model": "llava:13b",
     "temperature": 0.7, "max_tokens": 512},
]

# =============================================================================
# CoT Attribute Queries (7 dermoscopic attributes)
# =============================================================================

SYSTEM_PREFIX = (
    "You are a computer vision expert specializing in pattern recognition "
    "and spatial analysis of close-up surface images. "
    "Analyze the image strictly based on visual patterns you observe. "
    "Do not assume any medical or domain context.\n\n"
)

COT_QUERIES = {
    "Pigment_Network": {
        "prompt": SYSTEM_PREFIX + (
            "Analyze the line-network pattern in this image step by step.\n\n"
            "Step 1 — Grid Detection: Identify if there is a recurring lattice or "
            "honeycomb pattern of intersecting dark lines across the surface.\n\n"
            "Step 2 — Mesh Consistency: Are the holes in the grid uniform in size, "
            "or do they vary significantly (meshes becoming larger or distorted)?\n\n"
            "Step 3 — Line Morphology: Describe the lines of the grid. Are they thin "
            "and delicate, or thick, dark, and smudged?\n\n"
            "After completing all steps, state your final answer on a new line as:\n"
            "LABEL: <one of: absent, typical, atypical>"
        ),
        "labels": ["absent", "typical", "atypical"],
    },
    "Dots_Globules": {
        "prompt": SYSTEM_PREFIX + (
            "Analyze circular point features in this image step by step.\n\n"
            "Step 1 — Primitive Count: Locate all small, circumscribed circular "
            "primitives within the region of interest.\n\n"
            "Step 2 — Size Sorting: Differentiate between points (pinprick size) "
            "and globes (larger, distinct circles).\n\n"
            "Step 3 — Spatial Map: Are these circles concentrated in the center of "
            "the region, or scattered randomly at the periphery/edge?\n\n"
            "After completing all steps, state your final answer on a new line as:\n"
            "LABEL: <one of: absent, regular, irregular>"
        ),
        "labels": ["absent", "regular", "irregular"],
    },
    "Vascular_Structures": {
        "prompt": SYSTEM_PREFIX + (
            "Analyze linear red features in this image step by step.\n\n"
            "Step 1 — Color Isolation: Filter for red or pinkish tones within the "
            "region. Identify if these appear as dots, curves, or branches.\n\n"
            "Step 2 — Path Analysis: Describe the shape of the red features. Are they "
            "C-shaped (comma-like), tree-like (branching), or twisted loops?\n\n"
            "Step 3 — Distribution: Are the red structures confined to a specific "
            "zone or distributed throughout the region?\n\n"
            "After completing all steps, state your final answer on a new line as:\n"
            "LABEL: <one of: absent, comma, arborizing, dotted, polymorphous>"
        ),
        "labels": ["absent", "comma", "arborizing", "dotted", "polymorphous"],
    },
    "Blue_White_Structures": {
        "prompt": SYSTEM_PREFIX + (
            "Analyze blue and white color regions in this image step by step.\n\n"
            "Step 1 — Spectral Scan: Identify areas with a steel-blue or gray-blue "
            "hue anywhere in the image.\n\n"
            "Step 2 — Opacity Check: Does this area look like it has a frosted glass "
            "or ground glass white film overlaying the blue pigment?\n\n"
            "Step 3 — Border Definition: Is the blue area well-defined with sharp "
            "edges, or does it have blurry, ill-defined margins?\n\n"
            "After completing all steps, state your final answer on a new line as:\n"
            "LABEL: <one of: absent, blue_white_veil, regression_peppering>"
        ),
        "labels": ["absent", "blue_white_veil", "regression_peppering"],
    },
    "Streaks": {
        "prompt": SYSTEM_PREFIX + (
            "Analyze radial projections in this image step by step.\n\n"
            "Step 1 — Edge Detection: Examine the outermost boundary of the dark "
            "region in the image.\n\n"
            "Step 2 — Directional Mapping: Identify any linear projections or "
            "spokes that emanate outward from the center like a starburst pattern.\n\n"
            "Step 3 — Symmetry: Are these projections appearing around the entire "
            "circumference (symmetric/starburst), or only in one focal area "
            "(asymmetric/focal)?\n\n"
            "After completing all steps, state your final answer on a new line as:\n"
            "LABEL: <one of: absent, regular, irregular>"
        ),
        "labels": ["absent", "regular", "irregular"],
    },
    "Milia_Cysts": {
        "prompt": SYSTEM_PREFIX + (
            "Analyze bright sphere features in this image step by step.\n\n"
            "Step 1 — Contrast Peak: Identify small, distinct ivory or bright white "
            "circular structures within the region.\n\n"
            "Step 2 — Shape Check: Are they perfectly round small globes (starry), "
            "or larger and cloudy/opaque clusters?\n\n"
            "Step 3 — Artifact Filter: Differentiate from surface reflections or "
            "bubbles which usually have a specular (brightest point) highlight. "
            "True bright spheres are embedded within the surface texture.\n\n"
            "After completing all steps, state your final answer on a new line as:\n"
            "LABEL: <one of: absent, starry, cloudy>"
        ),
        "labels": ["absent", "starry", "cloudy"],
    },
    "Blotches": {
        "prompt": SYSTEM_PREFIX + (
            "Analyze structureless dark masses in this image step by step.\n\n"
            "Step 1 — Texture Density: Locate any dark area where the underlying "
            "pattern (like a grid or dots) is completely obscured by a solid mass "
            "of dark pigment.\n\n"
            "Step 2 — Light Transmission: Can you see through the dark area to any "
            "underlying structure, or is it 100% opaque and structureless?\n\n"
            "After completing all steps, state your final answer on a new line as:\n"
            "LABEL: <one of: absent, regular, irregular>"
        ),
        "labels": ["absent", "regular", "irregular"],
    },
}

ATTRIBUTE_KEYS = list(COT_QUERIES.keys())

# CoT label -> benchmark label mapping
COT_TO_BENCHMARK = {
    "Pigment_Network": {
        "typical": "present_uniform", "atypical": "present_irregular", "absent": "absent",
    },
    "Dots_Globules": {
        "regular": "present_central", "irregular": "present_peripheral", "absent": "absent",
    },
    "Vascular_Structures": {
        "comma": "other_vascular", "arborizing": "arborizing",
        "dotted": "dotted_glomerular", "polymorphous": "other_vascular", "absent": "absent",
    },
    "Blue_White_Structures": {
        "blue_white_veil": "blue_white_veil",
        "regression_peppering": "present_unspecified", "absent": "absent",
    },
    "Streaks": {
        "regular": "present", "irregular": "present", "absent": "absent",
    },
    "Milia_Cysts": {
        "starry": "milia_only", "cloudy": "comedo_only", "absent": "absent",
    },
    "Blotches": {
        "regular": "symmetric", "irregular": "asymmetric", "absent": "absent",
    },
}

# Phase 3: Lesion class definitions (dermoscopic criteria)
CLASS_DEFINITIONS = {
    "MEL": (
        "Melanoma (MEL): atypical pigment network with thick irregular lines, "
        "irregular dots/globules at periphery, blue-white veil, "
        "irregular streaks/pseudopods, regression structures (peppering), "
        "asymmetric blotches, polymorphous vessels (dotted + linear irregular)"
    ),
    "NV": (
        "Melanocytic Nevus (NV): typical/reticular pigment network with thin regular lines, "
        "regular dots/globules concentrated centrally, symmetric blotches, "
        "homogeneous brown pattern, no blue-white veil, no streaks"
    ),
    "BCC": (
        "Basal Cell Carcinoma (BCC): arborizing (tree-like branching) vessels, "
        "blue-grey ovoid nests, shiny white structures (blotches/strands), "
        "ulceration, absence of pigment network, leaf-like structures"
    ),
    "BKL": (
        "Benign Keratosis (BKL): comedo-like openings (dark plugs), "
        "milia-like cysts (white/yellow beads), gyri and ridges (cerebriform pattern), "
        "moth-eaten border, clod pattern, fissures"
    ),
    "AKIEC": (
        "Actinic Keratosis / Intraepithelial Carcinoma (AKIEC): surface scale/crust, "
        "radial dots, white clods, rosettes, "
        "glomerular/dotted vessels, erythematous base"
    ),
    "DF": (
        "Dermatofibroma (DF): central white scar-like patch, "
        "delicate peripheral pigment network (fading at edges), "
        "ring-like globules at periphery"
    ),
}

# CoT attribute -> UI display key mapping
ATTR_TO_UI_KEY = {
    "Pigment_Network": "Network",
    "Dots_Globules": "Structures",
    "Streaks": "Architecture",
    "Vascular_Structures": "Vessels",
    "Blue_White_Structures": "Regression",
    "Milia_Cysts": "Keratosis",
    "Blotches": "Colors",
}

# =============================================================================
# CoT Label Extraction (from test_ollama_cot_benchmark.py)
# =============================================================================

def extract_cot_label(raw, attr_key):
    """Extract the final label from a CoT response."""
    valid_labels = COT_QUERIES[attr_key]["labels"]
    text = raw.strip()

    # Primary: find explicit LABEL: line
    label_match = re.search(r"LABEL:\s*(.+)", text, re.IGNORECASE)
    if label_match:
        label_text = label_match.group(1).strip().lower()
        label_text = re.sub(r"[*\"`'_\[\](){}]", "", label_text).strip()
        for lbl in valid_labels:
            if lbl == label_text:
                return lbl
        for lbl in valid_labels:
            if lbl in label_text:
                return lbl

    # Fallback: scan last 3 lines for any valid label keyword
    last_lines = "\n".join(text.split("\n")[-3:]).lower()
    last_lines = re.sub(r"[*\"`'_\[\](){}]", "", last_lines)
    for lbl in valid_labels:
        if lbl in last_lines:
            return lbl

    # Attribute-specific keyword fallback on full text
    return _keyword_fallback(text.lower(), attr_key, valid_labels)


def _keyword_fallback(text, attr_key, valid_labels):
    """Last-resort keyword matching when LABEL: line is missing."""
    if attr_key == "Pigment_Network":
        if "atypical" in text or "irregular" in text or "distort" in text:
            return "atypical"
        if "typical" in text or "uniform" in text or "regular" in text or "delicate" in text:
            return "typical"
        if "absent" in text or "no grid" in text or "no lattice" in text or "no network" in text:
            return "absent"
        if "honeycomb" in text or "lattice" in text or "grid" in text or "network" in text:
            return "typical"

    elif attr_key == "Dots_Globules":
        if "irregular" in text or "scatter" in text or "peripher" in text or "random" in text:
            return "irregular"
        if "regular" in text or "central" in text or "homogen" in text or "uniform" in text:
            return "regular"
        if "absent" in text or "no dot" in text or "no glob" in text or "none" in text:
            return "absent"
        if "dot" in text or "glob" in text or "circle" in text or "point" in text:
            return "regular"

    elif attr_key == "Vascular_Structures":
        if "arboriz" in text or "branch" in text or "tree" in text:
            return "arborizing"
        if "comma" in text or "c-shape" in text or "curved" in text:
            return "comma"
        if "dotted" in text or "glomerul" in text or "punctate" in text or "loop" in text:
            return "dotted"
        if "polymorphous" in text or "multiple" in text or "mixed" in text:
            return "polymorphous"
        if "absent" in text or "no vessel" in text or "no vascular" in text or "none" in text:
            return "absent"
        if "vessel" in text or "vascular" in text or "red" in text:
            return "dotted"

    elif attr_key == "Blue_White_Structures":
        if "regression" in text or "pepper" in text or "granular" in text:
            return "regression_peppering"
        if "veil" in text or ("blue" in text and "white" in text):
            return "blue_white_veil"
        if "absent" in text or "no blue" in text or "none" in text:
            return "absent"
        if "blue" in text or "steel" in text or "gray" in text:
            return "blue_white_veil"

    elif attr_key == "Streaks":
        if "irregular" in text or "focal" in text or "asymmetr" in text:
            return "irregular"
        if "regular" in text or "starburst" in text or "symmetr" in text or "circumferen" in text:
            return "regular"
        if "absent" in text or "no streak" in text or "no projection" in text or "none" in text:
            return "absent"
        if "streak" in text or "radial" in text or "spoke" in text or "pseudopod" in text:
            return "irregular"

    elif attr_key == "Milia_Cysts":
        if "cloudy" in text or "opaque" in text or "large" in text or "cluster" in text:
            return "cloudy"
        if "starry" in text or "small" in text or "distinct" in text or "round" in text:
            return "starry"
        if "absent" in text or "no milia" in text or "none" in text or "no bright" in text:
            return "absent"
        if "milia" in text or "ivory" in text or "white" in text or "bead" in text:
            return "starry"

    elif attr_key == "Blotches":
        if "irregular" in text or "asymmetr" in text or "eccentric" in text:
            return "irregular"
        if "regular" in text or "symmetr" in text or "central" in text or "uniform" in text:
            return "regular"
        if "absent" in text or "no blotch" in text or "none" in text or "no dark" in text:
            return "absent"
        if "blotch" in text or "opaque" in text or "structureless" in text or "solid" in text:
            return "irregular"

    return "unclear"


def map_to_benchmark_label(attr_key, cot_label):
    """Convert a CoT prompt label to the original benchmark label."""
    mapping = COT_TO_BENCHMARK.get(attr_key, {})
    return mapping.get(cot_label, cot_label)


# =============================================================================
# Output Cleaning (reused from master pipeline)
# =============================================================================

def clean_phase2_output(text):
    """Sentence-level cleaning: remove refusals, gibberish, special tokens."""
    if not text or not text.strip():
        return ""

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    cleaned = []

    for sent in sentences:
        lower = sent.lower()
        if any(phrase in lower for phrase in REFUSAL_PHRASES):
            continue
        if any(token in sent for token in SPECIAL_TOKENS):
            continue

        # Skip repeated patterns
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

        non_ascii = sum(1 for c in sent if ord(c) > 127)
        if len(sent) > 5 and non_ascii > len(sent) * 0.3:
            continue
        alpha = sum(1 for c in sent if c.isalpha())
        if len(sent) > 5 and alpha < len(sent) * 0.4:
            continue

        cleaned.append(sent)

    result = " ".join(cleaned)
    result = re.sub(r"\s+", " ", result).strip()
    result = re.sub(r'\bred[- ]brown\b', 'brown', result, flags=re.IGNORECASE)
    result = re.sub(r'\breddish\b', 'brownish', result, flags=re.IGNORECASE)
    result = re.sub(r'\bred\b', 'brown', result, flags=re.IGNORECASE)
    return result


# =============================================================================
# Image Preparation Utilities
# =============================================================================

def prepare_image(filepath):
    """Resize to 224x224, enhance contrast, save to temp JPEG file."""
    temp_path = os.path.join(DATA_DIR, f"vlm_in_{uuid.uuid4().hex[:6]}.jpg")
    with Image.open(filepath) as img:
        img = img.convert("RGB").resize((448, 448), Image.Resampling.LANCZOS)
        img = ImageEnhance.Contrast(img).enhance(1.4)
        img.save(temp_path, quality=90)
    return temp_path


def encode_image_data_uri(image_path):
    """Read image and return data:image/jpeg;base64,... URI for llama-cpp-python."""
    with open(image_path, "rb") as f:
        raw = f.read()
    md5 = hashlib.md5(raw).hexdigest()
    print(f"[BACKEND] Image encoded: path={image_path}, size={len(raw)} bytes, md5={md5}")
    b64 = base64.b64encode(raw).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def encode_image_base64_raw(image_path):
    """Read image and return raw base64 string for Ollama."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# =============================================================================
# Ollama Utilities
# =============================================================================

def check_ollama_available():
    """Returns True if Ollama server is reachable."""
    if not HAS_OLLAMA:
        return False
    try:
        _ollama.list()
        return True
    except Exception:
        return False


def ensure_model_available(ollama_model):
    """Check if model is locally available; pull if missing."""
    try:
        models = _ollama.list()
        local_names = []
        for m in models.get("models", []):
            local_names.append(m.get("name", ""))
            local_names.append(m.get("name", "").split(":")[0])
        model_base = ollama_model.split(":")[0]
        if ollama_model in local_names or model_base in local_names:
            return True
    except Exception as e:
        print(f"  [WARN] Could not list models ({e}), attempting pull...")

    print(f"  [INFO] Pulling '{ollama_model}'...")
    try:
        current_digest = ""
        for progress in _ollama.pull(ollama_model, stream=True):
            digest = progress.get("digest", "")
            status = progress.get("status", "")
            total = progress.get("total") or 0
            completed = progress.get("completed") or 0
            if digest and digest != current_digest:
                current_digest = digest
                print(f"\n    [{status}]", end="", flush=True)
            elif total > 0:
                pct = 100.0 * completed / total
                print(f"\r    [{status}] {pct:.0f}%", end="", flush=True)
            else:
                print(f"\r    [{status}]", end="", flush=True)
        print()
        return True
    except Exception as e:
        print(f"  [ERROR] Pull failed for '{ollama_model}': {e}")
        return False


def query_ollama(ollama_model, image_b64, prompt, temperature=0.7, max_tokens=512):
    """Send a vision query to Ollama and return the response text."""
    response = _ollama.chat(
        model=ollama_model,
        messages=[{
            "role": "user",
            "content": prompt,
            "images": [image_b64],
        }],
        options={
            "temperature": temperature,
            "num_predict": max_tokens,
            "repeat_penalty": 1.5,
        },
    )
    return response["message"]["content"].strip()


# =============================================================================
# Phase 1: Multi-VLM CoT Attribute Extraction (raw outputs only)
# =============================================================================

def run_phase1_multi_vlm(image_b64):
    """Run 7 CoT attribute queries across 3 Ollama VLMs.

    Returns raw per-model labels — no concordance computed here.
    Concordance/discordance is determined by MedGemma in Phase 2.
    """
    per_model = {}
    models_used = []
    models_failed = []

    for cfg in OLLAMA_MODELS:
        model_name = cfg["name"]
        ollama_model = cfg["ollama_model"]
        print(f"\n[BACKEND] Phase 1 — Loading {model_name} ({ollama_model})...")

        if not ensure_model_available(ollama_model):
            print(f"[BACKEND]   SKIP: {model_name} not available")
            models_failed.append(model_name)
            continue

        models_used.append(model_name)
        per_model[model_name] = {}

        for attr_key, attr_cfg in COT_QUERIES.items():
            try:
                raw_answer = query_ollama(
                    ollama_model, image_b64, attr_cfg["prompt"],
                    temperature=cfg["temperature"], max_tokens=cfg["max_tokens"],
                )
                cot_label = extract_cot_label(raw_answer, attr_key)
                benchmark_label = map_to_benchmark_label(attr_key, cot_label)
                per_model[model_name][attr_key] = {
                    "raw": raw_answer,
                    "cot_label": cot_label,
                    "benchmark_label": benchmark_label,
                }
                print(f"[BACKEND]   {model_name}/{attr_key}: {cot_label}")
            except Exception as e:
                print(f"[BACKEND]   ERROR {model_name}/{attr_key}: {e}")
                per_model[model_name][attr_key] = {
                    "raw": f"ERROR: {e}",
                    "cot_label": "ERROR",
                    "benchmark_label": "ERROR",
                }

    return {
        "per_model": per_model,
        "models_used": models_used,
        "models_failed": models_failed,
    }


# =============================================================================
# Phase 2: MedGemma Vision CoT Review
# =============================================================================

ATTR_DISPLAY_NAMES = {
    "Pigment_Network": "Pigment Network (line-grid pattern)",
    "Dots_Globules": "Dots and Globules (circular features)",
    "Vascular_Structures": "Vascular Structures (red linear features)",
    "Blue_White_Structures": "Blue-White Structures (spectral features)",
    "Streaks": "Streaks (radial projections)",
    "Milia_Cysts": "Milia-like Cysts (bright spheres)",
    "Blotches": "Blotches (structureless dark masses)",
}


def _format_phase1_labels(per_model, attr_key):
    """Format Phase 1 model labels for a single attribute as a readable string."""
    parts = []
    for model_name, attrs in per_model.items():
        if attr_key in attrs:
            label = attrs[attr_key]["cot_label"]
            parts.append(f"{model_name}: {label}")
    return ", ".join(parts) if parts else "no Phase 1 data"


def build_phase2_prompt(attr_key, per_model):
    """Build per-attribute Phase 2 prompt.

    MedGemma sees the Phase 1 answers and must:
    1. Perform its own CoT analysis on the image
    2. Compare its finding with what Phase 1 models said
    3. Report CONCORDANCE or DISCORDANCE with Phase 1
    """
    attr_cfg = COT_QUERIES[attr_key]
    display_name = ATTR_DISPLAY_NAMES.get(attr_key, attr_key)

    lines = [f"Three vision models previously analyzed this dermoscopy image for {display_name}.\n"]
    lines.append("Their assessments:")
    for model_name, attrs in per_model.items():
        if attr_key in attrs:
            label = attrs[attr_key]["cot_label"]
            lines.append(f"- {model_name}: {label}")

    lines.append(
        "\nNow examine the image yourself using the same analysis framework.\n"
    )
    lines.append(attr_cfg["prompt"])
    lines.append(
        f"\nAfter your own analysis, compare your finding with the Phase 1 models above "
        f"and provide your clinical judgment.\n\n"
        f"End with exactly these lines:\n"
        f"ASSESSMENT: <your label from: {', '.join(attr_cfg['labels'])}>\n"
        f"CONCORDANCE: <concordant or discordant>\n"
        f"JUDGMENT: <your clinical opinion — what you observe in the image, why you "
        f"agree or disagree with Phase 1, what this feature suggests about the lesion, "
        f"and any diagnostic significance. 2-3 sentences.>"
    )

    return "\n".join(lines)


def run_phase2_medgemma_review(vlm, image_uri, phase1_result):
    """MedGemma vision CoT review with Phase 1 answers as context.

    For each attribute, MedGemma:
    - Sees the image + Phase 1 model answers
    - Performs its own CoT analysis
    - Reports its own label
    - Reports CONCORDANCE or DISCORDANCE with Phase 1
    """
    per_attribute = {}
    summary_lines = []

    for attr_key in ATTRIBUTE_KEYS:
        vlm.reset()
        prompt = build_phase2_prompt(attr_key, phase1_result["per_model"])

        try:
            res = vlm.create_chat_completion(
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_uri}},
                        {"type": "text", "text": prompt},
                    ]
                }],
                max_tokens=512,
                temperature=0.5,
                repeat_penalty=1.5,
            )
            raw = res["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[BACKEND] Phase 2 ERROR on {attr_key}: {e}")
            raw = f"ERROR: {e}"

        raw_cleaned = clean_phase2_output(raw)

        # Extract MedGemma's own label
        label = "unclear"
        assess_match = re.search(r"ASSESSMENT:\s*(.+)", raw, re.IGNORECASE)
        if assess_match:
            assess_text = assess_match.group(1).strip().lower()
            assess_text = re.sub(r"[*\"`'_\[\](){}]", "", assess_text).strip()
            for lbl in COT_QUERIES[attr_key]["labels"]:
                if lbl in assess_text:
                    label = lbl
                    break
        if label == "unclear":
            label = extract_cot_label(raw, attr_key)

        # Extract concordance/discordance judgment
        concordance = "unknown"
        conc_match = re.search(r"CONCORDANCE:\s*(concordant|discordant)", raw, re.IGNORECASE)
        if conc_match:
            concordance = conc_match.group(1).lower()
        else:
            # Infer from label comparison: if MedGemma's label matches any Phase 1 model
            p1_labels = [
                attrs[attr_key]["cot_label"]
                for attrs in phase1_result["per_model"].values()
                if attr_key in attrs and attrs[attr_key]["cot_label"] not in ("ERROR", "unclear")
            ]
            if label in p1_labels:
                concordance = "concordant"
            elif p1_labels:
                concordance = "discordant"

        # Extract judgment (clinical opinion for Phase 3)
        judgment = ""
        judg_match = re.search(r"JUDGMENT:\s*(.+)", raw, re.IGNORECASE | re.DOTALL)
        if judg_match:
            # Take everything after JUDGMENT: up to end or next structured field
            judg_text = judg_match.group(1).strip()
            # Clean to single paragraph
            judg_text = re.sub(r"\s+", " ", judg_text).strip()
            judgment = clean_phase2_output(judg_text)
        if not judgment and raw_cleaned:
            # Fallback: use the full cleaned response as judgment context
            judgment = raw_cleaned

        p1_summary = _format_phase1_labels(phase1_result["per_model"], attr_key)

        per_attribute[attr_key] = {
            "raw": raw_cleaned,
            "label": label,
            "concordance": concordance,
            "judgment": judgment,
            "phase1_labels": p1_summary,
        }

        summary_lines.append(
            f"{attr_key}: MedGemma={label} [{concordance}] "
            f"(Phase1: {p1_summary})"
        )
        print(f"[BACKEND] Phase 2 {attr_key}: {label} [{concordance}]")

    return {
        "per_attribute": per_attribute,
        "summary": "\n".join(summary_lines),
    }


# =============================================================================
# Phase 3: MedGemma Text-Only Classification
# =============================================================================

def build_phase3_prompt(phase1_result, phase2_result):
    """Build text-only classification prompt from Phase 1 raw answers,
    Phase 2 concordance/discordance, and class definitions."""
    lines = []

    # Section 1: Phase 1 — Raw multi-VLM answers per attribute
    lines.append("=== PHASE 1: MULTI-VLM ATTRIBUTE OBSERVATIONS ===")
    for attr_key in ATTRIBUTE_KEYS:
        display = ATTR_DISPLAY_NAMES.get(attr_key, attr_key)
        model_answers = []
        for model_name, attrs in phase1_result["per_model"].items():
            if attr_key in attrs:
                label = attrs[attr_key]["cot_label"]
                model_answers.append(f"{model_name}={label}")
        lines.append(f"  {display}: {', '.join(model_answers)}")

    # Section 2: Phase 2 — MedGemma judgments with concordance/discordance
    lines.append("\n=== PHASE 2: MEDGEMMA CLINICAL JUDGMENTS ===")
    for attr_key in ATTRIBUTE_KEYS:
        p2 = phase2_result["per_attribute"][attr_key]
        display = ATTR_DISPLAY_NAMES.get(attr_key, attr_key)
        conc_str = p2["concordance"].upper()
        lines.append(f"\n  {display}: {p2['label']} [{conc_str}]")
        if p2.get("judgment"):
            lines.append(f"    Opinion: {p2['judgment']}")

    # Section 3: Class Definitions
    lines.append("\n=== LESION CLASS DEFINITIONS (dermoscopic criteria) ===")
    for code, definition in CLASS_DEFINITIONS.items():
        lines.append(f"\n  {definition}")

    # Section 4: Classification Instruction
    lines.append("\n=== CLASSIFICATION TASK ===")
    lines.append(
        "Based on the attribute evidence above, classify this lesion into "
        "exactly ONE of: MEL, NV, BCC, BKL, AKIEC, DF.\n\n"
        "Consider which class definition best matches the observed attributes. "
        "Weight CONCORDANT attributes (MedGemma agrees with Phase 1 models) "
        "more heavily than DISCORDANT ones. "
        "When MedGemma disagrees with Phase 1, prefer MedGemma's assessment "
        "as it is the medical domain specialist.\n\n"
        "Respond with exactly one classification code on a single line:\n"
        "CLASSIFICATION: <code>"
    )

    return "\n".join(lines)


def extract_classification_code(raw_output):
    """Extract classification code from Phase 3 output."""
    text = raw_output.upper()

    # Primary: CLASSIFICATION: <code>
    match = re.search(r"CLASSIFICATION:\s*(\w+)", text)
    if match:
        candidate = match.group(1).strip()
        if candidate in VALID_CODES:
            return candidate

    # Fallback: first valid code found in text
    for code in VALID_CODES:
        if code in text:
            return code

    return "NV"  # safe default (most prevalent class)


def run_phase3_classification(vlm, phase1_result, phase2_result):
    """Text-only MedGemma classification using Phase 1+2 context + class definitions."""
    prompt = build_phase3_prompt(phase1_result, phase2_result)

    vlm.reset()
    res = vlm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=32,
        temperature=0.2,
    )
    raw_output = res["choices"][0]["message"]["content"].strip()
    code = extract_classification_code(raw_output)
    return code, raw_output


# =============================================================================
# DermPipeline — Main Pipeline Class
# =============================================================================

class DermPipeline:
    def __init__(self):
        """Load MedGemma once. Ollama models are managed by Ollama server."""
        self._ollama_available = False

        # Check Ollama availability
        if HAS_OLLAMA:
            self._ollama_available = check_ollama_available()
            if self._ollama_available:
                print("[BACKEND] Ollama server: connected")
                for cfg in OLLAMA_MODELS:
                    avail = ensure_model_available(cfg["ollama_model"])
                    status = "ready" if avail else "needs pull"
                    print(f"[BACKEND]   {cfg['name']}: {status}")
            else:
                print("[BACKEND] WARNING: Ollama not available — falling back to MedGemma-only Phase 1")
        else:
            print("[BACKEND] WARNING: ollama package not installed — falling back to MedGemma-only Phase 1")

        # Load MedGemma vision model (always needed for Phase 2 + 3)
        print("[BACKEND] Loading MedGemma vision model (one-time)...")
        self._handler = MedGemmaChatHandler(clip_model_path=SPECIALIST_PROJ, verbose=False)
        self._vlm = Llama(
            model_path=SPECIALIST_PATH,
            n_ctx=4096,
            n_gpu_layers=-1,
            chat_handler=self._handler,
            verbose=False,
        )
        print("[BACKEND] MedGemma vision model loaded and ready.")

    def _run_phase1_medgemma_only(self, image_uri):
        """Fallback Phase 1 using only MedGemma when Ollama is unavailable."""
        per_model = {"MedGemma": {}}
        print("[BACKEND] Phase 1 — MedGemma-only fallback (no Ollama)")

        for attr_key, attr_cfg in COT_QUERIES.items():
            self._vlm.reset()
            try:
                res = self._vlm.create_chat_completion(
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_uri}},
                            {"type": "text", "text": attr_cfg["prompt"]},
                        ]
                    }],
                    max_tokens=512,
                    temperature=0.7,
                    repeat_penalty=1.5,
                )
                raw = res["choices"][0]["message"]["content"].strip()
            except Exception as e:
                print(f"[BACKEND]   ERROR MedGemma/{attr_key}: {e}")
                raw = f"ERROR: {e}"

            cot_label = extract_cot_label(raw, attr_key)
            benchmark_label = map_to_benchmark_label(attr_key, cot_label)
            per_model["MedGemma"][attr_key] = {
                "raw": raw,
                "cot_label": cot_label,
                "benchmark_label": benchmark_label,
            }
            print(f"[BACKEND]   MedGemma/{attr_key}: {cot_label}")

        return {
            "per_model": per_model,
            "models_used": ["MedGemma"],
            "models_failed": [],
        }

    def process(self, filename: str):
        """Run the full 3-phase multi-VLM consensus pipeline."""
        original_path = os.path.join(DATA_DIR, filename)
        temp_path = os.path.join(DATA_DIR, f"vlm_in_{uuid.uuid4().hex[:6]}.jpg")
        with Image.open(original_path) as img:
            img = img.convert("RGB").resize((448, 448), Image.Resampling.LANCZOS)
            img = ImageEnhance.Contrast(img).enhance(1.4)
            img.save(temp_path, quality=90)

        print(f"\n[BACKEND] >>> STARTING MULTI-VLM ANALYSIS: {filename}")

        try:
            # Prepare both image formats
            image_uri = encode_image_data_uri(temp_path)
            image_b64 = encode_image_base64_raw(temp_path)

            # ── PHASE 1: Multi-VLM CoT Attribute Extraction (raw outputs) ──
            if self._ollama_available:
                phase1_result = run_phase1_multi_vlm(image_b64)
                print(f"[BACKEND] Phase 1 complete: {len(phase1_result['models_used'])} models")
            else:
                phase1_result = self._run_phase1_medgemma_only(image_uri)
                print("[BACKEND] Phase 1 complete: MedGemma-only fallback")

            # ── PHASE 2: MedGemma Vision CoT Review ──
            phase2_result = run_phase2_medgemma_review(
                self._vlm, image_uri, phase1_result
            )
            print("[BACKEND] Phase 2 complete: MedGemma review done")

            # ── PHASE 3: MedGemma Text-Only Classification ──
            code, raw_class = run_phase3_classification(
                self._vlm, phase1_result, phase2_result
            )
            print(f"[BACKEND] Phase 3 Classification: {code} (raw: {raw_class})")

            # ── Build Output ──
            return self._build_output(code, phase1_result, phase2_result, raw_class)

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _build_output(self, code, phase1_result, phase2_result, raw_class):
        """Build output dict compatible with server.py's expected keys."""

        # raw_obs: show Phase 1 answers + Phase 2 concordance per attribute
        raw_obs = {}
        for attr_key in ATTRIBUTE_KEYS:
            ui_key = ATTR_TO_UI_KEY.get(attr_key, attr_key)
            p2 = phase2_result["per_attribute"][attr_key]
            raw_obs[ui_key] = (
                f"Phase1: {p2['phase1_labels']} | "
                f"MedGemma: {p2['label']} [{p2['concordance']}]"
            )

        # adj_desc: Phase 2 summary
        adj_desc = phase2_result["summary"]

        # final_implication: detailed report
        report_lines = [
            "--- MULTI-VLM CONSENSUS CLASSIFICATION ---",
            f"FINAL CLASSIFICATION: {code}",
            f"Models used: {', '.join(phase1_result['models_used'])}",
            "",
            "PHASE 1 — RAW MULTI-VLM OBSERVATIONS:",
        ]
        for attr_key in ATTRIBUTE_KEYS:
            model_answers = []
            for model_name, attrs in phase1_result["per_model"].items():
                if attr_key in attrs:
                    model_answers.append(f"{model_name}={attrs[attr_key]['cot_label']}")
            report_lines.append(f"  {attr_key}: {', '.join(model_answers)}")

        report_lines.append("\nPHASE 2 — MEDGEMMA JUDGMENTS + CONCORDANCE:")
        for attr_key in ATTRIBUTE_KEYS:
            p2 = phase2_result["per_attribute"][attr_key]
            report_lines.append(
                f"  {attr_key}: {p2['label']} [{p2['concordance'].upper()}]"
            )
            if p2.get("judgment"):
                report_lines.append(f"    > {p2['judgment']}")

        report_lines.append(f"\nPHASE 3 RAW OUTPUT: {raw_class}")
        report_lines.append("------------------------------------")

        return {
            "ai_code": code,
            "adj_desc": adj_desc,
            "raw_obs": raw_obs,
            "final_implication": "\n".join(report_lines),
        }
