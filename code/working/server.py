import os
import gc
import json
import logging
import re
from PIL import Image, ImageEnhance, ImageFilter
from llama_cpp import Llama
from mcp.server.fastmcp import FastMCP

# --- CONFIGURATION ---
MODEL_DIR = "/home/project/models"
DATA_DIR = "/home/project/data"
SPECIALIST_PATH = os.path.join(MODEL_DIR, "specialist.gguf")
SPECIALIST_PROJ = os.path.join(MODEL_DIR, "specialist-mmproj.gguf")
MEDGEMMA_PATH = os.path.join(MODEL_DIR, "medgemma.gguf")

mcp = FastMCP("DermatoAI")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DermatoServer")

class DermPipeline:
    def _prepare_image(self, filename: str, size: int = 224) -> str:
        original_path = os.path.join(DATA_DIR, filename)
        temp_path = os.path.join(DATA_DIR, f"proc_{filename}")
        with Image.open(original_path) as img:
            img = img.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
            
            # --- AGGRESSIVE EDGE ENHANCEMENT ---
            # Boost contrast and apply an unsharp mask to make borders definitive
            img = ImageEnhance.Contrast(img).enhance(1.6)
            img = img.filter(ImageFilter.SHARPEN)
            img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
            
            img.save(temp_path)
        return os.path.abspath(temp_path)

    def process(self, filename: str):
        processed_path = self._prepare_image(filename)
        
        try:
            # --- PHASE 1: SPECIALIST ---
            vlm = Llama(model_path=SPECIALIST_PATH, clip_model_path=SPECIALIST_PROJ, n_ctx=2048, n_gpu_layers=-1, verbose=False)
            obs_queries = {
                "Architecture": "Is the lesion asymmetric or symmetric?",
                "Network": "Is the pigment network atypical, regular, or absent?",
                "Structures": "Note any blue-white haze or regression.",
                "Colors": "List colors: red, white, blue, black, brown."
            }
            raw_obs = {}
            for key, q in obs_queries.items():
                res = vlm.create_completion(prompt=f"answer en {q}\nResult:", max_tokens=32)
                val = res["choices"][0]["text"].strip().lower()
                
                # REPAIR LAYER
                if key == "Architecture":
                    # If it says 'no lesion' but found colors, it's asymmetric by default
                    if "no lesion" in val or "none" in val:
                        val = "asymmetric (diffuse)"
                    elif "asym" in val or ("ymmetric" in val and "as" in val):
                        val = "asymmetric"
                    elif "ymmetric" in val:
                        val = "symmetric"
                raw_obs[key] = val
            del vlm
            gc.collect()

            # --- PHASE 2: MEDGEMMA ADJUDICATOR ---
            reasoner = Llama(model_path=MEDGEMMA_PATH, n_ctx=4096, n_gpu_layers=-1, verbose=False)
            
            # Explicit instruction to NOT use lists
            desc_prompt = (
                f"<start_of_turn>user\n"
                f"Findings: {raw_obs['Architecture']} architecture, {raw_obs['Network']} network, colors: {raw_obs['Colors']}.\n"
                "Describe the clinical appearance in one sentence. No lists.<end_of_turn>\n"
                f"<start_of_turn>model\n"
                "The lesion presents as "
            )
            desc_res = reasoner.create_completion(prompt=desc_prompt, max_tokens=100, temperature=0.2, stop=["<end_of_turn>", "\n"])
            adjudicated_desc = "The lesion presents as " + desc_res["choices"][0]["text"].strip()

            class_prompt = (
                f"<start_of_turn>user\n"
                f"Description: {adjudicated_desc}\n"
                "Select one: MEL, NV, BCC, or BKL.<end_of_turn>\n"
                f"<start_of_turn>model\n"
                "CODE: "
            )
            class_res = reasoner.create_completion(prompt=class_prompt, max_tokens=10, temperature=0.01)
            raw_code = class_res["choices"][0]["text"].strip().upper()

            # --- PHASE 3: CLEANER & OVERRIDE ---
            potential_codes = ["MEL", "NV", "BCC", "BKL"]
            final_code = "UNKNOWN"
            for code in potential_codes:
                if code in raw_code:
                    final_code = code
                    break
            
            # Clinical override for polychromia
            color_count = sum(1 for c in ["red", "white", "blue", "black", "brown"] if c in raw_obs['Colors'])
            if color_count >= 3 or "blue" in raw_obs['Colors']:
                final_code = "MEL"

            del reasoner
            gc.collect()

            # --- PHASE 4: REPORT ---
            final_report = (
                f"--- INTEGRATED DERMATOLOGY ADJUDICATION REPORT ---\n"
                f"FINAL CLASSIFICATION: {final_code}\n\n"
                f"ADJUDICATOR DESCRIPTION:\n"
                f"{adjudicated_desc}\n\n"
                f"SPECIALIST RAW OBSERVATIONS:\n"
                f"- Architecture: {raw_obs['Architecture']}\n"
                f"- Pigment Network: {raw_obs['Network']}\n"
                f"- Observed Structures: {raw_obs['Structures']}\n"
                f"- Color Palette: {raw_obs['Colors']}\n"
                f"--------------------------------------------"
            )

            return {"specialist_findings": raw_obs, "final_implication": final_report}

        except Exception as e:
            return {"error": str(e)}
        finally:
            if os.path.exists(processed_path): os.remove(processed_path)

@mcp.tool()
def analyze_lesion(image_filename: str) -> str:
    pipeline = DermPipeline()
    return json.dumps(pipeline.process(image_filename), indent=2)

if __name__ == "__main__": mcp.run()
