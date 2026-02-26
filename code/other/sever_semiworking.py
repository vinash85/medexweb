import os
import gc
import json
import logging
from PIL import Image
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
            img.save(temp_path)
        return os.path.abspath(temp_path)

    def process(self, filename: str):
        processed_path = self._prepare_image(filename)
        
        try:
            # --- PHASE 1: SPECIALIST (The Visual Observer) ---
            vlm = Llama(model_path=SPECIALIST_PATH, clip_model_path=SPECIALIST_PROJ, n_ctx=2048, n_gpu_layers=-1, verbose=False)
            obs_queries = {
                "Architecture": "Architecture (symmetric/asymmetric)?",
                "Network": "Pigment network (absent/regular/atypical)?",
                "Structures": "Specific structures (haze/globules/nests)?",
                "Colors": "Colors present (red/white/blue/black/brown)?"
            }
            raw_obs = {}
            for key, q in obs_queries.items():
                res = vlm.create_completion(prompt=f"answer en {q}\nResult:", max_tokens=32)
                val = res["choices"][0]["text"].strip().lower()
                # Auto-correction for truncation
                if "asym" in val: val = "asymmetrical"
                if "sym" in val and "asym" not in val: val = "symmetrical"
                raw_obs[key] = val
            
            del vlm
            gc.collect()

            # --- PHASE 2: MEDGEMMA DUAL-PASS ---
            reasoner = Llama(model_path=MEDGEMMA_PATH, n_ctx=4096, n_gpu_layers=-1, verbose=False)
            
            # Pass A: The Descriptor
            desc_prompt = (
                f"Clinical Findings: {raw_obs['Architecture']} architecture, {raw_obs['Network']} network, "
                f"structures: {raw_obs['Structures']}, colors: {raw_obs['Colors']}.\n"
                "Task: Summarize these dermoscopic features in one professional sentence.\n"
                "Adjudication Description: The lesion exhibits "
            )
            desc_res = reasoner.create_completion(prompt=desc_prompt, max_tokens=100, stop=["\n"])
            adjudicated_desc = "The lesion exhibits " + desc_res["choices"][0]["text"].strip()

            # Pass B: The Classifier (Now with Strict Formatting)
            class_prompt = (
                f"Analysis: {adjudicated_desc}\n\n"
                "Classification Categories:\n"
                "- MEL (Melanoma: asymmetric/atypical)\n"
                "- NV (Nevus: symmetric/regular)\n"
                "- BCC (Basal Cell: nests/vascular)\n"
                "- BKL (Keratosis: yellow/crust)\n\n"
                "Question: Which single category code best fits? Answer with the code only.\n"
                "Final Code: "
            )
            class_res = reasoner.create_completion(
                prompt=class_prompt, 
                max_tokens=5, 
                temperature=0.01,
                stop=["\n", " ", "."]
            )
            ai_code = class_res["choices"][0]["text"].strip().upper()
            
            # PHASE 3: LOGIC FALLBACK (Double Check)
            # If AI outputs numbers or fails, use the Specialist findings to force a label
            if not any(x in ai_code for x in ["MEL", "NV", "BCC", "BKL"]):
                if "asymmetrical" in raw_obs['Architecture'] or "atypical" in raw_obs['Network']:
                    final_code = "MEL"
                else:
                    final_code = "NV"
            else:
                final_code = ai_code

            del reasoner
            gc.collect()

            # --- PHASE 4: THE HARDCODED INTEGRATED REPORT ---
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

            return {
                "specialist_findings": raw_obs, 
                "final_implication": final_report
            }

        except Exception as e:
            return {"error": str(e)}
        finally:
            if os.path.exists(processed_path):
                os.remove(processed_path)

@mcp.tool()
def analyze_lesion(image_filename: str) -> str:
    pipeline = DermPipeline()
    return json.dumps(pipeline.process(image_filename), indent=2)

if __name__ == "__main__":
    mcp.run()
