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
            # --- PHASE 1: SPECIALIST (Visual Evidence with Fuzzy Correction) ---
            vlm = Llama(model_path=SPECIALIST_PATH, clip_model_path=SPECIALIST_PROJ, n_ctx=2048, n_gpu_layers=-1, verbose=False)
            
            obs_queries = {
                "Architecture": "Carefully evaluate the symmetry. Is the lesion symmetric or asymmetric across its axes?",
                "Network": "Pigment network type (absent/regular/atypical)?",
                "Structures": "Specific structures (blue-white haze/globules/nests)?",
                "Colors": "Colors (red/white/blue/black/brown)?"
            }
            raw_obs = {}
            for key, q in obs_queries.items():
                res = vlm.create_completion(prompt=f"answer en {q}\nResult:", max_tokens=32)
                val = res["choices"][0]["text"].strip().lower()
                
                # --- FUZZY WORD REPAIR LOGIC ---
                # Fixes truncation like 'ymmetric' or 'asym'
                if key == "Architecture":
                    if "asym" in val or "ymmetric" in val.split(' ')[0] and "as" in val:
                         val = "asymmetric"
                    elif "ymmetric" in val:
                         val = "symmetric"
                
                raw_obs[key] = val
            
            del vlm
            gc.collect()

            # --- PHASE 2: MEDGEMMA ADJUDICATOR ---
            reasoner = Llama(model_path=MEDGEMMA_PATH, n_ctx=4096, n_gpu_layers=-1, verbose=False)
            
            # Adjudicator Pass 1: Narrative Description
            desc_prompt = (
                f"Visual Input: {raw_obs['Architecture']} architecture, {raw_obs['Network']} network, "
                f"features: {raw_obs['Structures']}, colors: {raw_obs['Colors']}.\n"
                "Task: Provide a high-level clinical description emphasizing the structural symmetry.\n"
                "Adjudication Description: "
            )
            desc_res = reasoner.create_completion(prompt=desc_prompt, max_tokens=100, stop=["\n"])
            adjudicated_desc = desc_res["choices"][0]["text"].strip()

            # Adjudicator Pass 2: Classification
            class_prompt = (
                f"Clinical Findings: {adjudicated_desc}\n"
                "Diagnostic Weights: [Asymmetric = High MEL Risk], [Symmetric = Low NV Risk].\n"
                "Final Code (MEL, NV, BCC, BKL): "
            )
            class_res = reasoner.create_completion(prompt=class_prompt, max_tokens=10, temperature=0.01)
            final_code = class_res["choices"][0]["text"].strip().upper().replace(".", "")
            
            del reasoner
            gc.collect()

            # --- PHASE 3: THE HARDCODED REPORT ---
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
            if os.path.exists(processed_path):
                os.remove(processed_path)

@mcp.tool()
def analyze_lesion(image_filename: str) -> str:
    pipeline = DermPipeline()
    return json.dumps(pipeline.process(image_filename), indent=2)

if __name__ == "__main__":
    mcp.run()
