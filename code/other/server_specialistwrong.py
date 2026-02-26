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
                "Architecture": "Evaluate symmetry. Is the lesion symmetric or asymmetric?",
                "Network": "Pigment network (absent/regular/atypical)?",
                "Structures": "Any specific structures (haze/globules/nests)?",
                "Colors": "Colors (red/white/blue/black/brown)?"
            }
            raw_obs = {}
            for key, q in obs_queries.items():
                res = vlm.create_completion(prompt=f"answer en {q}\nResult:", max_tokens=32)
                val = res["choices"][0]["text"].strip().lower()
                
                # --- FUZZY REPAIR: Fixes model truncation (e.g., 'ymmetric' -> 'asymmetric') ---
                if key == "Architecture":
                    if "asym" in val or "ymmetric" in val and "as" in val:
                        val = "asymmetric"
                    elif "ymmetric" in val:
                        val = "symmetric"
                raw_obs[key] = val
            
            del vlm
            gc.collect()

            # --- PHASE 2: MEDGEMMA ADJUDICATOR (Clinical Reasoning) ---
            reasoner = Llama(model_path=MEDGEMMA_PATH, n_ctx=4096, n_gpu_layers=-1, verbose=False)
            
            # Formatting findings for the prompt
            findings_summary = f"{raw_obs['Architecture']} architecture, {raw_obs['Network']} network, structures: {raw_obs['Structures']}, colors: {raw_obs['Colors']}."

            # Pass A: Adjudicator Description (Using Gemma Chat Template)
            desc_prompt = (
                f"<start_of_turn>user\n"
                f"You are a clinical adjudicator. Summarize these dermoscopic findings into a professional description:\n"
                f"{findings_summary}<end_of_turn>\n"
                f"<start_of_turn>model\n"
                f"The lesion presents with "
            )
            
            desc_res = reasoner.create_completion(prompt=desc_prompt, max_tokens=100, temperature=0.3, stop=["<end_of_turn>", "\n"])
            adjudicated_desc = "The lesion presents with " + desc_res["choices"][0]["text"].strip()

            # Pass B: Classification (Strict mapping)
            class_prompt = (
                f"<start_of_turn>user\n"
                f"Diagnosis based on: {adjudicated_desc}\n"
                "Return only the clinical code: MEL, NV, BCC, or BKL.<end_of_turn>\n"
                f"<start_of_turn>model\n"
                "FINAL CLASSIFICATION: "
            )
            
            class_res = reasoner.create_completion(prompt=class_prompt, max_tokens=10, temperature=0.01)
            final_code = class_res["choices"][0]["text"].strip().upper().replace(".", "")
            
            # LOGIC SAFETY: Ensure asymmetric always flags high-risk
            if "asymmetric" in raw_obs['Architecture'] and "MEL" not in final_code:
                final_code = "MEL (Suspicious)"

            del reasoner
            gc.collect()

            # --- PHASE 3: THE HARDCODED INTEGRATED REPORT ---
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
