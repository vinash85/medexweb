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
            # --- PHASE 1: SPECIALIST (Aggressive Architecture Check) ---
            vlm = Llama(model_path=SPECIALIST_PATH, clip_model_path=SPECIALIST_PROJ, n_ctx=2048, n_gpu_layers=-1, verbose=False)
            
            # We explicitly ask it to look for asymmetry first
            obs_queries = {
                "Architecture": "Is this lesion asymmetric? Answer 'asymmetric' if there is any imbalance in shape, otherwise 'symmetric'.",
                "Network": "Is the pigment network atypical or regular?",
                "Structures": "List structures like blue-white haze or nests.",
                "Colors": "List all colors (red, white, blue, black, brown)."
            }
            raw_obs = {}
            for key, q in obs_queries.items():
                res = vlm.create_completion(prompt=f"answer en {q}\nResult:", max_tokens=32)
                val = res["choices"][0]["text"].strip().lower()
                
                # REPAIR LAYER: Catch truncation AND common errors
                if key == "Architecture":
                    if "asym" in val or "ymmetric" in val and "as" in val:
                        val = "asymmetric"
                    elif "ymmetric" in val:
                        val = "symmetric"
                raw_obs[key] = val
            
            del vlm
            gc.collect()

            # --- PHASE 2: MEDGEMMA ADJUDICATOR (Strict Response) ---
            reasoner = Llama(model_path=MEDGEMMA_PATH, n_ctx=4096, n_gpu_layers=-1, verbose=False)
            
            # Pass A: The Descriptor
            desc_prompt = (
                f"<start_of_turn>user\n"
                f"Describe this skin lesion: {raw_obs['Architecture']} architecture, {raw_obs['Network']} network, "
                f"structures: {raw_obs['Structures']}, colors: {raw_obs['Colors']}.\n"
                "Provide a one-sentence clinical summary.<end_of_turn>\n"
                f"<start_of_turn>model\n"
                "The lesion exhibits "
            )
            desc_res = reasoner.create_completion(prompt=desc_prompt, max_tokens=80, temperature=0.2, stop=["<end_of_turn>", "\n"])
            adjudicated_desc = "The lesion exhibits " + desc_res["choices"][0]["text"].strip()

            # Pass B: The Classifier (Removed the 'Menu' style to prevent echoing)
            class_prompt = (
                f"<start_of_turn>user\n"
                f"Based on this description: '{adjudicated_desc}', classify the lesion.\n"
                "Options: MEL, NV, BCC, BKL. Answer with ONLY the code.<end_of_turn>\n"
                f"<start_of_turn>model\n"
                "CODE: "
            )
            class_res = reasoner.create_completion(prompt=class_prompt, max_tokens=5, temperature=0.01)
            final_code = class_res["choices"][0]["text"].strip().upper().replace("CODE:", "").replace(".", "").strip()
            
            # THE LOGIC OVERRIDE: If Specialist said 'symmetric' but you know it's not, 
            # we can force the report to show the correction if needed. 
            # For now, let's ensure the output is at least a valid code.
            if not any(x in final_code for x in ["MEL", "NV", "BCC", "BKL"]):
                final_code = "MEL" if "asymmetric" in raw_obs['Architecture'] else "NV"

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
            if os.path.exists(processed_path): os.remove(processed_path)

@mcp.tool()
def analyze_lesion(image_filename: str) -> str:
    pipeline = DermPipeline()
    return json.dumps(pipeline.process(image_filename), indent=2)

if __name__ == "__main__":
    mcp.run()
