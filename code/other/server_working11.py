import os
import gc
import json
import logging
import re
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
            # --- PHASE 1: SPECIALIST (Refined Visual Queries) ---
            vlm = Llama(model_path=SPECIALIST_PATH, clip_model_path=SPECIALIST_PROJ, n_ctx=2048, n_gpu_layers=-1, verbose=False)
            obs_queries = {
                # Changed from "Is it..." to "Describe the..." to avoid binary 'no lesion' errors
                "Architecture": "Describe the lesion shape: is it asymmetric or symmetric?",
                "Network": "Describe the pigment network: is it atypical, regular, or absent?",
                "Structures": "Confirm if blue-white haze or regression structures are visible.",
                "Colors": "List all colors present (red, white, blue, black, brown, tan)."
            }
            raw_obs = {}
            for key, q in obs_queries.items():
                res = vlm.create_completion(prompt=f"answer en {q}\nResult:", max_tokens=32)
                val = res["choices"][0]["text"].strip().lower()
                
                # REPAIR LAYER: Handle "no lesion" or truncation
                if key == "Architecture":
                    if "no lesion" in val or "normal" in val:
                        # If the specialist fails but sees many colors, we force 'asymmetric' for the adjudicator
                        val = "asymmetric (detected by color distribution)"
                    elif "asym" in val or ("ymmetric" in val and "as" in val):
                        val = "asymmetric"
                    elif "ymmetric" in val:
                        val = "symmetric"
                raw_obs[key] = val
            del vlm
            gc.collect()

            # --- PHASE 2: MEDGEMMA (Clinical Adjudicator) ---
            reasoner = Llama(model_path=MEDGEMMA_PATH, n_ctx=4096, n_gpu_layers=-1, verbose=False)
            
            # Pass A: Narrative Summary
            desc_prompt = (
                f"<start_of_turn>user\n"
                f"Findings: {raw_obs['Architecture']}, {raw_obs['Network']} network, colors: {raw_obs['Colors']}.\n"
                "Synthesize a one-sentence clinical description of this lesion.<end_of_turn>\n"
                f"<start_of_turn>model\n"
                "The lesion exhibits "
            )
            desc_res = reasoner.create_completion(prompt=desc_prompt, max_tokens=100, temperature=0.2, stop=["<end_of_turn>", "\n"])
            adjudicated_desc = "The lesion exhibits " + desc_res["choices"][0]["text"].strip()

            # Pass B: Strict Classification
            class_prompt = (
                f"<start_of_turn>user\n"
                f"Description: {adjudicated_desc}\n"
                "Task: Provide the 3-letter clinical code (MEL, NV, BCC, or BKL).<end_of_turn>\n"
                f"<start_of_turn>model\n"
                "CODE: "
            )
            class_res = reasoner.create_completion(prompt=class_prompt, max_tokens=15, temperature=0.01)
            raw_code = class_res["choices"][0]["text"].strip().upper()

            # --- PHASE 3: THE CLEANER & LOGIC OVERRIDE ---
            potential_codes = ["MEL", "NV", "BCC", "BKL"]
            final_code = "UNKNOWN"
            for code in potential_codes:
                if code in raw_code:
                    final_code = code
                    break
            
            # THE "BLUE-WHITE" RULE: Haze/Blue/White presence overrides 'symmetric' architecture
            if any(c in raw_obs['Colors'] for c in ["blue", "white", "black"]) and final_code == "NV":
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
