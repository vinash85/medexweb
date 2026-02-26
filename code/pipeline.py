import os
import gc
import json
import uuid
import time
from PIL import Image, ImageEnhance, ImageFilter
from llama_cpp import Llama

# --- CONFIGURATION ---
MODEL_DIR = "/home/project/models"
DATA_DIR = "/home/project/data/images"
SPECIALIST_PATH = os.path.join(MODEL_DIR, "specialist.gguf")
SPECIALIST_PROJ = os.path.join(MODEL_DIR, "specialist-mmproj.gguf")
MEDGEMMA_PATH = os.path.join(MODEL_DIR, "medgemma.gguf")

class DermPipeline:
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
        raw_obs = {}
        print(f"\n[BACKEND] >>> STARTING FRESH ANALYSIS: {filename}")
        
        try:
            # PHASE 1: Specialist (The Vision Model)
            # Re-initializing here is heavy, but essential to clear internal KV-cache
            vlm = Llama(
                model_path=SPECIALIST_PATH, 
                clip_model_path=SPECIALIST_PROJ, 
                n_ctx=1024, 
                n_gpu_layers=-1, 
                verbose=False
            )
            
            # Using a random seed for every image to break deterministic loops
            current_seed = int(time.time())
            
            obs_queries = {
                "Architecture": "Is this lesion symmetric or asymmetric?",
                "Network": "Does it have a pigment network? Answer Yes or No.",
                "Structures": "Describe structures like dots or haze. Say 'clear' if none.",
                "Colors": "What specific colors are in this lesion? (e.g. brown, black, white)"
            }

            for key, q in obs_queries.items():
                # Including the filename in the prompt makes the context unique
                prompt = f"IMAGE_REF: {filename}\nUSER: [img-0]\n{q}\nASSISTANT: Based on pixels, the"
                
                res = vlm.create_completion(
                    prompt=prompt, 
                    max_tokens=32, 
                    temperature=0.7,   # Non-zero temp allows for variability
                    repeat_penalty=1.5, # HIGH penalty stops the "red/blue" loop
                    seed=current_seed,
                    stop=["USER:", "IMAGE_REF:"]
                )
                val = res["choices"][0]["text"].strip().lower()
                
                # Cleanup hallucinated prefixes
                val = val.replace("lesion is", "").replace("evidence shows", "").replace(".", "").strip()
                
                if key == "Architecture":
                    val = "asymmetric" if "asym" in val else "symmetric"
                raw_obs[key] = val
                print(f"[BACKEND] Specialist ({key}): {val}")
            
            # HARD RESET: Clean up memory immediately
            vlm.reset()
            del vlm
            gc.collect()
            time.sleep(0.5) # Brief pause for GPU VRAM deallocation

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
