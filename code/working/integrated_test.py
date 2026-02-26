import os
import json
import time
import logging

# Set up simple logging to match the server
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestPipeline")

# --- CONFIGURATION ---
TEST_IMAGE = "test_image.jpg"
VALID_CODES = ["MEL", "NV", "BCC", "BKL", "AKIEC", "DF"]

def run_integrated_test(image_filename):
    print("\n" + "="*50)
    print("🚀 Starting Integrated Observer-Adjudicator Test")
    print(f"📸 Target Image: {image_filename}")
    print("="*50 + "\n")

    start_time = time.time()

    try:
        # Import the DermPipeline from your server script
        from server import DermPipeline
        pipeline = DermPipeline()

        # Run the full pipeline
        result = pipeline.process(image_filename)

        if "error" in result:
            print(f"❌ PIPELINE ERROR: {result['error']}")
            return

        # --- PARSING THE ADJUDICATED REPORT ---
        report = result.get("final_implication", "")
        specialist_obs = result.get("specialist_findings", {})

        print("🔬 PHASE 1: SPECIALIST RAW OBSERVATIONS")
        print(json.dumps(specialist_obs, indent=2))
        print("\n" + "-"*30)

        print("👨‍⚕️ PHASE 2: MEDGEMMA ADJUDICATION")
        
        # Logic to extract the code from our hardcoded header
        extracted_code = "NONE"
        if "FINAL CLASSIFICATION:" in report:
            # Splits at header, takes the next line, and cleans whitespace/formatting
            extracted_code = report.split("FINAL CLASSIFICATION:")[1].split("\n")[0].strip()
            # Handle potential markdown bolding if MedGemma added it (e.g., **MEL**)
            extracted_code = extracted_code.replace("*", "").upper()

        print(f"Extracted Verdict: {extracted_code}")
        print("-"*30)

        # --- VALIDATION LOGIC ---
        total_time = time.time() - start_time
        print(f"⏱️  Total Processing Time: {total_time:.2f} seconds")

        # Check if the extracted code is in our medical list
        match = any(code in extracted_code for code in VALID_CODES)
        
        if match:
            print(f"✅ SUCCESS: Clinical code [{extracted_code}] confirmed.")
            print("\n--- FULL GENERATED REPORT ---")
            print(report)
        else:
            print(f"⚠️  WARNING: Output received, but extracted code '{extracted_code}' is not in standard list.")
            print("Full output for debugging:")
            print(report)

    except ImportError:
        print("❌ ERROR: Could not find 'DermPipeline' in server.py. Check your filename.")
    except Exception as e:
        print(f"❌ TEST FAILED: {str(e)}")

if __name__ == "__main__":
    run_integrated_test(TEST_IMAGE)
