import os
import uuid
import threading
import pandas as pd
from flask import Flask, send_from_directory, request, jsonify
from pipeline import DermPipeline, DATA_DIR

app = Flask(__name__)
db = {"meta": {}}
jobs = {}  # job_id -> {"status": "running"|"done"|"error", "result": ...}

# Singleton pipeline — model loaded once, reused across all requests
_pipeline = None
_pipeline_lock = threading.Lock()

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = DermPipeline()
    return _pipeline

def load_data():
    df = pd.read_csv("DermsGemms.csv", skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df = df.drop_duplicates(subset=['Image'], keep='first')
    db["meta"] = df.set_index('Image').to_dict('index')
    print(f"Loaded {len(db['meta'])} unique image records.")

load_data()

def _run_analysis(job_id, fname):
    """Run pipeline in background thread, store result in jobs dict.

    Lock serializes GPU access — only one analysis at a time.
    """
    try:
        with _pipeline_lock:
            res = get_pipeline().process(fname)
        rec = db["meta"].get(os.path.splitext(fname)[0], {})
        jobs[job_id] = {
            "status": "done",
            "result": {
                "ai_code": res.get("ai_code", "UNK"),
                "raw_obs": res.get("raw_obs", {}),
                "adj_desc": res.get("adj_desc", ""),
                "final_implication": res.get("final_implication", "Error"),
                "gt_code": str(rec.get('Dx', 'UNK')).upper(),
                "gt_attr": rec.get('Lesion attributes', 'N/A')
            }
        }
    except Exception as e:
        jobs[job_id] = {"status": "error", "error": str(e)}

@app.route('/api/list-images')
def list_imgs():
    return jsonify([f"{i}.jpg" for i in db["meta"].keys()])

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Start analysis in background, return job_id immediately."""
    fname = request.json.get('filename')
    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {"status": "running"}
    t = threading.Thread(target=_run_analysis, args=(job_id, fname), daemon=True)
    t.start()
    return jsonify({"job_id": job_id})

@app.route('/api/status/<job_id>')
def job_status(job_id):
    """Poll for job completion. Returns status + result when done."""
    job = jobs.get(job_id)
    if not job:
        return jsonify({"status": "not_found"}), 404
    return jsonify(job)

@app.route('/images/<path:f>')
def serve_img(f): return send_from_directory(DATA_DIR, f)

@app.route('/')
def index(): return send_from_directory('.', 'index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6565, debug=False, threaded=True)
