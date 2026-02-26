import os
import pandas as pd
from flask import Flask, send_from_directory, request, jsonify
from pipeline import DermPipeline, DATA_DIR

app = Flask(__name__)
db = {"meta": {}}

def load_data():
    df = pd.read_csv("DermsGemms.csv", skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    # Remove duplicates to prevent the .to_dict('index') error
    df = df.drop_duplicates(subset=['Image'], keep='first')
    db["meta"] = df.set_index('Image').to_dict('index')
    print(f"Loaded {len(db['meta'])} unique image records.")

load_data()

@app.route('/api/list-images')
def list_imgs():
    return jsonify([f"{i}.jpg" for i in db["meta"].keys()])

@app.route('/api/analyze', methods=['POST'])
def analyze():
    fname = request.json.get('filename')
    # Fresh pipeline instance for every call
    res = DermPipeline().process(fname)
    rec = db["meta"].get(os.path.splitext(fname)[0], {})
    return jsonify({
        "ai_code": res.get("ai_code", "UNK"),
        "final_implication": res.get("final_implication", "Error"),
        "gt_code": str(rec.get('Dx', 'UNK')).upper(),
        "gt_attr": rec.get('Lesion attributes', 'N/A')
    })

@app.route('/images/<path:f>')
def serve_img(f): return send_from_directory(DATA_DIR, f)

@app.route('/')
def index(): return send_from_directory('.', 'index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6565, debug=False)
