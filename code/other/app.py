import os
import json
from flask import Flask, render_template, request, jsonify
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

app = Flask(__name__)
DATA_FOLDER = os.path.abspath("./data")

server_params = StdioServerParameters(
    command="docker",
    args=["run", "-i", "--rm", "-v", f"{os.path.abspath('./models')}:/models", "-v", f"{DATA_FOLDER}:/data", "derm-mcp"]
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # 1. Sync File Save
    file = request.files['image']
    filename = file.filename
    file.save(os.path.join(DATA_FOLDER, filename))

    # 2. Sync MCP Communication
    with stdio_client(server_params) as (read, write):
        with ClientSession(read, write) as session:
            session.initialize()
            result = session.call_tool("analyze_lesion", arguments={"filename": filename})
            # Convert stringified dict back to JSON
            report_data = eval(result.content[0].text) 
            
    return jsonify({
        "image_url": f"/static_data/{filename}",
        "specialists": report_data["specialists"],
        "diagnosis": report_data["diagnosis"]
    })

if __name__ == '__main__':
    app.run(port=5000)
