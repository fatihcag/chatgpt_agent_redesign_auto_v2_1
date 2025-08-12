# server.py
import os, zipfile
from flask import Flask, send_from_directory, jsonify
from agent import run as run_agent
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

def build_zip():
    os.makedirs("output", exist_ok=True)
    zpath = os.path.join("output", "output.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk("output"):
            for f in files:
                if f.endswith((".png",".txt",".csv")) and f != "output.zip":
                    p = os.path.join(root, f)
                    z.write(p, arcname=os.path.relpath(p, "output"))
    return zpath

@app.route("/", methods=["GET"])
def home():
    return "OK — /run ile başlat, /files ile listele, /download/output.zip ile indir."

@app.route("/run", methods=["POST","GET"])
def run_job():
    run_agent()               # agent.py içindeki run() fonksiyonunu çalıştırır
    z = build_zip()
    return jsonify({"status":"ok","zip":"/download/output.zip"})

@app.route("/files", methods=["GET"])
def list_files():
    os.makedirs("output", exist_ok=True)
    return jsonify(sorted(os.listdir("output")))

@app.route("/download/<path:fname>", methods=["GET"])
def download(fname):
    return send_from_directory("output", fname, as_attachment=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
