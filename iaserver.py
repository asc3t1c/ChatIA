#!/usr/bin/python
# by nu11secur1ty 2025
import os
import json
import re
import requests
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from bs4 import BeautifulSoup
from llama_cpp import Llama

app = Flask(__name__)

MODEL_FILENAME = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
KNOWLEDGE_FILE = 'knowledge.json'
ALLOWED_EXTENSIONS = {'.txt', '.py', '.pdf'}
SESSIONS_DIR = "sessions"
SESSION_FILE = os.path.join(SESSIONS_DIR, "session.json")
UPLOAD_FOLDER = "uploads"

# Ensure necessary folders exist
for folder in [SESSIONS_DIR, UPLOAD_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# ------------------- Model Download -------------------

def download_file(url, filename):
    print(f"⬇️ Downloading {url} ...")
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"✅ Downloaded and saved as {filename}")
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")

def ensure_file(filename, url):
    if not os.path.exists(filename):
        print(f"⚠️ Model '{filename}' not found. Downloading...")
        download_file(url, filename)
    else:
        print(f"✅ Model '{filename}' found.")

MODEL_DOWNLOAD_URL = (
    "https://huggingface.co/f0rc3ps/mistral-7b-nu11secur1ty/resolve/main/"
    "mistral-7b-nu11secur1ty-v0.1.Q4_K_M.gguf?download=true"
)
ensure_file(MODEL_FILENAME, MODEL_DOWNLOAD_URL)

llm = Llama(model_path=MODEL_FILENAME, n_ctx=2048, n_threads=4)

# ------------------- Knowledge Handling -------------------

def load_knowledge():
    if not os.path.exists(KNOWLEDGE_FILE):
        with open(KNOWLEDGE_FILE, 'w', encoding='utf-8') as f:
            json.dump([], f)
    with open(KNOWLEDGE_FILE, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

def save_knowledge(data):
    with open(KNOWLEDGE_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def extract_text(content, filename=None):
    if filename and filename.endswith(('.html', '.php')):
        soup = BeautifulSoup(content, "html.parser")
        content = soup.get_text()
    return re.sub(r'\s+', ' ', content).strip()

def add_to_knowledge(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    knowledge = load_knowledge()
    new = [s for s in sentences if s not in knowledge]
    knowledge.extend(new)
    save_knowledge(knowledge)
    return len(new)

# ------------------- Session Handling -------------------

def load_session():
    if os.path.exists(SESSION_FILE):
        with open(SESSION_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_session(session_data):
    for msg in session_data:
        if 'timestamp' not in msg:
            msg['timestamp'] = datetime.now().isoformat()
    with open(SESSION_FILE, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2, ensure_ascii=False)

def save_session_timestamped():
    session = load_session()
    for msg in session:
        if 'timestamp' not in msg:
            msg['timestamp'] = datetime.now().isoformat()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(SESSIONS_DIR, f"session_{timestamp}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(session, f, indent=2, ensure_ascii=False)
    return filename

# ------------------- Chat Logic -------------------

def best_knowledge_match(user_input, knowledge):
    user_words = set(re.findall(r'\b\w+\b', user_input.lower()))
    best_match = None
    best_score = 0
    for sentence in knowledge:
        sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
        overlap = len(user_words & sentence_words)
        if overlap > best_score:
            best_score = overlap
            best_match = sentence
    return best_match if best_score >= 3 else None

# ------------------- Routes -------------------

@app.route("/")
def serve_html():
    return send_from_directory(".", "index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "").strip()
    if not user_input:
        return jsonify({"from": "IA", "reply": "Say something..."})

    knowledge = load_knowledge()
    matched_reply = best_knowledge_match(user_input, knowledge)

    if not matched_reply:
        try:
            completion = llm.create_completion(
                prompt=f"[INST] {user_input} [/INST]",
                max_tokens=150,
                temperature=0.7
            )
            matched_reply = completion["choices"][0]["text"].strip()
        except Exception as e:
            matched_reply = f"⚠️ Model error: {e}"

    session = load_session()
    timestamp_now = datetime.now().isoformat()
    session.append({"user": user_input, "bot": matched_reply, "timestamp": timestamp_now})
    save_session(session)

    return jsonify({"from": "IA", "reply": matched_reply})

@app.route("/upload", methods=["POST"])
def upload_file():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file."}), 400

    filename = file.filename.lower()
    ext = os.path.splitext(filename)[-1]
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"⛔ Unsupported file type '{ext}'."}), 403

    save_path = os.path.join(UPLOAD_FOLDER, filename)
    try:
        file.save(save_path)
    except Exception as e:
        return jsonify({"error": f"⚠️ Failed to save file: {e}"}), 500

    try:
        with open(save_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        clean = extract_text(content, filename)
        added = add_to_knowledge(clean)
        return jsonify({"message": f"✅ Learned {added} new sentences from '{filename}'."})
    except Exception as e:
        return jsonify({"error": f"⚠️ Failed to process file: {e}"}), 500

@app.route("/learn-url", methods=["POST"])
def learn_url():
    data = request.json
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "No URL provided."}), 400

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        return jsonify({"error": f"Failed to fetch URL: {e}"}), 400

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "form", "aside"]):
        tag.decompose()

    text = soup.get_text(separator=' ', strip=True)
    clean_text = re.sub(r'\s+', ' ', text).strip()

    if len(clean_text) < 20:
        return jsonify({"error": "Extracted content too short to learn from."}), 400

    added = add_to_knowledge(clean_text)
    return jsonify({"message": f"✅ Learned {added} new sentences from URL."})

@app.route("/session", methods=["GET"])
def get_session():
    return jsonify(load_session())

@app.route("/save-session", methods=["POST"])
def save_session_route():
    filename = save_session_timestamped()
    return jsonify({"message": f"💾 Session saved as {filename}."})

# ------------------- Run -------------------

if __name__ == "__main__":
    print("🚀 ChatIA running at http://127.0.0.1:5000")
    app.run(debug=True)
