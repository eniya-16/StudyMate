import os, traceback
from flask import Flask, render_template, request, jsonify
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss

app = Flask(__name__)

# === MODELS (load once) ===
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
llm = pipeline("text-generation", model="ibm-granite/granite-3.3-2b-instruct")  # CPU

doc_chunks = []
index = None

# === UTILITIES ===
def normalize(embs):
    embs = np.array(embs, dtype="float32")
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms==0]=1
    return embs/norms

def chunk_text(text, chunk_size_words=250, overlap_words=50):
    words = text.split()
    chunks, start = [],0
    while start < len(words):
        end = min(start+chunk_size_words,len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk: chunks.append(chunk)
        start += chunk_size_words-overlap_words
    return chunks

def extract_text(path):
    doc = fitz.open(path)
    texts=[]
    for page in doc:
        text = page.get_text("text")
        if text.strip():
            texts.append(text.replace("\n"," ").strip())
    return texts

# === ROUTES ===
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    global doc_chunks, index
    try:
        pdf = request.files["pdf"]
        if not pdf: return jsonify({"status":"error","message":"No PDF uploaded"})
        path = os.path.join("uploads", pdf.filename)
        os.makedirs("uploads", exist_ok=True)
        pdf.save(path)

        texts = extract_text(path)
        if not texts:
            return jsonify({"status":"error","message":"❌ No extractable text found."})

        full_text = " ".join(texts)
        doc_chunks = chunk_text(full_text)
        embeddings = embedder.encode(doc_chunks, convert_to_numpy=True)
        embeddings = normalize(embeddings).astype("float32")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        return jsonify({"status":"ok","message":f"✅ Processed {len(doc_chunks)} chunks."})
    except Exception as e:
        return jsonify({"status":"error","message":str(e)+traceback.format_exc()})

@app.route("/ask", methods=["POST"])
def ask():
    global doc_chunks, index
    try:
        if index is None or not doc_chunks:
            return jsonify({"answer":"⚠️ Upload and process a PDF first."})
        q = request.json.get("question","")
        # Embed query
        q_emb = embedder.encode([q], convert_to_numpy=True)
        q_emb = normalize(q_emb).astype("float32")
        # Search top 5 chunks only (fast)
        k = min(5,len(doc_chunks))
        D,I = index.search(q_emb,k)
        retrieved = [doc_chunks[i] for i in I[0]]
        context = "\n\n".join(retrieved)
        prompt = f"Answer using ONLY the context below. If not found, reply 'I don't know'.\n\nContext:\n{context}\n\nQuestion:{q}\nAnswer:"

        # CPU-safe call
        resp = llm(prompt, max_new_tokens=100)
        ans = resp[0]["generated_text"]
        if ans.startswith(prompt): ans = ans[len(prompt):].strip()
        return jsonify({"answer":ans})
    except Exception as e:
        return jsonify({"answer":"⚠️ Error: "+str(e)})

if __name__=="__main__":
    app.run(debug=True, port=7860)
