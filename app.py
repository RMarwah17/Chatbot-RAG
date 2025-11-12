import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# ==== CONFIG ====
st.set_page_config(page_title="RAG Chatbot Demo", page_icon="🤖")

# Masukkan API key Gemini
genai.configure(api_key=st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY"))

MODEL = "models/gemini-2.5-flash"
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ==== LOAD DATA ====
with open("data/produk.txt", "r", encoding="utf-8") as f:
    docs = f.readlines()

doc_embeddings = embedder.encode(docs)
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(np.array(doc_embeddings))

# ==== STREAMLIT UI ====
st.title("🤖 Chatbot RAG dengan Gemini 2.5 Flash")
st.caption("Menggunakan RAG untuk menjawab berdasarkan dokumen lokal")

query = st.text_input("Tanyakan sesuatu tentang produk:")
if query:
    q_emb = embedder.encode([query])
    D, I = index.search(np.array(q_emb), k=2)
    context = "\n".join([docs[i] for i in I[0]])

    prompt = f"""
    Kamu adalah asisten produk. Jawablah pertanyaan berikut berdasarkan konteks:
    KONTEKS:
    {context}

    PERTANYAAN:
    {query}
    """

    model = genai.GenerativeModel(MODEL)
    response = model.generate_content(prompt)
    st.write("### 💬 Jawaban:")
    st.write(response.text)
