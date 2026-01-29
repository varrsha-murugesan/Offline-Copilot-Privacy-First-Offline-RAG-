import os
import glob
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama
from pdf2image import convert_from_path
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

DATA_FOLDER = "data/raw_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "phi3"   


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    pages_text = []

    for page_num, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()

        # OCR fallback for Canva / image-like PDF pages
        if len(text) < 30:
            images = convert_from_path(
                pdf_path,
                first_page=page_num + 1,
                last_page=page_num + 1
            )
            text = pytesseract.image_to_string(images[0]).strip()

        pages_text.append((page_num + 1, text))

    return pages_text


def chunk_text(text, chunk_size=350, overlap=60):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

def build_faiss_index(chunks, embedder):
    embeddings = embedder.encode(chunks, convert_to_numpy=True).astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def search_chunks(query, embedder, index, chunks, metadata, top_k=3):
    q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(q_emb, top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        results.append({
            "rank": rank + 1,
            "chunk": chunks[idx],
            "meta": metadata[idx],
            "distance": float(distances[0][rank])
        })
    return results

def ollama_health_check():
    try:
        _ = ollama.list()
        return True
    except Exception:
        return False

def answer_with_ollama(query, retrieved_chunks):
    context = "\n\n".join(
        [f"[Source {i+1}] {item['chunk']}" for i, item in enumerate(retrieved_chunks)]
    )

    prompt = f"""
You are an offline assistant.
Answer ONLY using the sources.
If not found, say: "I couldn't find that in the uploaded documents."

Question: {query}

Sources:
{context}

Give a short clear answer and cite like (Source 1).
"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]

# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Offline Copilot", page_icon="🤖", layout="wide")
st.title("🤖 Privacy-First Offline Copilot (Offline RAG)")
st.caption("Runs locally with FAISS + SentenceTransformers + Ollama. No cloud needed.")

# Init session
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "metadata" not in st.session_state:
    st.session_state.metadata = []
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "embedder" not in st.session_state:
    st.session_state.embedder = None
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False

# Sidebar
st.sidebar.header("⚙️ Setup")

pdf_files = glob.glob(os.path.join(DATA_FOLDER, "*.pdf"))

st.sidebar.write(f"📄 PDFs found: **{len(pdf_files)}**")
if len(pdf_files) == 0:
    st.sidebar.warning("Put your PDF inside: data/raw_docs/")

ollama_ok = ollama_health_check()
if ollama_ok:
    st.sidebar.success("✅ Ollama is running")
else:
    st.sidebar.error("❌ Ollama not running")
    st.sidebar.info("Open a new terminal and run:  ollama run phi3")

st.sidebar.markdown("---")

if st.sidebar.button("📌 Build Index"):
    if len(pdf_files) == 0:
        st.error("No PDFs found in data/raw_docs/")
    else:
        with st.spinner("Loading embedding model + indexing PDFs..."):
            if st.session_state.embedder is None:
                st.session_state.embedder = SentenceTransformer(EMBED_MODEL)

            all_chunks = []
            all_metadata = []

            for pdf_path in pdf_files:
                pages = extract_text_from_pdf(pdf_path)
                for page_num, page_text in pages:
                    page_text = page_text.strip()
                    if not page_text:
                        continue
                    chunks = chunk_text(page_text)
                    for c in chunks:
                        all_chunks.append(c)
                        all_metadata.append({
                            "file": os.path.basename(pdf_path),
                            "page": page_num
                        })

            if len(all_chunks) == 0:
                st.error("No text extracted from PDF. Try another PDF.")
            else:
                st.session_state.faiss_index = build_faiss_index(all_chunks, st.session_state.embedder)
                st.session_state.chunks = all_chunks
                st.session_state.metadata = all_metadata
                st.session_state.index_ready = True
                st.success(f"✅ Index built! Total chunks: {len(all_chunks)}")

st.markdown("## 💬 Ask your documents")
query = st.text_input("Ask something (example: What are my skills?)")

if st.button("Ask"):
    if not st.session_state.index_ready:
        st.warning("Build the index first from sidebar 👉")
    elif not query.strip():
        st.warning("Type a question first.")
    elif not ollama_ok:
        st.error("Ollama is not running. Run:  ollama run phi3")
    else:
        with st.spinner("Thinking..."):
            results = search_chunks(
                query=query,
                embedder=st.session_state.embedder,
                index=st.session_state.faiss_index,
                chunks=st.session_state.chunks,
                metadata=st.session_state.metadata,
                top_k=3
            )
            answer = answer_with_ollama(query, results)

        st.markdown("### ✅ Answer")
        st.write(answer)

        st.markdown("### 📌 Sources")
        for r in results:
            with st.expander(f"Source {r['rank']} | {r['meta']['file']} | Page {r['meta']['page']}"):
                st.write(r["chunk"])
