import os
import re
import requests
import feedparser
import fitz  # PyMuPDF
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer

# -----------------------------
# 0️⃣ Global Configurations
# -----------------------------
BASE_URL = "http://export.arxiv.org/api/query?"
QUERY = "search_query=cat:cs.CL&start=0&max_results=50&sortBy=submittedDate&sortOrder=descending"

PDF_DIR = Path("arxiv_pdfs")
TXT_DIR = Path("arxiv_texts")
PDF_DIR.mkdir(exist_ok=True)
TXT_DIR.mkdir(exist_ok=True)

MODEL_NAME = "all-MiniLM-L6-v2"

# -----------------------------
# 1️⃣ Download PDF
# -----------------------------
def safe_filename(title: str, max_len=80) -> str:
    name = re.sub(r'[\\/*?:"<>|]', "_", title)
    return name[:max_len]

def download_pdfs():
    feed = feedparser.parse(BASE_URL + QUERY)
    print("找到论文数量:", len(feed.entries))

    for entry in feed.entries:
        pdf_url = entry.id.replace("abs", "pdf") + ".pdf"
        title = safe_filename(entry.title.replace(" ", "_"))
        filename = PDF_DIR / f"{title}.pdf"

        if not filename.exists():
            try:
                r = requests.get(pdf_url)
                with open(filename, "wb") as f:
                    f.write(r.content)
                print("已下载:", filename.name)
            except Exception as e:
                print("下载失败:", pdf_url, e)

# -----------------------------
# 2️⃣ PDF → Text
# -----------------------------
def extract_text_from_pdf(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    texts = []
    for page in doc:
        text = page.get_text("text").strip()
        if text:
            texts.append(text)
    full_text = "\n".join(texts)
    return re.sub(r"\s+", " ", full_text)

def save_texts():
    documents = []
    for pdf in PDF_DIR.glob("*.pdf"):
        raw_text = extract_text_from_pdf(pdf)
        if raw_text:
            documents.append(raw_text)
            out_file = TXT_DIR / f"{pdf.stem}.txt"
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(raw_text)
            print(f"抽取完成: {pdf.name}, 文本长度: {len(raw_text)}")
    return documents

# -----------------------------
# 3️⃣ Chunking Text
# -----------------------------
def chunk_text(doc_id: str, text: str, chunk_size=500, overlap=100) -> List[Dict]:
    words = text.split()
    chunks = []
    i = 0
    chunk_id = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append({
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "text": chunk_text
        })
        i += chunk_size - overlap
        chunk_id += 1
    return chunks

# -----------------------------
# 4️⃣ Vector Indexing
# -----------------------------
def build_index(chunks: List[Dict]):
    """生成向量并建立 FAISS 索引"""
    texts = [c["text"] for c in chunks]
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, embeddings, chunks, model

# -----------------------------
# 5️⃣ Query
# -----------------------------
def search(query: str, index, model, chunks, top_k=3):
    """查询并返回最相关的文本块"""
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(q_emb, top_k)

    results = []
    for idx in I[0]:
        results.append(chunks[idx])
    return results

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Step 1: 下载论文
    download_pdfs()

    # Step 2: PDF 转文本
    documents = save_texts()

    # Step 3: 切块
    all_chunks = []
    for pdf_file in PDF_DIR.glob("*.pdf"):
        doc_id = pdf_file.stem
        text = extract_text_from_pdf(pdf_file)
        chunks = chunk_text(doc_id, text)
        all_chunks.extend(chunks)
    print(f"Total chunks: {len(all_chunks)}")

    # Step 4: 建立索引
    index, embeddings, chunks, model = build_index(all_chunks)

    # Step 5: 测试查询
    query = "What are recent methods in machine translation?"
    results = search(query, index, model, chunks)

    print("\n🔍 Search Results:")
    for r in results:
        print(f"- Doc: {r['doc_id']}, Chunk: {r['chunk_id']}, Text: {r['text'][:200]}...")
