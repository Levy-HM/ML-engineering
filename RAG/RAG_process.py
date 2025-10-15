import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from fastapi import FastAPI
import uvicorn
from pathlib import Path
import re
import asyncio
import nest_asyncio

# 修复事件循环问题
nest_asyncio.apply()

# -----------------------------
# 0️⃣ 全局配置
# -----------------------------
TXT_DIR = Path("arxiv_texts")
MODEL_NAME = "all-MiniLM-L6-v2"

# -----------------------------
# 1️⃣ 文本处理
# -----------------------------
def chunk_text(doc_id: str, text: str, chunk_size=512, overlap=50) -> list:
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

def load_chunks_from_files():
    """直接从文本文件加载所有chunks"""
    all_chunks = []
    for txt_file in TXT_DIR.glob("*.txt"):
        doc_id = txt_file.stem
        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        if text.strip():
            chunks = chunk_text(doc_id, text)
            all_chunks.extend(chunks)
    
    print(f"Loaded {len(all_chunks)} chunks from {len(list(TXT_DIR.glob('*.txt')))} documents")
    return all_chunks

# -----------------------------
# 2️⃣ 混合索引类（完全内存版）
# -----------------------------
class HybridIndex:
    def __init__(self, embed_model="all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(embed_model)
        self.index = None
        self.bm25 = None
        self.chunks = []  # 存储所有chunk信息
        self.chunk_texts = []  # 存储所有chunk文本

    def build_indices(self):
        # 加载数据
        self.chunks = load_chunks_from_files()
        self.chunk_texts = [chunk["text"] for chunk in self.chunks]
        
        if not self.chunks:
            print("Warning: No chunks found")
            return

        # 构建FAISS索引
        embeddings = self.embedder.encode(self.chunk_texts, convert_to_numpy=True)
        embeddings = np.array(embeddings).astype("float32")
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

        # 构建BM25索引
        tokenized_docs = [doc.split() for doc in self.chunk_texts]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        print(f"Built indices with {len(self.chunks)} chunks")

    def faiss_search(self, query, k=3):
        if self.index is None:
            return []
        
        q_vec = self.embedder.encode([query], convert_to_numpy=True).astype("float32")
        D, I = self.index.search(q_vec, k)
        
        results = []
        for i, d in zip(I[0], D[0]):
            if i < len(self.chunks):
                # 将距离转换为相似度分数 (0-1)，并转换为Python float
                similarity = float(1 / (1 + d))
                results.append((i, similarity))
        return results

    def bm25_search(self, query, k=3):
        if not self.bm25:
            return []
        
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # 获取前k个结果
        top_indices = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_indices:
            if idx < len(self.chunks):
                # 归一化BM25分数到0-1范围，并转换为Python float
                normalized_score = float(min(scores[idx] / 20, 1.0))
                results.append((idx, normalized_score))
        return results

    def hybrid_search(self, query, k=3, alpha=0.6):
        faiss_results = dict(self.faiss_search(query, k*2))
        bm25_results = dict(self.bm25_search(query, k*2))
        
        # 合并和重新排序
        all_indices = set(list(faiss_results.keys()) + list(bm25_results.keys()))
        combined = []
        
        for idx in all_indices:
            v_score = faiss_results.get(idx, 0.0)
            k_score = bm25_results.get(idx, 0.0)
            
            # 加权融合，确保结果为Python float
            combined_score = float(alpha * v_score + (1 - alpha) * k_score)
            combined.append((idx, combined_score))
        
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:k]

# -----------------------------
# 3️⃣ FastAPI服务
# -----------------------------
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时
    global hindex
    hindex = HybridIndex()
    hindex.build_indices()
    yield
    # 关闭时（无需清理）

app = FastAPI(lifespan=lifespan)
hindex = None

@app.get("/hybrid_search")
async def hybrid_search(query: str, k: int = 3, alpha: float = 0.6):
    if hindex is None:
        return {"error": "Index not initialized"}
    
    results = hindex.hybrid_search(query, k, alpha)
    
    # 获取结果详情
    detailed_results = []
    for idx, score in results:
        if idx < len(hindex.chunks):
            chunk = hindex.chunks[idx]
            detailed_results.append({
                "doc_id": chunk["doc_id"],
                "chunk_id": chunk["chunk_id"],
                "score": round(score, 4),  # 已经是Python float
                "content": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"]
            })
    
    return {
        "query": query,
        "alpha": alpha,
        "results": detailed_results
    }

@app.get("/search_comparison")
async def search_comparison(query: str, k: int = 3):
    """比较三种搜索方法"""
    if hindex is None:
        return {"error": "Index not initialized"}
    
    vector_results = hindex.faiss_search(query, k)
    bm25_results = hindex.bm25_search(query, k)
    hybrid_results = hindex.hybrid_search(query, k)
    
    # 格式化结果
    def format_results(indices_scores):
        results = []
        for idx, score in indices_scores:
            if idx < len(hindex.chunks):
                chunk = hindex.chunks[idx]
                results.append({
                    "doc_id": chunk["doc_id"],
                    "chunk_id": chunk["chunk_id"],
                    "score": round(score, 4)  # 转换为Python float并保留4位小数
                })
        return results
    
    return {
        "query": query,
        "vector_search": format_results(vector_results),
        "bm25_search": format_results(bm25_results),
        "hybrid_search": format_results(hybrid_results)
    }

@app.get("/health")
async def health_check():
    return {
        "status": "ready", 
        "index_ready": hindex is not None,
        "chunk_count": len(hindex.chunks) if hindex else 0
    }

@app.get("/")
async def root():
    return {
        "message": "Hybrid Search API is running",
        "endpoints": {
            "/hybrid_search": "Hybrid search with query parameter",
            "/search_comparison": "Compare different search methods",
            "/health": "Health check"
        }
    }

# -----------------------------
# 4️⃣ 服务器运行函数
# -----------------------------
def run_server():
    """运行服务器的函数"""
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    
    print("Starting server at http://localhost:8000")
    print("Endpoints:")
    print("  GET /hybrid_search?query=your_query")
    print("  GET /search_comparison?query=your_query") 
    print("  GET /health")
    print("  GET /")
    print("\nPress Ctrl+C to stop the server")
    
    # 运行服务器（这会阻塞）
    server.run()

# -----------------------------
# 5️⃣ 主运行
# -----------------------------
if __name__ == "__main__":
    # 确保目录存在
    TXT_DIR.mkdir(exist_ok=True)
    
    # 先进行快速测试
    print("Running quick test...")
    test_index = HybridIndex()
    test_index.build_indices()
    
    if test_index.chunks:
        test_queries = ["machine translation", "transformer", "neural network", "deep learning", "artificial intelligence", "natural language processing", "computer vision", "reinforcement learning", "generative models", "self-supervised learning"]

                        
        for query in test_queries:
            results = test_index.hybrid_search(query, 2)
            print(f"\nQuery: {query}")
            for idx, score in results:
                if idx < len(test_index.chunks):
                    chunk = test_index.chunks[idx]
                    print(f"  Score: {score:.3f} | Doc: {chunk['doc_id']} | Chunk: {chunk['chunk_id']}")
    else:
        print("No chunks found for testing")
    
    print("\n" + "="*50)
    
    # 启动服务器（这会阻塞，直到手动停止）
    run_server()
