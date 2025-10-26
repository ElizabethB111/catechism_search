# catechism_api.py
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from rank_bm25 import BM25Okapi
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
import json

print("Loading Catechism search system...")

# Load data
df = pd.read_csv("catechism_corpus_clean.csv")
df.dropna(subset=["text"], inplace=True)

# Load models
encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Load pre-computed embeddings and index
embeddings = np.load("embeddings_store/catechism_embeddings.npy")
index = faiss.read_index("embeddings_store/catechism_faiss.index")

with open("embeddings_store/catechism_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

def simple_tokenize(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
    basic_stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
    return [t for t in tokens if t not in basic_stopwords and len(t) > 1]

tokenized_corpus = [simple_tokenize(t) for t in metadata["text"]]
bm25 = BM25Okapi(tokenized_corpus)
best_alpha = 0.3

def expand_query(query, encoder, top_k=3):
    expansions = set()
    expansions.add(query)
    return list(expansions)[:5]

def hybrid_search(query, top_k=10, alpha=0.5, rerank_top_n=5, expand=True):
    queries = expand_query(query, encoder) if expand else [query]
    all_hybrid_scores = {}

    for q in queries:
        query_emb = encoder.encode(q, normalize_embeddings=True, convert_to_numpy=True)
        D, I = index.search(np.array([query_emb]), top_k * 2)
        semantic_scores = {i: float(D[0][rank]) for rank, i in enumerate(I[0])}

        tokenized_q = simple_tokenize(q)
        bm25_scores_all = bm25.get_scores(tokenized_q)
        bm25_top = np.argsort(bm25_scores_all)[::-1][:top_k * 2]
        bm25_scores = {i: float(bm25_scores_all[i]) for i in bm25_top}

        for i in set(semantic_scores.keys()) | set(bm25_scores.keys()):
            bm = bm25_scores.get(i, 0)
            sem = semantic_scores.get(i, 0)
            all_hybrid_scores[i] = all_hybrid_scores.get(i, 0) + alpha * bm + (1 - alpha) * sem

    top_candidates = sorted(all_hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:rerank_top_n]
    candidate_pairs = [(query, metadata["text"][i]) for i, _ in top_candidates]

    if candidate_pairs:
        rerank_scores = reranker.predict(candidate_pairs)
        rerank_probs = 1 / (1 + np.exp(-np.array(rerank_scores)))
        reranked = sorted(zip([i for i, _ in top_candidates], rerank_probs), key=lambda x: x[1], reverse=True)
    else:
        reranked = []

    results = []
    for rank, (idx, score) in enumerate(reranked, start=1):
        results.append({
            "rank": rank,
            "paragraph": metadata["paragraph"][idx],
            "text": metadata["text"][idx],
            "rerank_score": float(score)
        })
    return results

app = Flask(__name__)
CORS(app)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        return super().default(obj)

app.json_encoder = NumpyEncoder

@app.route('/')
def home():
    return jsonify({
        "message": "Catechism Hybrid Retrieval API",
        "version": "1.0",
        "status": "active"
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

@app.route('/info', methods=['GET'])
def system_info():
    return jsonify({
        "paragraphs": len(df),
        "best_alpha": best_alpha
    })

@app.route('/search', methods=['POST'])
def search_catechism():
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        results = hybrid_search(query)
        
        formatted_results = []
        for result in results:
            formatted_results.append({
                "rank": int(result["rank"]),
                "paragraph": int(result["paragraph"]) if str(result["paragraph"]).isdigit() else result["paragraph"],
                "score": float(result["rerank_score"]),
                "text": result["text"],
                "preview": result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"]
            })
        
        return jsonify({
            "query": query,
            "results": formatted_results
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/search_simple', methods=['GET'])
def search_simple():
    try:
        query = request.args.get('query', '').strip()
        
        if not query:
            return jsonify({"error": "Query parameter is required"}), 400
        
        results = hybrid_search(query, top_k=5, rerank_top_n=5)
        
        simplified_results = []
        for result in results:
            simplified_results.append({
                "rank": int(result["rank"]),
                "paragraph": int(result["paragraph"]) if str(result["paragraph"]).isdigit() else result["paragraph"],
                "score": float(result["rerank_score"]),
                "text": result["text"][:300] + "..." if len(result["text"]) > 300 else result["text"]
            })
        
        return jsonify({
            "query": query,
            "results": simplified_results
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Catechism API Server...")
    print("📚 Available at: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
