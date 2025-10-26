# catechism_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from rank_bm25 import BM25Okapi
import re
import os

# Professional page config
st.set_page_config(
    page_title="Catechism Search",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f3d7a;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    .result-card {
        background-color: #f8f9fa;
        border-left: 4px solid #1f3d7a;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .paragraph-badge {
        background-color: #1f3d7a;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: bold;
        font-size: 0.9rem;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .confidence-badge {
        background-color: #28a745;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.9rem;
        display: inline-block;
        margin-left: 0.5rem;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        color: #666;
        border-top: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load models with caching for performance"""
    with st.spinner("🔄 Loading AI models and Catechism database..."):
        try:
            # Load data
            df = pd.read_csv("catechism_corpus_clean.csv")
            df.dropna(subset=["text"], inplace=True)
            
            # Load models
            encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
            reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            
            # Load pre-computed embeddings
            embeddings = np.load("embeddings_store/catechism_embeddings.npy")
            index = faiss.read_index("embeddings_store/catechism_faiss.index")
            
            with open("embeddings_store/catechism_metadata.pkl", "rb") as f:
                metadata = pickle.load(f)
                
          
            return encoder, reranker, index, metadata, df
            
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            return None, None, None, None, None

def simple_tokenize(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
    basic_stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
    return [t for t in tokens if t not in basic_stopwords and len(t) > 1]

@st.cache_resource
def setup_bm25(metadata):
    tokenized_corpus = [simple_tokenize(t) for t in metadata["text"]]
    return BM25Okapi(tokenized_corpus)

def hybrid_search(query, encoder, reranker, index, metadata, bm25, top_k=5):
    """Search function optimized for performance"""
    try:
        # Semantic search
        query_emb = encoder.encode(query, normalize_embeddings=True, convert_to_numpy=True)
        D, I = index.search(np.array([query_emb]), top_k * 3)
        semantic_scores = {i: float(D[0][rank]) for rank, i in enumerate(I[0])}

        # Lexical search
        tokenized_q = simple_tokenize(query)
        bm25_scores_all = bm25.get_scores(tokenized_q)
        bm25_top = np.argsort(bm25_scores_all)[::-1][:top_k * 3]
        bm25_scores = {i: float(bm25_scores_all[i]) for i in bm25_top}

        # Combine scores
        all_scores = {}
        for i in set(semantic_scores.keys()) | set(bm25_scores.keys()):
            sem = semantic_scores.get(i, 0)
            lex = bm25_scores.get(i, 0)
            all_scores[i] = 0.7 * sem + 0.3 * lex  # Weight towards semantic

        # Get top candidates
        top_candidates = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        candidate_pairs = [(query, metadata["text"][i]) for i, _ in top_candidates]

        # Rerank
        if candidate_pairs:
            rerank_scores = reranker.predict(candidate_pairs)
            reranked = sorted(zip([i for i, _ in top_candidates], rerank_scores), 
                            key=lambda x: x[1], reverse=True)
        else:
            reranked = []

        # Format results
        results = []
        for rank, (idx, score) in enumerate(reranked, start=1):
            results.append({
                "rank": rank,
                "paragraph": metadata["paragraph"][idx],
                "text": metadata["text"][idx],
                "score": float(score)
            })
        return results
        
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

def main():
    # Professional header
    st.markdown('<h1 class="main-header">📚 Catechism AI Search</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Enterprise Semantic Search Platform • Catechism of the Catholic Church</p>', unsafe_allow_html=True)
    
    # Load models
    encoder, reranker, index, metadata, df = load_models()
    
    if encoder is None:
        st.error("Failed to load required models. Please check if all data files are available.")
        return
    
    bm25 = setup_bm25(metadata)
    
    # Search interface
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.subheader("🔍 Ask a Question")
        query = st.text_input(
            "Enter your theological question:",
            placeholder="e.g., What is the significance of baptism?",
            label_visibility="collapsed"
        )
        
        # Quick examples
        st.write("**Quick Examples:**")
        examples = st.columns(2)
        example_questions = [
            "What is the meaning of the resurrection?",
            "Why do Catholics venerate Mary?",
            "What is the purpose of the sacraments?",
            "How does one make a good confession?"
        ]
        
        for i, example in enumerate(example_questions):
            with examples[i % 2]:
                if st.button(example, use_container_width=True):
                    st.session_state.last_query = example
                    st.rerun()
    
    # Search execution
    if query:
        with st.spinner("🔍 Searching 3,260 Catechism paragraphs..."):
            results = hybrid_search(query, encoder, reranker, index, metadata, bm25, top_k=5)
        
        if results:
            st.success(f"✅ Found {len(results)} relevant Catechism passages")
            
            # Display results
            for result in results:
                with st.container():
                    st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
                    st.markdown(f'<span class="paragraph-badge">Paragraph {result["paragraph"]}</span>', unsafe_allow_html=True)
                    st.markdown(f'<span class="confidence-badge">Confidence: {result["score"]:.1%}</span>', unsafe_allow_html=True)
                    st.write("")  # Spacing
                    st.write(result["text"])
                    st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("No results found. Try rephrasing your question or using different keywords.")
    
    # Professional footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <strong>Catechism Search Platform</strong><br>
        Powered by Sentence Transformers • FAISS Vector Search • Hybrid AI Algorithms<br>
        Processing 3,260 Catechism paragraphs with enterprise-grade accuracy
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
