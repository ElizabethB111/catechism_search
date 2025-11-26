# catechism_streamlit.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from rank_bm25 import BM25Okapi
import re
import base64
import math

# === Page Config ===
st.set_page_config(
    page_title="Catechism Search",
    page_icon="https://raw.githubusercontent.com/ElizabethB111/catechism_search/main/icons8-catholic-50%203.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# === Background Image ===
def set_background(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* Dark overlay for readability */
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.35);
            z-index: 0;
        }}

        .main > div {{
            position: relative;
            z-index: 1;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("Pentecost_wp.jpg")

# === Custom CSS ===
st.markdown("""
<style>
    /* Global light text */
    body, .stApp, .main > div {
        color: #f0f0f0 !important;
    }

    .main-header {
        font-size: 3rem;
        color: #f8f8ff;
        font-weight: 700;
        margin: 0;
        text-shadow: 1px 1px 3px black;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #f0f0f0;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
        text-shadow: 1px 1px 2px black;
    }
    .result-card {
        background-color: rgba(255,255,255,0.85);
        border-left: 4px solid #1f3d7a;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        color: #111;
    }
    .paragraph-badge {
        background-color: #1f3d7a;
        color: #ffffff;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: bold;
        font-size: 0.9rem;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .confidence-badge {
        background-color: #28a745;
        color: #ffffff;
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
        color: #f0f0f0;
        text-shadow: 1px 1px 2px black;
    }

    /* Make all Streamlit buttons have dark text */
    .stButton>button {
        color: #111 !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# === Load Models ===
@st.cache_resource
def load_models():
    try:
        df = pd.read_csv("catechism_corpus_clean.csv")
        df.dropna(subset=["text"], inplace=True)

        encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        embeddings = np.load("embeddings_store/catechism_embeddings.npy")
        index = faiss.read_index("embeddings_store/catechism_faiss.index")

        with open("embeddings_store/catechism_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        return encoder, reranker, index, metadata, df

    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None, None, None, None

# === Tokenizer & BM25 ===
def simple_tokenize(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
    return [t for t in tokens if t not in stopwords and len(t) > 1]

@st.cache_resource
def setup_bm25(metadata):
    tokenized = [simple_tokenize(t) for t in metadata["text"]]
    return BM25Okapi(tokenized)

# === Hybrid Search ===
def hybrid_search(query, encoder, reranker, index, metadata, bm25, top_k=5):
    try:
        query_emb = encoder.encode(query, normalize_embeddings=True, convert_to_numpy=True)
        D, I = index.search(np.array([query_emb]), top_k * 3)
        semantic_scores = {i: float(D[0][r]) for r, i in enumerate(I[0])}

        tokenized_q = simple_tokenize(query)
        bm25_all = bm25.get_scores(tokenized_q)
        bm25_top_idx = np.argsort(bm25_all)[::-1][:top_k * 3]
        bm25_scores = {i: float(bm25_all[i]) for i in bm25_top_idx}

        combined = {}
        for i in set(semantic_scores) | set(bm25_scores):
            combined[i] = 0.7 * semantic_scores.get(i, 0) + 0.3 * bm25_scores.get(i, 0)

        candidates = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
        pairs = [(query, metadata["text"][i]) for i, _ in candidates]

        rerank_scores = reranker.predict(pairs)

        # Convert raw scores to probabilities using sigmoid
        prob_scores = [1 / (1 + math.exp(-s)) for s in rerank_scores]

        ranked = sorted(zip([i for i, _ in candidates], prob_scores), key=lambda x: x[1], reverse=True)

        results = []
        for r, (idx, score) in enumerate(ranked, start=1):
            results.append({
                "rank": r,
                "paragraph": metadata["paragraph"][idx],
                "text": metadata["text"][idx],
                "score": float(score)
            })
        return results

    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

# === Main App ===
def main():

    # Header (Catechism Search)
    st.markdown("""
        <div style="display:flex; align-items:center; justify-content:center; margin-bottom:0.5rem;">
            <h1 class="main-header">Catechism Search</h1>
        </div>
        <p class="sub-header">&nbsp;</p>
    """, unsafe_allow_html=True)

    # Load Models
    encoder, reranker, index, metadata, df = load_models()
    if encoder is None:
        return

    bm25 = setup_bm25(metadata)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # Ask Question header (spaced down)
        st.markdown("""
            <div style="display:flex; align-items:center; justify-content:center; margin-bottom:1rem; margin-top:2rem;">
                <h1 class="main-header">Ask a Question</h1>
            </div>
            <p class="sub-header">&nbsp;</p>
        """, unsafe_allow_html=True)

        # Initialize
        if "query" not in st.session_state:
            st.session_state.query = ""

        # Input
        query = st.text_input(
            "Enter your theological question:",
            value=st.session_state.query,
            placeholder="e.g., What is the significance of baptism?",
            label_visibility="collapsed",
            key="search_input"
        )
        st.session_state.query = query

        # Example buttons
        st.write("**Quick Examples:**")
        examples = st.columns(2)
        example_questions = [
            "What is the communion of the saints?",
            "How can I pray?",
            "What is the purpose of the sacraments?",
            "How does one make a good confession?"
        ]

        for i, example in enumerate(example_questions):
            with examples[i % 2]:
                if st.button(example, key=f"ex_{i}"):
                    st.session_state.query = example

    # Search Results
    if st.session_state.query:
        with st.spinner("🔍 Searching 3,260 Catechism paragraphs..."):
            results = hybrid_search(
                st.session_state.query, encoder, reranker, index, metadata, bm25
            )

        if results:
            st.success(f"Found {len(results)} relevant Catechism passages")

            for r in results:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown(
                    f'<span class="paragraph-badge">Paragraph {r["paragraph"]}</span>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<span class="confidence-badge">Confidence: {r["score"]:.1%}</span>',
                    unsafe_allow_html=True
                )
                st.write(r["text"])
                st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.warning("No results found. Try rephrasing your question.")

    # Footer
    st.markdown("---")
    st.markdown("""
        <div class="footer">
            <strong>Catechism Search</strong><br>
            A resource for deepening understanding of the Catholic faith.
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()










