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

# Override theme colors for slider
st.markdown("""
    <style>
    :root {
        --primary-color: #ffffff;
    }
    </style>
""", unsafe_allow_html=True)

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

    /* Style the slider to be white */
    .stSlider label {
        color: #f8f8f8 !important;
        font-weight: 500;
        text-shadow: 1px 1px 2px black;
    }
    /* Change the slider track color */
    .stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {
        color: #f8f8f8 !important;
    }
    /* Change the active/filled portion of the slider */
    .stSlider [data-baseweb="slider"] > div > div > div > div {
        background-color: #ffffff !important;
    }
    /* Change the slider thumb (circle) */
    .stSlider [data-baseweb="slider"] > div > div > div > div[role="slider"] {
        background-color: #ffffff !important;
        box-shadow: 0 0 0 2px #ffffff !important;
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

        ranked = sorted(zip([i for i, _ in candidates], rerank_scores), key=lambda x: x[1], reverse=True)

        results = []
        for r, (idx, _) in enumerate(ranked, start=1):
            results.append({
                "rank": r,
                "paragraph": metadata["paragraph"][idx],
                "text": metadata["text"][idx]
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
            <h1 class="main-header">Catechism of the Catholic Church</h1>
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
        # Small subtitle
        st.markdown("""
            <div style="text-align: center; margin-bottom: 2rem;">
                <p style="color: #f0f0f0; font-size: 1.1rem; font-weight: 300; text-shadow: 1px 1px 2px black;">
                    Find paragraphs and teachings from the official Catechism
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Ask Question header
        st.markdown("""
            <div style="text-align: center; margin-bottom: 1.5rem;">
                <h2 style="color: #f8f8ff; font-weight: 600; text-shadow: 1px 1px 2px black;">
                    Ask a Question
                </h2>
            </div>
        """, unsafe_allow_html=True)

        # Initialize
        if "query" not in st.session_state:
            st.session_state.query = ""
        if "num_results" not in st.session_state:
            st.session_state.num_results = 5

        # Number of results slider
        num_results = st.slider(
            "Number of results to display:",
            min_value=1,
            max_value=20,
            value=st.session_state.num_results,
            step=1,
            key="num_results_slider"
        )
        st.session_state.num_results = num_results

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
            "Why do Catholics pray to Mary?",
            "What is God like?",
            "What is the purpose of the sacraments?",
            "How can I make a good confession?",
            "What is human dignity?",
            "What is the communion of saints?"
        ]

        for i, example in enumerate(example_questions):
            with examples[i % 2]:
                if st.button(example, key=f"ex_{i}"):
                    st.session_state.query = example

    # === Search Results ===
    if st.session_state.query:
        with st.spinner("🔍 Searching 3,260 Catechism paragraphs..."):
            results = hybrid_search(
                st.session_state.query, encoder, reranker, index, metadata, bm25,
                top_k=st.session_state.num_results
            )

        if results:
            # Light "Found X passages" message
            st.markdown(
                f'<div style="color:#f8f8f8; font-weight:bold; font-size:1rem; margin-bottom:0.5rem;">'
                f'Found {len(results)} relevant Catechism passages</div>',
                unsafe_allow_html=True
            )

            for r in results:
                st.markdown(
                    f'''
                    <div class="result-card" style="background-color: rgba(0,0,0,0.6); color: #f8f8f8; padding: 1rem; border-radius: 0 8px 8px 0; margin: 1rem 0;">
                        <span class="paragraph-badge">Paragraph {r["paragraph"]}</span>
                        <div style="margin-top:0.5rem; line-height:1.5;">{r["text"]}</div>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )

        else:
            st.warning("No results found. Try rephrasing your question.")

    # Footer
    st.markdown("---")
    st.markdown("""
        <div class="footer">
            <strong>Catechism Search</strong><br>
            A resource for deepening understanding of the Catholic faith using SentenceTransformers, FAISS, BM25, and CrossEncoder.
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()







