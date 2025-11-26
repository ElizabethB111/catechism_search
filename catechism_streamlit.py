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
from pathlib import Path

# -------------------------
# Assets (update filenames if needed)
# -------------------------
BACKGROUND_IMAGE = "eucharist_minimalist.png"     # your minimalist Eucharist (no text)
ROSARY_OVERLAY = "rosary_overlay.png"             # transparent rosary beads overlay
MONSTRANCE_ICON = "monstrance_icon.png"           # small golden monstrance icon
CROSS_WATERMARK = "cross_watermark.png"           # faint cross watermark
CANDLE_ICON = "candle_icon.png"                   # candle icon for flicker animation

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Catechism Search",
    page_icon=MONSTRANCE_ICON if Path(MONSTRANCE_ICON).exists() else "🔔",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------
# Helpers: load image as base64 for CSS backgrounds
# -------------------------
def _img_to_base64(path):
    try:
        with open(path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception:
        return None

bg_b64 = _img_to_base64(BACKGROUND_IMAGE)
rosary_b64 = _img_to_base64(ROSARY_OVERLAY)
monstrance_b64 = _img_to_base64(MONSTRANCE_ICON)
cross_b64 = _img_to_base64(CROSS_WATERMARK)
candle_b64 = _img_to_base64(CANDLE_ICON)

# -------------------------
# Global styles: glassmorphism, typography, icons, animation
# -------------------------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Georgia:ital@0;1');

    :root {{
        --gold: rgba(255, 215, 64, 0.98);
        --soft-gold: rgba(255, 230, 150, 0.9);
        --glass-bg: rgba(10, 10, 10, 0.36);
        --glass-border: rgba(255,255,255,0.08);
        --text-light: rgba(250,250,250,0.98);
    }}

    /* Background image + overlays */
    .stApp {{
        background-image: url("data:image/png;base64,{bg_b64}") !important;
        background-size: cover !important;
        background-position: center center !important;
        background-repeat: no-repeat !important;
        background-attachment: fixed !important;
        font-family: Georgia, serif !important;
    }}

    /* dark overlay to preserve readability while keeping artwork visible */
    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        background: linear-gradient(180deg, rgba(0,0,0,0.38), rgba(0,0,0,0.42));
        z-index: 0;
        pointer-events: none;
    }}

    /* rosary overlay top-right (subtle) */
    .rosary-overlay {{
        position: fixed;
        top: 6vh;
        right: 3vw;
        width: 12vw;
        max-width: 140px;
        opacity: 0.18;
        z-index: 1;
        background-image: url("data:image/png;base64,{rosary_b64}");
        background-size: contain;
        background-repeat: no-repeat;
        pointer-events: none;
    }}

    /* cross watermark center faint */
    .cross-watermark {{
        position: fixed;
        left: 50%;
        top: 18%;
        transform: translate(-50%, -50%);
        width: 28vw;
        max-width: 420px;
        opacity: 0.06;
        z-index: 0;
        background-image: url("data:image/png;base64,{cross_b64}");
        background-size: contain;
        background-repeat: no-repeat;
        pointer-events: none;
        filter: drop-shadow(0 0 12px rgba(255,240,200,0.03));
    }}

    /* small monstrance icon in the corner */
    .monstrance-corner {{
        position: fixed;
        left: 2.5vw;
        top: 2.2vh;
        width: 48px;
        height: 48px;
        background-image: url("data:image/png;base64,{monstrance_b64}");
        background-size: contain;
        background-repeat: no-repeat;
        z-index: 3;
        pointer-events: none;
        filter: drop-shadow(0 3px 8px rgba(0,0,0,0.35));
        opacity: 0.98;
    }}

    /* Candle flicker animation - subtle glow behind title area */
    @keyframes flicker {{
      0%   {{ opacity: 0.82; transform: translateY(0px) scale(1); filter: blur(0.0px); }}
      20%  {{ opacity: 0.9; transform: translateY(-0.6px) scale(1.002); filter: blur(.2px); }}
      40%  {{ opacity: 0.78; transform: translateY(0.3px) scale(0.999); filter: blur(.1px); }}
      60%  {{ opacity: 0.88; transform: translateY(-0.2px) scale(1.001); filter: blur(.12px); }}
      100% {{ opacity: 0.82; transform: translateY(0px) scale(1); filter: blur(0.0px); }}
    }}

    .candle-flicker {{
        position: absolute;
        right: 1.4rem;
        top: 0.25rem;
        width: 26px;
        height: 26px;
        background-image: url("data:image/png;base64,{candle_b64}");
        background-size: contain;
        background-repeat: no-repeat;
        animation: flicker 3s ease-in-out infinite;
        z-index: 4;
        opacity: 0.95;
        pointer-events: none;
    }}

    /* main headings - gold glow */
    .main-header {{
        font-size: 2.6rem;
        color: var(--text-light);
        font-weight: 700;
        margin: 0;
        letter-spacing: 0.6px;
        text-shadow:
            0 1px 0 rgba(0,0,0,0.6),
            0 0 12px rgba(255, 230, 150, 0.18),
            0 0 6px rgba(255, 215, 64, 0.08);
    }}

    .sub-header {{
        font-size: 1.05rem;
        color: rgba(245,245,245,0.92);
        margin-top: 0.25rem;
        margin-bottom: 1.2rem;
        font-weight: 400;
        font-style: italic;
        opacity: 0.95;
    }}

    /* glassmorphism card for results and inputs */
    .glass-card {{
        background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
        border: 1px solid var(--glass-border);
        backdrop-filter: blur(8px) saturate(110%);
        -webkit-backdrop-filter: blur(8px) saturate(110%);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.35);
        color: var(--text-light);
    }}

    .result-card {{
        background: linear-gradient(180deg, rgba(255,255,255,0.86), rgba(255,255,255,0.78));
        border-radius: 10px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
        color: #111;
        border-left: 6px solid rgba(255, 215, 64, 0.95);
    }}

    .paragraph-badge {{
        background: transparent;
        color: #6b4b00;
        padding: 0.22rem 0.65rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.9rem;
        display: inline-block;
        margin-bottom: 0.5rem;
        border: 1px solid rgba(0,0,0,0.06);
    }}

    .confidence-badge {{
        background: linear-gradient(90deg, rgba(255,215,64,0.12), rgba(255,215,64,0.04));
        color: #6b4b00;
        padding: 0.22rem 0.65rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        margin-left: 0.6rem;
        border: 1px solid rgba(0,0,0,0.04);
    }}

    /* subtle divider fade */
    .soft-divider {{
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
        margin: 1rem 0;
    }}

    /* ensure markdown and writes are readable */
    .stMarkdown, .stText {{
        color: var(--text-light) !important;
    }}

    /* responsiveness tweaks for title area */
    .title-and-input {{
        position: relative;
        z-index: 3;
    }}

    /* Remove default Streamlit background behind main columns to let glass show */
    .block-container {{
        padding-top: 1.2rem;
    }}

    </style>
    """,
    unsafe_allow_html=True
)

# Add floating decorative elements to DOM
st.markdown(
    """
    <div class="rosary-overlay"></div>
    <div class="cross-watermark"></div>
    <div class="monstrance-corner" aria-hidden="true"></div>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Load Models (cached)
# -------------------------
@st.cache_resource
def load_models():
    try:
        # Dataframe with paragraphs
        df = pd.read_csv("catechism_corpus_clean.csv")
        df.dropna(subset=["text"], inplace=True)

        # sentence encoder + cross-encoder reranker
        encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # faiss & embeddings
        embeddings = np.load("embeddings_store/catechism_embeddings.npy")
        index = faiss.read_index("embeddings_store/catechism_faiss.index")

        with open("embeddings_store/catechism_metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        # ensure metadata is a pandas-like dict with "text" and "paragraph"
        return encoder, reranker, index, metadata, df

    except Exception as e:
        st.error(f"❌ Error loading models or assets: {e}")
        return None, None, None, None, None

# -------------------------
# Tokenizer & BM25 setup
# -------------------------
def simple_tokenize(text):
    tokens = re.findall(r'\b\w+\b', (text or "").lower())
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
    return [t for t in tokens if t not in stopwords and len(t) > 1]

@st.cache_resource
def setup_bm25(metadata):
    try:
        tokenized = [simple_tokenize(t) for t in metadata["text"]]
        return BM25Okapi(tokenized)
    except Exception:
        return None

# -------------------------
# Small numeric helper to show confidence as percent
# -------------------------
def _sigmoid(x):
    try:
        return 1 / (1 + math.exp(-float(x)))
    except Exception:
        return 0.0

# -------------------------
# Hybrid search (keeps your previous logic, adds safety)
# -------------------------
def hybrid_search(query, encoder, reranker, index, metadata, bm25, top_k=5):
    try:
        # get query embedding
        query_emb = encoder.encode(query, normalize_embeddings=True, convert_to_numpy=True)
        D, I = index.search(np.array([query_emb]), top_k * 3)

        # semantic distances -> convert to positive similarity scores (faiss returns distances)
        # If using inner-product index D is similarity; this code tolerates either.
        semantic_scores = {}
        for r, i in enumerate(I[0]):
            val = float(D[0][r])
            semantic_scores[i] = val

        # BM25
        tokenized_q = simple_tokenize(query)
        bm25_scores = {}
        if bm25 is not None:
            bm25_all = bm25.get_scores(tokenized_q)
            bm25_top_idx = np.argsort(bm25_all)[::-1][:top_k * 3]
            for i in bm25_top_idx:
                bm25_scores[int(i)] = float(bm25_all[i])

        # combine (weights tuned toward semantic)
        combined = {}
        for i in set(list(semantic_scores.keys()) + list(bm25_scores.keys())):
            s = semantic_scores.get(i, 0.0)
            b = bm25_scores.get(i, 0.0)
            combined[i] = 0.72 * s + 0.28 * b

        # pick candidates
        candidates = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
        pairs = [(query, metadata["text"][i]) for i, _ in candidates]

        # rerank using cross-encoder
        rerank_scores = reranker.predict(pairs) if len(pairs) else []
        ranked = sorted(zip([i for i, _ in candidates], rerank_scores), key=lambda x: x[1], reverse=True)

        results = []
        for r, (idx, score) in enumerate(ranked, start=1):
            confidence = _sigmoid(score)   # map to 0..1 for friendly display
            results.append({
                "rank": r,
                "paragraph": metadata["paragraph"][idx],
                "text": metadata["text"][idx],
                "raw_score": float(score),
                "confidence": float(confidence)
            })
        return results

    except Exception as e:
        st.error(f"Search error: {e}")
        return []

# -------------------------
# Main App UI
# -------------------------
def main():
    # Header area (glass)
    st.markdown(
        """
        <div class="title-and-input">
            <div style="display:flex; align-items:center; gap:14px; justify-content:center; margin-bottom:6px;">
                <div style="display:flex; align-items:center;">
                    <h1 class="main-header">Catechism Search</h1>
                </div>
            </div>
            <div style="text-align:center; width:100%; margin-bottom:0.8rem;">
                <div class="sub-header">A Study Tool for the Catechism of the Catholic Church</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Add a small candle flicker in the top-right of the central column visually
    st.markdown('<div style="position:relative;"><div class="candle-flicker"></div></div>', unsafe_allow_html=True)

    # Load models
    with st.spinner("Loading models & embeddings…"):
        encoder, reranker, index, metadata, df = load_models()
    if encoder is None:
        st.warning("Models or data could not be loaded. Check files and model availability.")
        return

    bm25 = setup_bm25(metadata)

    # Layout columns
    col1, col2, col3 = st.columns([1, 2.4, 1])

    with col2:
        # Input glass card
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        # session state handling
        if "query" not in st.session_state:
            st.session_state.query = ""

        query = st.text_input(
            "Enter your theological question:",
            value=st.session_state.query,
            placeholder="e.g., What is the significance of baptism?",
            label_visibility="collapsed",
            key="search_input"
        )
        st.session_state.query = query

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
                if st.button(example, key=f"example_{i}"):
                    st.session_state.query = example

        st.markdown("</div>", unsafe_allow_html=True)

    # perform search if there's a query
    if st.session_state.query:
        with st.spinner("🔍 Searching Catechism passages…"):
            results = hybrid_search(
                st.session_state.query, encoder, reranker, index, metadata, bm25
            )

        if results:
            st.success(f"Found {len(results)} relevant Catechism passages")
            # results area with subtle divider
            st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

            for r in results:
                # each result uses the light result-card style (paper-like)
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown(
                    f'<div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:0.35rem;">'
                    f'<div><span class="paragraph-badge">Paragraph {r["paragraph"]}</span>'
                    f'<span class="confidence-badge">Confidence: {r["confidence"]*100:.0f}%</span></div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                # main text
                st.write(r["text"])
                st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.warning("No results found. Try rephrasing your question or using another example.")

    # footer (glass)
    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="glass-card" style="text-align:center; margin-top:10px;">
            <strong style="color:var(--gold); font-size:1.02rem;">Catechism Search</strong><br>
            <small style="color:rgba(250,250,250,0.85);">A resource for deepening understanding of the Catholic faith.</small>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()





