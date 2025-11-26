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
        color: #111;  /* readable on white card */
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
        padding: 0.3








