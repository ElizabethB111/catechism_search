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
            "What is the communion of the saints?",
            "What is God like?",
            "What is the purpose of the sacraments?",
            "How can I make a good confession?"
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
            A resource for deepening understanding of the Catholic faith.
        </div>
    """, unsafe_allow_html=True)










