# -*- coding: utf-8 -*-
"""
Created on Sun May  3 09:46:58 2026

@author: Sidhant
"""

import streamlit as st
import os
from rag_engine import (
    process_document,
    load_vector_store,
    answer_question
)

# ============================================================
# PAGE CONFIGURATION
# Must be the first Streamlit command in the file.
# Sets the browser tab title, icon, and layout width.
# "wide" uses the full browser width instead of a narrow column.
# ============================================================

st.set_page_config(
    page_title="Leading Global Network Transaction Processing Rules Assistant",
    page_icon="💳",
    layout="wide"
)

# ============================================================
# PAGE HEADER
# ============================================================

st.title("Leading Global Network Transaction Processing Rules Assistant")
st.caption("Ask questions grounded in the official Transaction Processing Rules for Issuers and Acquirers")

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Document Setup")

    if os.path.exists("faiss_index"):
        st.success("Document already indexed")

        if st.button("Re-index document"):
            import shutil
            shutil.rmtree("faiss_index")
            st.rerun()

    else:
        st.info("Upload the Network PDF to begin")

        uploaded_file = st.file_uploader(
            "Upload Network Transaction Processing Rules PDF",
            type="pdf"
        )

        if uploaded_file:
            pdf_path = "docs/network_rules.pdf"
            os.makedirs("docs", exist_ok=True)

            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.read())

            with st.spinner("Processing document... this takes 2-3 minutes"):
                vector_store, chunk_count = process_document(pdf_path)
                st.session_state.vector_store = vector_store
                st.success(f"Indexed {chunk_count} chunks")
                st.rerun()

    st.divider()

    # ── Advanced Settings ─────────────────────────────────
    # k slider lets you experiment with retrieval quantity
    # without changing code. Caption shows estimated token
    # cost impact so you can see the trade-off directly.

    st.header("Advanced Settings")

    k_value = st.slider(
        "Chunks to retrieve (k)",
        min_value=2,
        max_value=10,
        value=5,
        step=1,
        help=(
            "More chunks = more context but higher token cost. "
            "Fewer chunks = lower cost but may miss edge cases."
        )
    )

    # Each chunk is ~200 tokens so multiply to estimate context size
    st.caption(f"Estimated context: ~{k_value * 200} tokens per query")

    st.divider()

    # ── Sample Questions ──────────────────────────────────
    # Clickable buttons pre-fill the question input.
    # Stored in session_state as pending_question so the
    # main area text box can read and display it.

    st.header("Sample Questions")

    sample_questions = [
        "What are my obligations as an issuer for 3DS authentication?",
        "What are the rules for declining a recurring transaction?",
        "What are chargeback timeframes for issuers?",
        "What is issuer liability for contactless transactions?",
        "What are the rules for issuer authorization response times?"
    ]

    for q in sample_questions:
        if st.button(q, key=q, use_container_width=True):
            st.session_state.pending_question = q


# ============================================================
# LOAD VECTOR STORE INTO SESSION STATE
# Auto-loads from disk on page refresh so user never has to
# re-upload the PDF after the first indexing run.
# ============================================================

if "vector_store" not in st.session_state:
    if os.path.exists("faiss_index"):
        with st.spinner("Loading indexed document..."):
            st.session_state.vector_store = load_vector_store()


# ============================================================
# QUESTION INPUT
# Pre-fills with pending_question if a sample button was
# clicked. Clears pending_question after loading so the user
# can edit freely without it snapping back.
# ============================================================

question = st.text_area(
    "Your question",
    value=st.session_state.get("pending_question", ""),
    placeholder="e.g. What are my authorization obligations as an issuer for 3DS transactions?",
    height=80
)

if "pending_question" in st.session_state:
    del st.session_state.pending_question


# ============================================================
# ASK BUTTON
# Narrow first column keeps button left-aligned and small.
# ============================================================

col1, col2 = st.columns([1, 5])
with col1:
    ask_button = st.button("Ask", type="primary", use_container_width=True)


# ============================================================
# MAIN Q&A LOGIC
# answer_question now returns three values:
#   answer          — GPT generated answer string
#   retrieved_chunks — list of (Document, score) tuples
#   traces          — list of three trace dicts:
#                     [query_rewrite, retrieval, generation]
# Each trace dict powers one section of the reasoning panel.
# ============================================================

if ask_button and question:

    if "vector_store" not in st.session_state:
        st.error("Please upload and index the document first")

    else:
        with st.spinner("Searching rules and generating answer..."):
            # Unpack all three return values  ← key change
            # Old: answer, retrieved_chunks = answer_question(...)
            # New: traces is the third return value
            answer, retrieved_chunks, traces = answer_question(
                question,
                st.session_state.vector_store,
                k=k_value
            )

        # ── Answer ────────────────────────────────────────

        st.subheader("Answer")
        st.markdown(answer)
        st.divider()

        # ── Reasoning Trace ───────────────────────────────
        # Shows every decision the pipeline made for this query.
        # expanded=True means the panel is open by default
        # so you see the reasoning immediately after the answer.

        with st.expander("Show reasoning trace", expanded=True):

            for trace in traces:

                # Step header and explanation of why this step exists
                st.markdown(f"**{trace['step']}**")
                st.caption(trace["reason"])

                # ── Step 1: Query Rewriting ───────────────
                # Shows original question vs rewritten keywords.
                # Side by side so you can see the transformation.

                if trace["step"] == "Query Rewriting":
                    col_a, col_b = st.columns(2)

                    with col_a:
                        st.markdown("Original question")
                        # st.code renders monospace box — good for
                        # showing the exact text without formatting
                        st.code(trace["original"], language=None)

                    with col_b:
                        st.markdown("Rewritten for retrieval")
                        st.code(trace["rewritten"], language=None)

                    st.caption(
                        f"Tokens used for rewriting: "
                        f"{trace['tokens_used']}"
                    )

                # ── Step 2: Retrieval ─────────────────────
                # Shows each retrieved chunk with its score,
                # page number, quality rating, and text preview.
                # Color coding: green=strong, orange=moderate, red=weak.

                elif trace["step"] == "Retrieval":
                    st.caption(
                        f"Query sent to vector store: "
                        f"`{trace['query_used'][:80]}...`"
                    )

                    for chunk in trace["chunks"]:

                        # Map quality to Streamlit color name
                        # :green[], :orange[], :red[] are inline
                        # color formatters in st.markdown
                        if "strong" in chunk["quality"]:
                            color = "green"
                        elif "moderate" in chunk["quality"]:
                            color = "orange"
                        else:
                            color = "red"

                        st.markdown(
                            f"Chunk {chunk['rank']} — "
                            f"Page {chunk['page']} — "
                            f"Score: `{chunk['score']}` — "
                            f":{color}[{chunk['quality']}]"
                        )

                        # Small preview of what's in the chunk
                        # so you can see if retrieval is on-topic
                        st.caption(chunk["preview"])

                # ── Step 3: Generation ────────────────────
                # Shows token breakdown per prompt section and
                # actual cost for this query.
                # Four metric cards in a row for quick scanning.

                elif trace["step"] == "Generation":
                    breakdown = trace["prompt_breakdown"]
                    usage     = trace["token_usage"]

                    col_a, col_b, col_c, col_d = st.columns(4)

                    # st.metric renders a labeled number card
                    col_a.metric(
                        "System tokens",
                        breakdown["system_prompt_tokens"]
                    )
                    col_b.metric(
                        "Context tokens",
                        breakdown["context_tokens"],
                        help="Biggest cost driver — reduce k to lower this"
                    )
                    col_c.metric(
                        "Output tokens",
                        usage["output"]
                    )
                    col_d.metric(
                        "Query cost",
                        f"${trace['cost_usd']}"
                    )

                    st.caption(
    f"Model: {trace.get('model', 'N/A')} | "
    f"Temperature: {trace.get('temperature', 0)} | " 
    f"Total tokens: {usage.get('total', 0)}"
)

                # Divider between each trace step
                st.divider()

        # ── Retrieved Sections ────────────────────────────
        # Raw chunk text in a collapsible panel.
        # Collapsed by default (expanded=False) so it doesn't
        # push the reasoning trace off screen.
        # Shows full 400-char preview of each chunk so you can
        # verify the retrieved text is relevant.

        with st.expander(
            f"Retrieved sections ({len(retrieved_chunks)} chunks)",
            expanded=False
        ):
            st.caption(
                "Exact sections retrieved from the Network rules "
                "document. The answer is grounded strictly in these."
            )

            for i, (doc, score) in enumerate(retrieved_chunks):
                st.markdown(
                    f"**Section {i+1} — Page {doc.metadata['page']}** "
                    f"*(similarity score: {round(score, 3)})*"
                )
                st.text(doc.page_content[:400] + "...")
                st.divider()