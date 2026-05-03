# -*- coding: utf-8 -*-
"""
Refined RAG Engine - May 2026
Upgrades: Parent-Child Retrieval for Stronger Similarity Scores.
"""

import os
import re
import tiktoken
from dotenv import load_dotenv
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

def count_tokens(text, model="gpt-4o-mini"):
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

# ============================================================
# STAGE 1: DOCUMENT LOADING + TOC FILTERING
# ============================================================

def load_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text_blocks = page.get_text("blocks")
        full_text = "\n".join([block[4] for block in text_blocks])

        # FILTER: Skip Table of Contents (high density of dots)
        if full_text.count("....") > 15 or len(full_text.strip()) < 100:
            continue

        pages.append({
            "page_number": page_num + 1,
            "text": full_text
        })

    print(f"Loaded {len(pages)} content-rich pages.")
    return pages

# ============================================================
# STAGE 2: CHUNKING (PARENT-CHILD LOGIC)
# ============================================================

def chunk_documents(pages):
    # This splitter creates the 'Child' chunks for high-accuracy retrieval
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, 
        chunk_overlap=50,
        separators=["\n", ". ", " "]
    )

    # This pattern identifies the 'Parent' section boundaries
    section_pattern = re.compile(r'\n(?=\d+\.\d+|Chapter \d+|CHAPTER \d+)')

    all_chunks = []
    
    for page_data in pages:
        full_page_text = page_data["text"]
        
        # We treat each identified section on a page as a 'Parent'
        sections = section_pattern.split(full_page_text)
        
        for section in sections:
            parent_context = section.strip()
            if len(parent_context) < 50:
                continue
                
            # Create small children from this specific parent section
            children = child_splitter.split_text(parent_context)
            
            for child in children:
                all_chunks.append(Document(
                    page_content=child.strip(),
                    metadata={
                        "page": page_data["page_number"],
                        "parent_context": parent_context, # The 'Big' context for the LLM
                        "source": "Mastercard Rules"
                    }
                ))
    
    print(f"Created {len(all_chunks)} child chunks for vector indexing.")
    return all_chunks

# ============================================================
# STAGE 3: VECTOR STORE
# ============================================================

def build_vector_store(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def load_vector_store():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# ============================================================
# STAGE 4a: DENSE QUERY REWRITING
# ============================================================

def rewrite_query_for_retrieval(question):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    rewrite_prompt = f"""You are an expert in Mastercard Transaction Processing Rules. 
Convert the user's question into a technical, dense descriptive paragraph 
using official Mastercard terminology (e.g., 'Mastercard Identity Check', 'DE 48', 'ISO 8583'). 

Question: {question}

Technical Retrieval Paragraph:"""

    response = llm.invoke(rewrite_prompt)
    rewritten = response.content.strip()

    trace = {
        "step": "Query Rewriting",
        "original": question,
        "rewritten": rewritten,
        "tokens_used": count_tokens(rewrite_prompt) + count_tokens(rewritten),
        "reason": "Used Parent-Child indexing. Searching with small semantic units to increase match scores."
    }
    return rewritten, trace

# ============================================================
# STAGE 4b: RETRIEVAL
# ============================================================

def retrieve_relevant_chunks(question, vector_store, k=5):
    results = vector_store.similarity_search_with_score(question, k=k)
    chunk_traces = []
    
    for i, (doc, score) in enumerate(results):
        # Adjusted thresholds for 2026: L2 scores are more sensitive with smaller chunks
        if score < 0.45: quality = "strong match"
        elif score < 0.65: quality = "moderate match"
        else: quality = "weak match"

        chunk_traces.append({
            "rank": i + 1,
            "page": doc.metadata["page"],
            "score": round(score, 4),
            "quality": quality,
            "preview": doc.page_content[:120].strip() + "..."
        })

    trace = {
        "step": "Retrieval",
        "query_used": question,
        "chunks": chunk_traces,
        "reason": "Retrieved 'Child' chunks (400 chars) for maximum similarity score accuracy."
    }
    return results, trace

# ============================================================
# STAGE 5: GENERATION (PARENT-AWARE)
# ============================================================

def generate_answer(question, retrieved_chunks):
    context_parts = []
    # Deduplicate parent contexts so we don't send the same large section twice
    seen_parents = set()
    
    for i, (doc, score) in enumerate(retrieved_chunks):
        p_context = doc.metadata["parent_context"]
        if p_context not in seen_parents:
            context_parts.append(f"[Source: Page {doc.metadata['page']}]\n{p_context}")
            seen_parents.add(p_context)
            
    context = "\n\n---\n\n".join(context_parts)

    system_prompt = (
        "You are a Mastercard compliance expert. Answer strictly using the provided sections. "
        "Differentiate roles: Issuers vs Acquirers vs Merchants. "
        "If a specific Data Element (DE) is mentioned, provide its full context."
    )

    user_prompt = f"""CONTEXT FROM MASTERCARD RULES:
{context}

---
QUESTION: {question}

INSTRUCTIONS:
1. Provide 'Key Rules' citing Page Numbers.
2. List 'Exceptions' only if clearly stated.
3. List 'Issuer Action Required' based on the role defined in the text."""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])

    usage = response.response_metadata.get("token_usage", {})
    actual_input = usage.get("prompt_tokens", 0)
    actual_output = usage.get("completion_tokens", 0)
    cost = (actual_input * 0.00015 / 1000) + (actual_output * 0.00060 / 1000)

    trace = {
        "step": "Generation",
        "model": "gpt-4o-mini",
        "temperature": 0,
        "prompt_breakdown": {
            "system_prompt_tokens": count_tokens(system_prompt),
            "context_tokens": count_tokens(context),
            "question_tokens": count_tokens(question),
        },
        "token_usage": {"input": actual_input, "output": actual_output, "total": actual_input + actual_output},
        "cost_usd": round(cost, 6),
        "reason": "Used full Parent context for the answer while maintaining Child-level search scores."
    }

    return response.content, trace

# ============================================================
# PIPELINES
# ============================================================

def process_document(pdf_path):
    pages = load_pdf(pdf_path)
    chunks = chunk_documents(pages)
    vector_store = build_vector_store(chunks)
    return vector_store, len(chunks)

def answer_question(question, vector_store, k=5):
    traces = []
    rewritten_query, trace1 = rewrite_query_for_retrieval(question)
    traces.append(trace1)
    
    retrieved, trace2 = retrieve_relevant_chunks(rewritten_query, vector_store, k=k)
    traces.append(trace2)
    
    answer, trace3 = generate_answer(question, retrieved)
    traces.append(trace3)

    return answer, retrieved, traces