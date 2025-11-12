import os
import textwrap
import json
from typing import List, Tuple

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



def ensure_groq_client(api_key: str):

    api_key = api_key.strip() if api_key else ""
    if not api_key:
        raise ValueError("No Groq API key provided. Please set GROQ_API_KEY or enter it in the sidebar.")

    try:
        from groq import Groq

        client = Groq(api_key=api_key)

        def groq_chat(messages: List[dict], model: str = "llama-3.1-8b-instant"):


            resp = client.chat.completions.create(messages=messages, model=model)
            if hasattr(resp, "choices"):
                return resp.choices[0].message.content
            return resp["choices"][0]["message"]["content"]

        return groq_chat

    except Exception:
        try:
            from langchain_groq import ChatGroq
            from langchain.schema import HumanMessage, SystemMessage, AIMessage

            llm = ChatGroq(api_key=api_key)

            def groq_chat(messages: List[dict], model: str = None):
                lc_msgs = []
                for m in messages:
                    if m["role"] == "system":
                        lc_msgs.append(SystemMessage(content=m["content"]))
                    elif m["role"] == "user":
                        lc_msgs.append(HumanMessage(content=m["content"]))
                    else:
                        lc_msgs.append(AIMessage(content=m["content"]))
                out = llm.generate([lc_msgs])
                return out.generations[0][0].text

            return groq_chat

        except Exception:
            raise RuntimeError(
                "Could not initialize Groq client. Please install `groq` or `langchain-groq` and provide a valid GROQ_API_KEY."
            )


def validate_groq_key(api_key: str) -> bool:

    if not api_key:
        return False
    try:
        from groq import Groq

        client = Groq(api_key=api_key.strip())
        # Lightweight test to verify key validity
        _ = client.models.list()
        return True
    except Exception as e:
        if "401" in str(e) or "Invalid API Key" in str(e):
            return False
        return False


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i : i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


def build_vector_store(docs: List[str]):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=20000)
    X = vectorizer.fit_transform(docs)
    return vectorizer, X


def retrieve(query: str, docs: List[str], vectorizer: TfidfVectorizer, X, top_k: int = 3):
    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, X).flatten()
    idx_top = sims.argsort()[::-1][:top_k]
    return [(int(i), float(sims[i])) for i in idx_top]



st.set_page_config(page_title="HR Generative AI — Groq + Streamlit", layout="wide")

st.title("Emlployee query about HR policy")

with st.sidebar:
    st.header("Configuration")

    groq_api_key = st.text_input("API Key (or set as env var API_KEY)", type="password")
    if not groq_api_key:
        groq_api_key = os.getenv("GROQ_API_KEY", "")

    if groq_api_key:
        with st.spinner("Validating API key..."):
            is_valid = validate_groq_key(groq_api_key)
        if is_valid:
            st.success("API Key is valid!")
        else:
            st.error("Invalid Groq API Key. Please double-check your key.")
    else:
        st.info("Please enter your API key to enable the chatbot.")

    model_choice = st.selectbox("Groq Model", options=["llama-3.1-8b-instant","llama-3.1-70b-versatile"], index=0)
    top_k = st.slider("Retrieval: top K chunks", min_value=1, max_value=6, value=3)

st.markdown("---")

left, right = st.columns([1, 1])

with left:
    st.subheader("Business Requirements Document (BRD)")
    st.markdown(
        textwrap.dedent(
            """
        **Problem Statement**  
        Employees repeatedly ask HR-related questions (leave policy, reimbursement steps, payroll timelines).  
        Current HR channels are slow, inconsistent, and inefficient.

        **Goals & Objectives**
        - Provide instant, accurate answers to HR FAQs.  
        - Reduce HR team workload by 40% in 6 months.  
        - Maintain compliance and accuracy.

        **Stakeholders**
        - Employees (users)
        - HR team (content owners)
        - IT and Compliance teams

        **Scope & Assumptions**
        - Scope: FAQs, policy clarifications, reimbursement, leave management.
        - Assumes HR provides official documents for reference.
        """
        )
    )

st.subheader("User Stories")
st.markdown(
        """
        **1.  As a new employee**, I want to understand the company onboarding process so that I can complete all required formalities smoothly.  
        *Acceptance:* Returns the steps, required documents, and timelines for onboarding.

        **2.  As an employee**, I want to know my available leave balance so that I can plan my vacations effectively.  
        *Acceptance:* Provides an accurate count of remaining leaves and how to check them on the HR portal.

        **3.  As a manager**, I want to understand the performance review process so that I can prepare fair and timely evaluations.  
        *Acceptance:* Lists review timelines, rating scales, and submission steps.

        **4.  As an HR representative**, I want to update policy changes in the chatbot so that employees always receive the latest information.  
        *Acceptance:* Allows uploading a new HR policy document and reflects updates instantly.


        """
    )

with right:
    st.subheader("High-level Solution Architecture & Tools")
    st.markdown(
        textwrap.dedent(
            """
        **Architecture Overview**
        - **Frontend:** Streamlit web UI
        - **LLM Backend:** API key
        - **Retrieval:** TF-IDF (prototype) → Vector DB in production
        - **Storage:** CSV for logs (prototype), SQL/NoSQL in production
        - **Flow:** User query → Retrieve docs → Combine context → Send to LLM → Response

        **Technologies**
        - Streamlit, Python, API key
        - scikit-learn (TF-IDF)
        - Pandas for logging
        """
        )
    )

st.markdown("---")
st.header("Knowledge Base — Upload or Paste HR Docs")

uploaded_docs = []
text_input = st.text_area("Paste HR policy or document text:", height=150)
uploaded_files = st.file_uploader("Upload any files", accept_multiple_files=True)

if uploaded_files:
    for f in uploaded_files:
        content = f.read().decode("utf-8", errors="ignore")
        uploaded_docs.append((f.name, content))

all_texts = [t for (_, t) in uploaded_docs]
if text_input.strip():
    all_texts.insert(0, text_input.strip())

if not all_texts:
    st.info("No content uploaded. Please add HR policy text to continue.")
else:
    chunks = []
    for doc in all_texts:
        chunks.extend(chunk_text(doc, chunk_size=300, overlap=50))
    vectorizer, X = build_vector_store(chunks)
    st.success(f"Knowledge base ready with {len(chunks)} chunks.")

    query = st.text_input("Ask a question to the HR chatbot:")
    if st.button("Ask") and query.strip():
        if not groq_api_key:
            st.error("Please provide your API_KEY in the sidebar.")
        else:
            groq_chat = ensure_groq_client(groq_api_key)
            retrieved = retrieve(query, chunks, vectorizer, X, top_k=top_k)
            context = "\n\n---\n\n".join([chunks[i] for i, _ in retrieved])

            system_prompt = (
                "You are an HR assistant. "
                "You only answer HR-related questions about leave policy, reimbursement, payroll, benefits, and company HR processes. "
                "If the user asks about anything else unrelated to HR, politely respond: "
                "'I'm sorry, I can only help with HR-related questions.' "
                "Always keep your tone professional and concise."
            )

            user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            with st.spinner("Generating answer ..."):
                try:
                    answer = groq_chat(messages, model=model_choice)
                    st.markdown("Answer")
                    st.write(answer)

                    log = {
                        "question": query,
                        "answer": answer,
                        "chunks": [i for i, _ in retrieved],
                    }
                    pd.DataFrame([log]).to_csv("hr_chat_logs.csv", mode="a", header=False, index=False)
                    st.success("Interaction logged.")
                except Exception as e:
                    st.error(f"Groq error: {e}")

st.caption("Prototype assignment — includes BRD, user stories, architecture, and a functional Groq + Streamlit chatbot (HR-only mode).")
