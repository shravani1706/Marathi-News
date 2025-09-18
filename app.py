import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import faiss

# Import your functions from main.py
from main import (
    predict_category,
    suggest_similar_news,
    summarize_marathi_text,
    translate_text,
    extract_marathi_text,
    load_all_embeddings,
    build_faiss_index,
    load_stopwords,
)

# -------------------------------
# Load models and resources
# -------------------------------
@st.cache_resource
def load_resources():
    tokenizer = AutoTokenizer.from_pretrained("marathi_news_model")
    model = AutoModelForSequenceClassification.from_pretrained("marathi_news_model")
    embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    stopwords = load_stopwords()

    # Load FAISS index
    ids, headlines, embeddings = load_all_embeddings()
    index = build_faiss_index(embeddings)

    cat2idx = {c: i for i, c in enumerate(['entertainment', 'sports', 'state'])}
    return tokenizer, model, embedder, index, headlines, stopwords, cat2idx


tokenizer, model, embedder, faiss_index, headlines, stopwords, cat2idx = load_resources()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📰 Marathi News Assistant")

task = st.sidebar.radio("Choose Task", ["Classify", "Recommend", "Summarize", "Translate"])

# --- Classification ---
if task == "Classify":
    st.header("🧾 Classify News")
    text = st.text_area("Enter Marathi news headline:")
    if st.button("Classify"):
        category = predict_category(text, tokenizer, model, cat2idx, stopwords)
        st.success(f"Predicted Category: **{category}**")

# --- Recommendation ---
elif task == "Recommend":
    st.header("🔍 Recommend Similar News")
    text = st.text_area("Enter Marathi news headline:")
    if st.button("Recommend"):
        results = suggest_similar_news(text, embedder, faiss_index, headlines, stopwords)
        if not results:
            st.warning("No similar news found.")
        else:
            st.subheader("Similar News Articles:")
            for title, score in results:
                st.write(f"- {title} (Similarity: {score:.3f})")

# --- Summarization ---
elif task == "Summarize":
    st.header("✂️ Summarize News")
    url = st.text_input("Enter a Marathi news article URL:")
    if st.button("Summarize"):
        article_text = extract_marathi_text(url)
        if not article_text:
            st.error("Could not extract Marathi text from the URL.")
        else:
            summary = summarize_marathi_text(article_text)
            st.subheader("Summary:")
            st.write(summary)

# --- Translation ---
elif task == "Translate":
    st.header("🌍 Translate News")
    url = st.text_input("Enter a Marathi news article URL:")
    target_lang = st.selectbox("Target Language", ["hi", "en"])
    if st.button("Translate"):
        article_text = extract_marathi_text(url)
        if not article_text:
            st.error("Could not extract Marathi text from the URL.")
        else:
            translated = translate_text(article_text, target_lang)
            st.subheader("Translation:")
            st.write(translated)
