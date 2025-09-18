import os
import re
import sqlite3
import pandas as pd
import numpy as np
import stopwordsiso as stopwordsiso
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_tokenize
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    pipeline,
    MBartForConditionalGeneration,
    MBart50Tokenizer,
)
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup

# =================== PREPROCESSING ===================
def load_stopwords():
    return stopwordsiso.stopwords("mr")

def clean_text(text, stopwords):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r"[^\u0900-\u097F\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    if len(text.strip()) < 3:
        return None
    factory = IndicNormalizerFactory()
    normalizer = factory.get_normalizer("mr")
    text = normalizer.normalize(text)
    tokens = indic_tokenize.trivial_tokenize(text, lang="mr")
    tokens = [w for w in tokens if w not in stopwords and len(w.strip()) > 1]
    cleaned = " ".join(tokens)
    return cleaned if cleaned.strip() else None

def is_valid_embedding(emb, expected_dim):
    return (emb is not None and not np.isnan(emb).any() and np.any(emb != 0) and emb.shape[-1] == expected_dim)

# =================== DATASET LOADING ===================
def load_dataset(train_path, valid_path):
    train_df = pd.read_csv(train_path, encoding="utf-8")
    valid_df = pd.read_csv(valid_path, encoding="utf-8")
    return train_df, valid_df

# =================== DATABASE ===================
def create_db(db_name="news_articles.db", feature_dim=384):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS news_articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            headline TEXT,
            clean_text TEXT,
            label TEXT,
            embedding BLOB
        )
        """
    )
    conn.commit()
    conn.close()

def insert_into_db(df, embeddings, db_name="news_articles.db", expected_dim=384):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    for i, row in df.iterrows():
        emb = embeddings[i]
        if not is_valid_embedding(emb, expected_dim):
            continue
        emb_bytes = emb.tobytes()
        cursor.execute(
            """
            INSERT INTO news_articles (headline, clean_text, label, embedding)
            VALUES (?, ?, ?, ?)
            """,
            (row["headline"], row["clean_text"], row["label"], emb_bytes),
        )
    conn.commit()
    conn.close()

def load_all_embeddings(db_name="news_articles.db", feature_dim=384):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("SELECT id, headline, embedding FROM news_articles")
    rows = cursor.fetchall()
    ids, headlines, embeddings = [], [], []
    for r in rows:
        if r[2] is None:
            continue
        emb = np.frombuffer(r[2], dtype=np.float32)
        if emb.shape[0] == feature_dim and not np.isnan(emb).any():
            ids.append(r[0])
            headlines.append(r[1])
            embeddings.append(emb)
    conn.close()
    if len(embeddings) == 0:
        raise ValueError("No valid embeddings found in DB.")
    return ids, headlines, np.vstack(embeddings)

# =================== DATASET CLASS ===================
class MarathiNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item

# =================== MODEL TRAINING ===================
def get_class_weights(labels):
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float)

def train_model(train_dataset, valid_dataset, model, tokenizer, epochs=5, batch_size=16, lr=2e-5, patience=2):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_labels = [item['labels'].item() for item in train_dataset]
    class_weights = get_class_weights(train_labels).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    best_f1 = 0
    epochs_no_improve = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        model.eval()
        all_preds, all_labels = [], []
        val_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = loss_fn(logits, labels)
                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        avg_val_loss = val_loss / len(valid_loader)
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Macro F1: {val_f1:.4f}")
        print("Validation Classification Report:")
        print(classification_report(all_labels, all_preds, zero_division=0))
        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_no_improve = 0
            model.save_pretrained("marathi_news_model")
            tokenizer.save_pretrained("marathi_news_model")
            print("Best model saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

# =================== EMBEDDING AND SIMILARITY ===================
def get_embeddings(texts, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings.astype(np.float32)

def build_faiss_index(embeddings):
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

def suggest_similar_news(input_headline, embedder, index, db_headlines, stopwords, top_k=5):
    cleaned = clean_text(input_headline, stopwords)
    if not cleaned:
        return []
    input_emb = embedder.encode([cleaned], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(input_emb)
    D, I = index.search(input_emb, top_k)
    results = []
    for idx, dist in zip(I[0], D[0]):
        if idx >= 0 and idx < len(db_headlines):
            results.append((db_headlines[idx], float(dist)))
    return results

def rebuild_faiss_index(db_name="news_articles.db"):
    ids, headlines, embeddings = load_all_embeddings(db_name)
    index = build_faiss_index(embeddings)
    faiss.write_index(index, "faiss_index.bin")
    print("FAISS index rebuilt and saved.")
    return index, headlines

# =================== ARTICLE EXTRACTION ===================
def contains_marathi(text):
    return bool(re.search(r"[\u0900-\u097F]", text))

def extract_marathi_text(url):
    if url.startswith("[") and "](" in url and url.endswith(")"):
        url = url[url.index("(")+1:-1]
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, timeout=10, headers=headers)
        soup = BeautifulSoup(r.text, "html.parser")
        paragraphs = soup.find_all('p')
        marathi_texts = [p.get_text().strip() for p in paragraphs if contains_marathi(p.get_text())]
        return " ".join(marathi_texts)
    except Exception as e:
        print(f"Error extracting article text: {e}")
        return None

# =================== TRANSLATION MODELS ===================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MBart model for both Marathi to English and Hindi translation
translation_model_name = "facebook/mbart-large-50-many-to-many-mmt"
translation_tokenizer = MBart50Tokenizer.from_pretrained(translation_model_name)
translation_model = MBartForConditionalGeneration.from_pretrained(translation_model_name)
translation_model.to(device)

def chunk_text(text, max_words=400):
    words = text.split()
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i+max_words])

def translate_text(text, target_lang):
    translation_tokenizer.src_lang = "mr_IN"
    
    if target_lang.lower() == "hi":
        tgt_lang_code = "hi_IN"
    elif target_lang.lower() == "en":
        tgt_lang_code = "en_XX"
    else:
        raise ValueError("Unsupported target language; use 'hi' or 'en'.")
        
    translations = []
    for chunk in chunk_text(text, max_words=400):
        encoded = translation_tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        forced_bos_token_id = translation_tokenizer.lang_code_to_id.get(
            tgt_lang_code, translation_tokenizer.lang_code_to_id["en_XX"])
            
        generated_tokens = translation_model.generate(
            **encoded,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=256,
            num_beams=5,
            early_stopping=True,
        )
        decoded = translation_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        translations.append(decoded)
    return " ".join(translations)

# =================== SUMMARIZATION ===================
summarizer = pipeline(
    "summarization",
    model="csebuetnlp/mT5_multilingual_XLSum",
    tokenizer="csebuetnlp/mT5_multilingual_XLSum"
)

def clean_article_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def summarize_marathi_text(article_text):
    cleaned_text = clean_article_text(article_text)
    chunk_size = 1000
    chunks = [cleaned_text[i:i+chunk_size] for i in range(0, len(cleaned_text), chunk_size)]
    summary_parts = []
    for chunk in chunks:
        summary = summarizer(
            chunk,
            max_length=180,
            min_length=100,
            length_penalty=2.0,
            num_beams=4,
            truncation=True
        )
        summary_parts.append(summary[0]['summary_text'].replace('\n', ' ').strip())
    return " ".join(summary_parts)

# =================== CLASSIFICATION ===================
def predict_category(text, tokenizer, model, cat2idx, stopwords):
    cleaned = clean_text(text, stopwords)
    if not cleaned:
        return "Invalid input text"
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(0)
        pred_idx = torch.argmax(logits).item()
    idx2cat = {v: k for k, v in cat2idx.items()}
    return idx2cat.get(pred_idx, "Unknown")

# =================== INFERENCE LOOP ===================
def run_inference_loop():
    tokenizer = AutoTokenizer.from_pretrained("marathi_news_model")
    model = AutoModelForSequenceClassification.from_pretrained("marathi_news_model")
    embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    index = faiss.read_index("faiss_index.bin")
    stopwords = load_stopwords()
    _, headlines, _ = load_all_embeddings()
    cat2idx = {c: i for i, c in enumerate(['entertainment', 'sports', 'state'])}
    print("Options:\n1. Classify and Recommend\n2. Summarize Marathi Article from URL\n3. Translate Marathi Article from URL\nexit to quit")
    while True:
        choice = input("Enter choice (1/2/3/exit): ").strip()
        if choice.lower() == "exit":
            print("see you again.")
            break
        elif choice == "1":
            user_text = input("Enter Marathi news headline or text: ").strip()
            cat = predict_category(user_text, tokenizer, model, cat2idx, stopwords)
            print(f"\nPredicted Category: {cat}")
            recommendations = suggest_similar_news(user_text, embedder, index, headlines, stopwords)
            print("Similar News Articles:")
            for title, score in recommendations:
                print(f"- {title} (Similarity: {score:.3f})")
        elif choice == "2":
            url = input("Enter the Marathi news article URL: ").strip()
            article_text = extract_marathi_text(url)
            if not article_text or len(article_text) < 50:
                print("Failed to extract or insufficient Marathi text from the given URL.")
                continue
            print("Generating summary ...")
            summary = summarize_marathi_text(article_text)
            print(f"\nSummary:\n{summary}")
        elif choice == "3":
            url = input("Enter the Marathi news article URL: ").strip()
            article_text = extract_marathi_text(url)
            if not article_text or len(article_text) < 50:
                print("Failed to extract or insufficient Marathi text from the given URL.")
                continue
            tgt_lang = input("Enter target language for translation (hi/en) [default hi]: ").strip().lower()
            if tgt_lang not in ['hi', 'en']:
                tgt_lang = 'hi'
            print(f"Translating to {tgt_lang} ...")
            translation = translate_text(article_text, tgt_lang)
            print(f"\nTranslation:\n{translation}")
        else:
            print("Invalid choice, please enter 1, 2, 3, or exit.")

if __name__ == "__main__":
    # Uncomment to train your model if needed
    # main()
    run_inference_loop()
    

