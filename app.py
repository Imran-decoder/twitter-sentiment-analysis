import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import re
import nltk
import html
from nltk.corpus import stopwords
from keras.preprocessing.sequence import pad_sequences
from typing import List, Tuple

# Local import
from twittes_collector import scrape_user_page

# Ensure NLTK data
nltk.download("stopwords")
nltk.download("punkt")

# -------------------
# CONSTANTS
# -------------------
RNN_MODEL_PATH = "sentiment_rnn_model.weights.h5"
TOKENIZER_PATH = "tokenizer.pkl"
ML_MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
MAX_SEQUENCE_LENGTH = 100

# -------------------
# Caching
# -------------------
@st.cache_resource
def load_rnn_model():
    return tf.keras.models.load_model(RNN_MODEL_PATH)

@st.cache_resource
def load_tokenizer():
    with open(TOKENIZER_PATH, "rb") as handle:
        return pickle.load(handle)

@st.cache_resource
def load_stopwords():
    return set(stopwords.words("english"))

@st.cache_resource
def load_ml_model_and_vectorizer():
    with open(ML_MODEL_PATH, "rb") as mf:
        model = pickle.load(mf)
    with open(VECTORIZER_PATH, "rb") as vf:
        vectorizer = pickle.load(vf)
    return model, vectorizer

# -------------------
# Preprocessing
# -------------------
def clean_text(text: str, stop_words: List[str]) -> str:
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower().split()
    text = [w for w in text if w not in stop_words]
    return " ".join(text)

def preprocess_rnn_input(text: str, tokenizer):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    cleaned = " ".join(tokens)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding="post")
    return padded

# -------------------
# Predictions
# -------------------
def predict_sentiment_rnn(text: str, model, tokenizer) -> Tuple[str, float]:
    padded = preprocess_rnn_input(text, tokenizer)
    pred = model.predict(padded)
    idx = np.argmax(pred, axis=1)[0]
    categories = ["Negative", "Neutral", "Positive"]
    return categories[idx], float(np.max(pred))

def predict_sentiment_ml(text: str, model, vectorizer, stop_words) -> Tuple[str, float]:
    cleaned = clean_text(text, stop_words)
    X = vectorizer.transform([cleaned])
    pred = model.predict(X)
    label = "Positive" if int(pred[0]) == 1 else "Negative"
    conf = 1.0
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        conf = float(max(proba))
    return label, conf

# -------------------
# UI Setup
# -------------------
st.set_page_config(page_title="Twitter Sentiment", page_icon="🕊️", layout="wide")
st.title("🕊️ Twitter Sentiment Analysis")
st.write("Analyze text sentiment using RNN (Deep Learning) or ML (Pickle) model. You can also fetch tweets via Selenium.")

with st.sidebar:
    st.header("Settings")
    model_choice = st.radio("Choose model:", ["RNN (TensorFlow)", "ML (Pickle)"])
    mode = st.radio("Mode:", ["Input text", "Fetch tweets (Selenium)"])
    tweet_count = st.slider("Tweets to fetch", 1, 20, 5)
    show_conf_threshold = st.slider("Highlight low confidence (<)", 50, 100, 60)

# Load resources
stop_words = load_stopwords()
if model_choice == "RNN (TensorFlow)":
    model = load_rnn_model()
    tokenizer = load_tokenizer()
else:
    model, vectorizer = load_ml_model_and_vectorizer()

# -------------------
# Main Interaction
# -------------------
if mode == "Input text":
    st.subheader("Analyze custom text")
    user_text = st.text_area("Enter text", height=150)
    if st.button("Analyze"):
        if not user_text.strip():
            st.warning("Please enter some text.")
        else:
            if model_choice == "RNN (TensorFlow)":
                label, conf = predict_sentiment_rnn(user_text, model, tokenizer)
            else:
                label, conf = predict_sentiment_ml(user_text, model, vectorizer, stop_words)

            badge_class = "badge-pos" if label == "Positive" else ("badge-neg" if label == "Negative" else "badge-neu")
            st.markdown(f"<div class='card'><div class='tweet-text'>{html.escape(user_text)}</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align:center'><div class='{badge_class}'>{label}</div></div>", unsafe_allow_html=True)
            st.write(f"Confidence: **{conf*100:.1f}%**")

else:
    st.subheader("Fetch Tweets by Username")
    username = st.text_input("Twitter username (without @)", "")
    if st.button("Fetch & Analyze"):
        if not username.strip():
            st.warning("Enter a username.")
        else:
            try:
                df_tweets = scrape_user_page(username, limit=tweet_count, max_scrolls=30, headless=True)
                texts = df_tweets["text"].astype(str).tolist() if "text" in df_tweets.columns else df_tweets.iloc[:,0].astype(str).tolist()
                results = []
                for text in texts:
                    if model_choice == "RNN (TensorFlow)":
                        label, conf = predict_sentiment_rnn(text, model, tokenizer)
                    else:
                        label, conf = predict_sentiment_ml(text, model, vectorizer, stop_words)
                    results.append({"text": text, "label": label, "conf": conf})

                pos_count = sum(1 for r in results if r["label"] == "Positive")
                neg_count = sum(1 for r in results if r["label"] == "Negative")
                st.metric("Total Tweets", len(results))
                st.metric("Positive", pos_count)
                st.metric("Negative", neg_count)
                st.bar_chart({"Positive": pos_count, "Negative": neg_count})

                for res in results:
                    badge_class = "badge-pos" if res["label"] == "Positive" else ("badge-neg" if res["label"] == "Negative" else "badge-neu")
                    st.markdown(
                         f"""
                        <div class="card">
                            <div class="tweet-row">
                                <span class="tweet-icon">💬</span>
                                <span class="tweet-text">{html.escape(res['text'])}</span>
                            </div>
                            <div class="result-row">
                                <span class="{badge_class}">{res['label']}</span>
                                <span class="conf-badge">Confidence: {res['conf']*100:.1f}%</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            except Exception as e:
                st.error(f"Error fetching tweets: {e}")

# -------------------
# Styling
# -------------------
st.markdown(
    """
    <style>
    .card {
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 12px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
    }
    .tweet-text { font-size: 15px; line-height: 1.4; }
    .badge-pos { background-color: #28a745; color: white; padding:6px 10px; border-radius: 8px; }
    .badge-neg { background-color: #dc3545; color: white; padding:6px 10px; border-radius: 8px; }
    .badge-neu { background-color: #ffc107; color: #333; padding:6px 10px; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)
