import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from twittes_collector import scrape_user_page
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple, List, Dict
import html


# --- Page config ---
st.set_page_config(page_title="Twitter Sentiment", page_icon="🕊️", layout="wide")

# --- Styling (small custom CSS) ---
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
    .small-muted { color: #6c757d; font-size:13px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Caching utilities ---
@st.cache_resource
def load_stopwords() -> List[str]:
    # Download if not present
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")
    return stopwords.words("english")

@st.cache_resource
def load_model_and_vectorizer(model_path: str = "models/model.pkl", vect_path: str = "models/vectorizer.pkl"):
    # Returns (model, vectorizer)
    with open(model_path, "rb") as mf:
        model = pickle.load(mf)
    with open(vect_path, "rb") as vf:
        vectorizer = pickle.load(vf)
    return model, vectorizer


# --- Text preprocessing & prediction ---
def preprocess_text(text: str, stop_words: List[str]) -> str:
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower().split()
    text = [w for w in text if w not in stop_words]
    return " ".join(text)

def sanitize_tweet_text(text: str) -> str:
    """
    Extra sanitization in case the scraper still returns stray HTML or UI fragments.
    - Collapse excessive whitespace
    - Strip angle-bracket fragments if any remain
    - Trim and return plain text
    """
    if text is None:
        return ""
    # remove any leftover HTML tags just in case
    text = re.sub(r"<[^>]+>", "", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def escape_and_preserve(text: str) -> str:
    # sanitize then escape and convert newlines to <br>
    safe = sanitize_tweet_text(text)
    return html.escape(safe).replace("\n", "<br>")


def predict_sentiment(text: str, model, vectorizer, stop_words: List[str]) -> Tuple[str, float]:
    """
    Returns (label_str, confidence_float)
    label_str = "Positive" or "Negative"
    confidence_float = probability of predicted class (0-1) if available, else 1.0
    """
    cleaned = preprocess_text(text, stop_words)
    X = vectorizer.transform([cleaned])
    pred = model.predict(X)
    label = "Positive" if int(pred[0]) == 1 else "Negative"
    conf = 1.0
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        conf = float(max(proba))
    return label, conf

# --- UI layout ---
st.title("🕊️ Twitter Sentiment Analysis")
st.write("Analyze raw text or fetch recent tweets for a username. The UI shows per-tweet sentiment and summary metrics.")

# Sidebar: mode + model path inputs
with st.sidebar:
    st.header("Settings")
    mode = st.selectbox("Mode", ["Input text", "Get tweets from user"])
    model_path = st.text_input("Model file path", value="model.pkl")
    vect_path = st.text_input("Vectorizer file path", value="vectorizer.pkl")
    tweet_count = st.slider("Tweets to fetch (when using scraper)", 1, 20, 5)
    show_conf_threshold = st.slider("Confidence threshold to highlight", 50, 100, 60)
    st.caption("Tip: Upload model/vectorizer to the app directory or set different paths.")

# Load resources with feedback
with st.spinner("Loading resources..."):
    stop_words = load_stopwords()
    try:
        model, vectorizer = load_model_and_vectorizer(model_path, vect_path)
        model_loaded = True
    except Exception as e:
        st.sidebar.error(f"Failed to load model/vectorizer: {e}")
        model = None
        vectorizer = None
        model_loaded = False


# --- Main interactive sections ---
if mode == "Input text":
    st.subheader("Analyze custom text")
    with st.form("text_form", clear_on_submit=False):
        user_text = st.text_area("Enter text to analyze", height=150, placeholder="Type or paste text here...")
        submitted = st.form_submit_button("Analyze")
    if submitted:
        if not user_text.strip():
            st.warning("Please enter some text.")
        elif not model_loaded:
            st.error("Model not loaded. Check model and vectorizer paths in sidebar.")
        else:
            with st.spinner("Predicting..."):
                clean_text = f'"{sanitize_tweet_text(user_text)}"'
                clean_text = clean_text + ' "end" '
                label, conf = predict_sentiment(clean_text, model, vectorizer, stop_words)
            col1, col2 = st.columns([3,1])
            with col1:
                st.markdown(f"<div class='card'><div class='tweet-text'>{user_text}</div></div>", unsafe_allow_html=True)
            with col2:
                badge_class = "badge-pos" if label == "Positive" else "badge-neg"
                st.markdown(f"<div style='text-align:center'><div class='{badge_class}'>{label}</div></div>", unsafe_allow_html=True)
                st.write(f"Confidence: **{conf*100:.1f}%**")
            # small explanation
            st.caption("Note: confidence shown only if model supports `predict_proba()`.")

else:
    st.subheader("Fetch Tweets by Username")
    with st.form("user_form"):
        username = st.text_input("Twitter username (without @)", placeholder="e.g. twitter")
        submit_user = st.form_submit_button("Fetch & Analyze")
    if submit_user:
        if not username.strip():
            st.warning("Enter a username.")
        elif not model_loaded:
            st.error("Model not loaded. Check model/vectorizer paths.")
        else:
            # Use twittes_collector.scrape_user_page to fetch tweets (Selenium)
            with st.spinner(f"Using Selenium to fetch up to {tweet_count} tweets from @{username} ... This may take a few seconds."):
                try:
                    # headless=True runs Chrome in background. Set headless=False for debugging (shows browser).
                    df_tweets = scrape_user_page(username, limit=tweet_count, max_scrolls=30, headless=True)
                except Exception as e:
                    st.error(f"Error fetching tweets with Selenium: {e}")
                    df_tweets = None

            # Validate returned data (expecting a DataFrame with 'text' column or similar)
            if df_tweets is None or (hasattr(df_tweets, "empty") and df_tweets.empty):
                st.info("No tweets found or an error occurred while scraping. If the profile requires login, try running with headless=False and log in manually for testing.")
            else:
                # Normalize possible column names to get text
                if "text" in df_tweets.columns:
                    texts = df_tweets["text"].astype(str).tolist()
                elif "content" in df_tweets.columns:
                    texts = df_tweets["content"].astype(str).tolist()
                else:
                    # fallback: take the first text-like column
                    texts = []
                    for col in df_tweets.columns:
                        if df_tweets[col].dtype == object:
                            texts = df_tweets[col].astype(str).tolist()
                            break

                # Analyze
                results = []
                for text in texts:
                    label, conf = predict_sentiment(text, model, vectorizer, stop_words)
                    results.append({"text": text, "label": label, "conf": conf})

                # Summary metrics
                pos_count = sum(1 for r in results if r["label"] == "Positive")
                neg_count = sum(1 for r in results if r["label"] == "Negative")
                st.write("---")
                mcol1, mcol2, mcol3 = st.columns([1,1,2])
                mcol1.metric("Total Tweets", len(results))
                mcol2.metric("Positive", pos_count)
                mcol3.metric("Negative", neg_count)

                # Small bar chart
                st.bar_chart({"Sentiment": {"Positive": pos_count, "Negative": neg_count}})

                # Display tweets in cards - two columns layout (same as your UI)
                st.write("### Tweets")


                def escape_and_preserve(text: str) -> str:
                    # Escape HTML characters and convert newlines to <br> so they show in the card
                    return html.escape(text).replace("\n", "<br>")


                for i, res in enumerate(results):
                    col_left, col_right = st.columns(2)
                    target_col = col_left if i % 2 == 0 else col_right
                    badge_class = "badge-pos" if res["label"] == "Positive" else "badge-neg"
                    low_conf = (res["conf"] * 100) < show_conf_threshold
                    extra_note = ""
                    if low_conf:
                        extra_note = f"<div class='small-muted'>Low confidence: {res['conf'] * 100:.1f}%</div>"

                    # Escape the text so any HTML fragments in tweet content are shown literally
                    safe_text_html = f"<div class='tweet-text'>{escape_and_preserve(res['text'])}</div>"

                    card_html = f"""
                    <div class='card'>
                      <div style='display:flex; justify-content:space-between; align-items:center;'>
                        <div style='flex:1; margin-right:10px;'>
                          {safe_text_html}
                          {extra_note}
                        </div>
                        <div style='min-width:110px; text-align:right;'>
                          <div class='{badge_class}'>{res['label']}</div>
                          <div class='small-muted'>{res['conf'] * 100:.1f}%</div>
                        </div>
                      </div>
                    </div>
                    """


                    target_col.markdown(card_html, unsafe_allow_html=True)

                st.success("Analysis complete.")


# Footer small note
st.markdown("<hr><div class='small-muted'>Built with Streamlit · Model loaded from disk · Use responsibly — respect Twitter scraping rules.</div>", unsafe_allow_html=True)
