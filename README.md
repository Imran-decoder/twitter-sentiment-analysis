# Twitter Sentiment Analysis

A project to perform sentiment classification on Twitter data using classical and deep learning techniques.

---

## 📂 Project Links & Resources

- **GitHub Repository**: https://github.com/Imran-decoder/twitter-sentiment-analysis.git   
- **Colab Notebook — RNN Approach**: *[Twitter sentiment analysis using RNN.ipynb](https://colab.research.google.com/drive/1fYidvABB2Rm4131DXGcj7vEpD3aIx9OY?usp=sharing)*  
- **Colab Notebook — Linear Regression Approach**: *[LinearRegression.ipynb*](https://colab.research.google.com/drive/1dtIOlNtFmOiPUoFJsNCkvA6QvNBwApz7?usp=sharing)  
- **Datasets**:  
  • Twitter Sentiment Analysis Dataset (for RNN notebook)  
  • Sentiment140 dataset (≈1.6 million tweets) (for linear regression notebook)

---

## 📁 Repo snapshot (key files)
- `app.py` — Flask app / inference server for the models. :contentReference[oaicite:3]{index=3}  
- `twittes_collector.py` — tweet collection script (Selenium-based / scraping). :contentReference[oaicite:4]{index=4}  
- `model.pkl` — saved classical ML model (scikit-learn). :contentReference[oaicite:5]{index=5}  
- `vectorizer.pkl` — pickled vectorizer (TF-IDF / CountVectorizer). :contentReference[oaicite:6]{index=6}  
- `tokenizer.pkl` — Keras tokenizer for RNN inference. :contentReference[oaicite:7]{index=7}  
- `sentiment_rnn_model.weights.h5` — RNN weights (Keras). :contentReference[oaicite:8]{index=8}  
- `sentiment_mapping.pkl` — label mapping (index → sentiment). :contentReference[oaicite:9]{index=9}  
- `requirements.txt` — Python dependencies. :contentReference[oaicite:10]{index=10}  
- `Dockerfile` — containerization for the app. :contentReference[oaicite:11]{index=11}

---

## 🎯 Project Goal
Provide a compact pipeline to:
1. Collect tweets (stream / scrape).  
2. Train / evaluate classical and RNN-based sentiment models.  
3. Serve predictions via a lightweight Flask app (and a Docker image) for quick demos / integration.

---

## 🧰 Technologies & Libraries (high-level)
- **Language:** Python 3.x. :contentReference[oaicite:12]{index=12}  
- **Data wrangling:** `pandas`, `numpy`  
- **NLP / preprocessing:** `nltk` / tokenization, stopwords, `re` for text cleaning  
- **Classical ML:** `scikit-learn` (vectorizers, linear models / logistic regression / LinearRegression experiments)  
- **Deep Learning (RNN):** `tensorflow` / `keras` — embedding + LSTM/GRU model saved as weights `.h5` + `tokenizer.pkl` for inference. :contentReference[oaicite:13]{index=13}  
- **Web Serving:** `Flask` (app.py serves inference UI / API)  
- **Data collection / scraping:** `selenium` + browser driver (script `twittes_collector.py`) — sample csv included. :contentReference[oaicite:14]{index=14}  
- **Deployment:** `Dockerfile` for containerized serving. :contentReference[oaicite:15]{index=15}

---
## Preview:
<img width="1470" height="802" alt="Screenshot 2025-10-10 at 5 12 09 PM" src="https://github.com/user-attachments/assets/81d278b1-e70a-46b4-a277-8c9179537c2b" />

## ⚡ Quickstart — Local (dev)

1. **Clone**
```bash
git clone https://github.com/Imran-decoder/twitter-sentiment-analysis.git
cd twitter-sentiment-analysis


