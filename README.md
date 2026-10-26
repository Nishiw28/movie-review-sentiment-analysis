---
title: Movie Review Sentiment Analysis
emoji: 🎬
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
---

# Movie Review Sentiment Analysis

A machine learning web app that predicts whether a movie review is **Positive** or **Negative**. It uses **CountVectorizer** to turn review text into numbers, a **Support Vector Classifier (SVC)** to classify them, and **Gradio** to provide a simple web interface.

## 🚀 Live Demo

[Open Live Demo](https://huggingface.co/spaces/Nishi2809/movie-review-sentiment-analysis)

## How It Works

```
Movie Review
     ↓
CountVectorizer
     ↓
Numerical Feature Vectors
     ↓
SVC / Support Vector Classifier
     ↓
Positive / Negative
```

1. **Movie Review** – the text entered by the user (or the `review` column during training).
2. **CountVectorizer** – builds a vocabulary from all reviews and counts how many times each word appears in a review. It is applied only to the `review` text.
3. **Numerical Feature Vectors** – each review becomes a row of word counts that a model can work with.
4. **SVC** – learns from the vectors and the `sentiment` labels, then predicts the label for new reviews.
5. **Positive / Negative** – the prediction shown in the Gradio app.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- CountVectorizer
- SVC (Support Vector Classifier)
- Gradio

## Dataset

The project uses the **IMDB Dataset** (`IMDB Dataset.csv`), which contains **50,000 movie reviews** with two columns:

- `review` – the movie review text
- `sentiment` – the label (`positive` or `negative`)

## Model Performance

On a held-out 20% test split (`random_state=42`) the model reached **88.02% accuracy** (see the notebook).

## Features

- Enter a movie review in a textbox
- Submit it through the Gradio interface
- Get a Positive / Negative sentiment prediction
- Empty input is handled with a friendly message

## Project Structure

```
Movie-Review-Sentiment-Analysis/
├── .devcontainer/
├── Movie_Review_Sentiment.ipynb
├── IMDB Dataset.csv
├── sentiment_model.joblib
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

- `Movie_Review_Sentiment.ipynb` – trains the model and saves it
- `sentiment_model.joblib` – the saved CountVectorizer + SVC model
- `app.py` – the Gradio web app (loads the saved model)
- `.devcontainer/` – ready-to-use development environment (GitHub Codespaces / VS Code)

## How to Run Locally

```bash
pip install -r requirements.txt
python app.py
```

This loads the saved model and starts the Gradio app locally. Open the local URL printed in the terminal (usually http://127.0.0.1:7860) in your browser.

To retrain the model, run `Movie_Review_Sentiment.ipynb` (SVC training on the dataset is slow and can take a long time).
