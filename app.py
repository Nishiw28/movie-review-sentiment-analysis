"""
Movie Review Sentiment Analysis
Movie Review -> CountVectorizer -> Numerical Features -> SVC -> Positive / Negative -> Gradio Web App

The model is trained in Movie_Review_Sentiment.ipynb and saved as sentiment_model.joblib.
This app loads the saved model, so it starts quickly.
"""

import gradio as gr
import joblib

# Load the saved CountVectorizer and SVC model
saved = joblib.load("sentiment_model.joblib")
vectorizer = saved["vectorizer"]
model = saved["model"]


def predict_sentiment(user_review):
    if user_review is None or not user_review.strip():
        return "Please enter a movie review."

    # Convert the review to numbers with the same CountVectorizer, then predict
    review_vector = vectorizer.transform([user_review])
    prediction = model.predict(review_vector)[0]
    return "Positive 😊" if prediction == "positive" else "Negative 😞"


demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=6, label="Movie Review", placeholder="Type or paste a movie review here..."),
    outputs=gr.Textbox(label="Predicted Sentiment"),
    title="Movie Review Sentiment Analysis",
    description="Enter a movie review to find out whether it is Positive or Negative. "
                "Built with CountVectorizer and SVC (Support Vector Classifier).",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
