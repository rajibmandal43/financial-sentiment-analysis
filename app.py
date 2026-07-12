# from flask import Flask, render_template, request
# from transformers import pipeline

# app = Flask(__name__)

# # Load FinBERT model
# import torch
# classifier = pipeline(
#     task="sentiment-analysis",
#     model="ProsusAI/finbert",
#     tokenizer="ProsusAI/finbert",
#     framework="pt",
#     device=-1
# )

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     text = request.form["text"]

#     result = classifier(text)[0]

#     label = result["label"].lower()
#     score = round(result["score"] * 100, 2)

#     return render_template(
#         "index.html",
#         prediction=label,
#         confidence=score,
#         text=text
#     )

# if __name__ == "__main__":
#     app.run(debug=True)

import streamlit as st
from transformers import pipeline

st.set_page_config(
    page_title="Financial Sentiment Analysis",
    page_icon="💹",
)

st.title("💹 Financial Sentiment Analysis")
st.write("Powered by FinBERT")

@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="ProsusAI/finbert",
        tokenizer="ProsusAI/finbert"
    )

classifier = load_model()

text = st.text_area(
    "Enter financial news",
    height=200
)

if st.button("Analyze Sentiment"):

    if text.strip():

        result = classifier(text)[0]

        label = result["label"]
        score = result["score"] * 100

        st.success(f"Prediction: {label}")

        st.progress(int(score))

        st.write(f"Confidence: {score:.2f}%")