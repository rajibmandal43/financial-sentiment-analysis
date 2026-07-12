from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Load FinBERT model
import torch
classifier = pipeline(
    task="sentiment-analysis",
    model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert",
    framework="pt",
    device=-1
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["text"]

    result = classifier(text)[0]

    label = result["label"].lower()
    score = round(result["score"] * 100, 2)

    return render_template(
        "index.html",
        prediction=label,
        confidence=score,
        text=text
    )

if __name__ == "__main__":
    app.run(debug=True)
