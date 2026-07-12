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


from flask import Flask, render_template, request
import requests
import os

app = Flask(__name__)

# Hugging Face Inference API
API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"

HF_TOKEN = os.getenv("HF_TOKEN")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


def predict_sentiment(text):
    payload = {
        "inputs": text
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        print(response.text)
        return None

    result = response.json()

    # Handle different response formats
    if isinstance(result, list):

        if len(result) > 0 and isinstance(result[0], list):
            result = result[0]

        best = max(result, key=lambda x: x["score"])

        return {
            "label": best["label"].lower(),
            "score": round(best["score"] * 100, 2)
        }

    return None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    text = request.form["text"]

    prediction = predict_sentiment(text)

    if prediction is None:
        return render_template(
            "index.html",
            prediction="Error",
            confidence=0,
            text=text
        )

    return render_template(
        "index.html",
        prediction=prediction["label"],
        confidence=prediction["score"],
        text=text
    )


if __name__ == "__main__":
    app.run(debug=True)