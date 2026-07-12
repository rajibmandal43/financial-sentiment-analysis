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

API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"

HF_TOKEN = os.getenv("HF_TOKEN")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    text = request.form["text"]

    response = requests.post(
        API_URL,
        headers=headers,
        json={"inputs": text}
    )

    result = response.json()

    if isinstance(result, dict) and "error" in result:
        return render_template(
            "index.html",
            prediction="Error",
            confidence=0,
            text=text
        )

    scores = result[0]

    best = max(scores, key=lambda x: x["score"])

    label = best["label"].lower()
    confidence = round(best["score"] * 100, 2)

    return render_template(
        "index.html",
        prediction=label,
        confidence=confidence,
        text=text
    )


if __name__ == "__main__":
    app.run(debug=True)