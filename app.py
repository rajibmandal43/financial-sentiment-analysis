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

# Hugging Face Model API
API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"

HF_TOKEN = os.getenv("HF_TOKEN")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}


def predict_sentiment(text):
    payload = {
        "inputs": text
    }

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=60
        )

        print("=" * 60)
        print("STATUS CODE:", response.status_code)
        print("RAW RESPONSE:")
        print(response.text)
        print("=" * 60)

        # If API returned an error
        if response.status_code != 200:
            return {
                "label": f"API Error ({response.status_code})",
                "score": 0
            }

        result = response.json()

        # Handle Hugging Face error response
        if isinstance(result, dict):

            if "error" in result:
                return {
                    "label": result["error"],
                    "score": 0
                }

            if "estimated_time" in result:
                return {
                    "label": "Model is loading. Please try again in a few seconds.",
                    "score": 0
                }

            return {
                "label": "Unexpected API Response",
                "score": 0
            }

        # Handle nested list response
        if isinstance(result, list):

            if len(result) > 0 and isinstance(result[0], list):
                result = result[0]

            best = max(result, key=lambda x: x["score"])

            return {
                "label": best["label"].lower(),
                "score": round(best["score"] * 100, 2)
            }

        return {
            "label": "Unknown Response",
            "score": 0
        }

    except Exception as e:
        print("EXCEPTION:")
        print(str(e))

        return {
            "label": str(e),
            "score": 0
        }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    text = request.form["text"]

    prediction = predict_sentiment(text)

    return render_template(
        "index.html",
        prediction=prediction["label"],
        confidence=prediction["score"],
        text=text
    )


if __name__ == "__main__":
    app.run(debug=True)