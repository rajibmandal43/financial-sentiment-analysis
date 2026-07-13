import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Financial Sentiment Analysis", page_icon="💹", layout="wide")

st.markdown("""
<style>
.main {background:#f4f7fb;}
.header{
padding:20px;border-radius:15px;
background:linear-gradient(90deg,#0f4c81,#2563eb);
color:white;text-align:center;margin-bottom:20px;}
.block-container{padding-top:2rem;}
.stButton>button{width:100%;height:3rem;border-radius:10px;font-size:18px;}
.result{padding:15px;border-radius:10px;font-size:22px;font-weight:bold;text-align:center;}
.footer{text-align:center;color:gray;margin-top:30px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header"><h1>💹 Financial Sentiment Analysis</h1><p>Analyze financial news using FinBERT</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("📌 About")
    st.write("This application uses **ProsusAI/FinBERT** to classify financial news.")
    st.markdown("### Possible Predictions")
    st.write("🟢 Positive\n\n🔴 Negative\n\n⚪ Neutral")
    st.markdown("---")
    st.info("Model: ProsusAI/FinBERT")

@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis",
                    model="ProsusAI/finbert",
                    tokenizer="ProsusAI/finbert")

classifier = load_model()

examples = [
    "",
    "The company reported record profits this quarter.",
    "Revenue declined sharply due to weak consumer demand.",
    "The board announced no major changes for the coming year."
]

choice = st.selectbox("📄 Try an example", examples)

default = "" if choice == "" else choice

text = st.text_area("📰 Enter Financial News",
                    value=default,
                    height=220,
                    placeholder="Paste a financial headline or news article here...")

if st.button("🔍 Analyze Sentiment"):
    if text.strip():
        with st.spinner("Analyzing..."):
            result = classifier(text)[0]

        label = result["label"].capitalize()
        score = result["score"] * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction", label)
        c2.metric("Confidence", f"{score:.2f}%")
        c3.metric("Model", "FinBERT")

        if label.lower() == "positive":
            st.success("🟢 Positive Sentiment")
        elif label.lower() == "negative":
            st.error("🔴 Negative Sentiment")
        else:
            st.warning("⚪ Neutral Sentiment")

        st.subheader("Confidence")
        st.progress(min(100, int(score)))
        st.write(f"**Confidence Score:** {score:.2f}%")

        st.markdown("### 📋 Raw Model Output")
        st.json(result)
    else:
        st.warning("Please enter some financial news.")

st.markdown("---")
st.markdown('<div class="footer">Built with ❤️ using Streamlit + Hugging Face Transformers (FinBERT)</div>', unsafe_allow_html=True)
