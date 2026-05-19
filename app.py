import streamlit as st
import re
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------- MODEL ----------------
model_name = "facebook/bart-large-cnn"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# ---------------- CLEAN TEXT ----------------
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ---------------- SUMMARIZATION ----------------
def summarize(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=120,
        min_length=40,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ---------------- UI ----------------
st.set_page_config(page_title="PDF Summarizer AI", layout="centered")

st.title("📄 AI PDF Summarizer")
st.write("Upload a PDF and get a quick AI-generated summary")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

# ---------------- PROCESS ----------------
if uploaded_file is not None:

    reader = PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    text = clean_text(text)

    # ⚡ IMPORTANT: LIMIT TEXT (prevents freezing)
    text = text[:3000]

    if len(text) < 50:
        st.error("❌ No readable text found in PDF")
    else:

        st.info("Generating summary... please wait ⏳")

        try:
            summary = summarize(text)

            st.success("✅ Summary Generated!")

            st.subheader("📌 Summary")
            st.write(summary)

            st.download_button(
                "📥 Download Summary",
                summary,
                file_name="summary.txt"
            )

        except Exception as e:
            st.error(f"Error: {e}")
