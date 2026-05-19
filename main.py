import os
import re
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model (SAFE METHOD)
model_name = "facebook/bart-large-cnn"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

folder_path = "pdfs"


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def split_text(text, chunk_size=1000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


def summarize(text):
    inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")

    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=150,
        min_length=50,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


pdf_files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]


for pdf_file in pdf_files:

    pdf_path = os.path.join(folder_path, pdf_file)
    print(f"\nReading PDF: {pdf_file}")

    reader = PdfReader(pdf_path)

    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    if not text.strip():
        print("No readable text found.")
        continue

    text = clean_text(text)
    chunks = split_text(text)

    final_summary = ""

    for chunk in chunks:
        try:
            final_summary += summarize(chunk) + "\n"
        except:
            continue

    print("\nSummary:\n")
    print(final_summary)
    print("\n" + "=" * 60)