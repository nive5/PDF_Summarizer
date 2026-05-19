import os
from PyPDF2 import PdfReader
from transformers import pipeline

# PDF folder
folder_path = "pdfs"

# Load GPT-2 model
generator = pipeline(
    "text-generation",
    model="gpt2"
)

# Get all PDF files
pdf_files = [
    file for file in os.listdir(folder_path)
    if file.endswith(".pdf")
]

# Process PDFs
for pdf_file in pdf_files:

    pdf_path = os.path.join(folder_path, pdf_file)

    print(f"\nReading PDF: {pdf_file}")

    # Read PDF
    reader = PdfReader(pdf_path)

    text = ""

    # Skip first 5 pages and take useful pages
    for page in reader.pages[5:12]:

        page_text = page.extract_text()

        if page_text:
            text += page_text

    # Reduce text size
    text = text[:1200]

    # Create prompt
    prompt = (
        "Summarize the following book content in simple words:\n\n"
        + text
    )

    # Generate summary
    result = generator(
        prompt,
        max_new_tokens=80,
        num_return_sequences=1
    )

    # Extract summary
    generated_text = result[0]['generated_text']

    # Remove prompt from output
    summary = generated_text.replace(prompt, "")

    print("\nSummary:")
    print(summary.strip())

    print("\n" + "=" * 60)