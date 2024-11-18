import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from PIL import Image
import fitz  # PyMuPDF

# Load the summarization model and tokenizer (T5 or BART)
model_name = "facebook/bart-large-cnn"  # Options: "t5-small", "t5-large", or "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page_num in range(pdf.page_count):
            page = pdf[page_num]
            text += page.get_text("text")  # Extract text only, ignoring images
    return text

# Function to generate a summary for the extracted text
def summarize_text(text, max_length=400, min_length=80):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(
        inputs.input_ids,
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Streamlit frontend
st.title("PDF Summarizer Using BART")
st.write("Upload a PDF document, and the application will extract and summarize its content.")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file...", type=["pdf"])

if uploaded_file is not None:
    # Save the uploaded file
    with open("uploaded_file.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Extract text
    st.write("Extracting text from the PDF...")
    pdf_text = extract_text_from_pdf("uploaded_file.pdf")
    st.write("Extracted Text Preview:")
    st.text(pdf_text[:1000] + "...")  # Show the first 1000 characters as a preview

    # Generate summary
    if st.button("Generate Summary"):
        st.write("Generating summary, please wait...")
        summary = summarize_text(pdf_text)
        st.success("Summary Generated!")
        st.write("Summary:")
        st.text_area("Summary", summary, height=300)

# Sidebar with instructions
st.sidebar.title("Instructions")
st.sidebar.info(
    """
    1. Upload a PDF file using the uploader above.
    2. Click "Generate Summary" to summarize the document.
    3. View and copy the summary from the output field.
    """
)
