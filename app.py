import streamlit as st
from transformers import pipeline
import PyPDF2

# Initialize the Granite 3.3 model
st.title("PDF Q&A with Granite 3.3")

@st.cache_resource  # caches the model so it doesn't reload every run
def load_model():
    return pipeline("text-generation", model="ibm-granite/granite-3.3-2b-instruct")

pipe = load_model()

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    
    st.success("PDF loaded successfully!")

    # Ask question
    question = st.text_input("Ask a question about the PDF:")

    if question:
        # Prepare prompt
        prompt = f"Document:\n{text}\n\nQuestion: {question}\nAnswer:"

        # Generate answer
        result = pipe(prompt, max_length=500, do_sample=True)[0]['generated_text']
        answer = result.replace(prompt, "").strip()

        st.subheader("Answer:")
        st.write(answer)
