import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pdfplumber
from textwrap import wrap
import google.generativeai as genai
from dotenv import load_dotenv


load_dotenv()

model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text


def chunk_text(text, max_chunk_length=500):
    return wrap(text, max_chunk_length)

st.title("Upload Resume & Ask Questions")
uploaded_file = st.file_uploader("Upload your Resume PDF", type=["pdf"])
if uploaded_file is not None:
    resume_text = extract_text_from_pdf(uploaded_file)
    resume_chunks = chunk_text(resume_text)

    embeddings = model.encode(resume_chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))

    chunk_map = {i: chunk for i, chunk in enumerate(resume_chunks)}






    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    # Create the model
    model_gemini = genai.GenerativeModel("gemini-1.5-flash")




    st.title("Ask About Me - Resume Q&A")

    question = st.text_input("Ask a question about me:")


    if question:
        # Encode user question
        query_vec = model.encode([question])
        query_vec = np.array(query_vec).astype('float32')

        # Search top 2 closest chunks from resume
        top_k = 2
        D, I = index.search(query_vec, top_k)

        retrieved_chunks = [chunk_map[i] for i in I[0]]

        # Compose prompt
        prompt = f"""
        You are a helpful assistant reviewing a candidate's resume.
    
        Using the information below, answer the question thoughtfully:
        - If the info is in the resume, answer accordingly.
        - If it isn't, give a positive, human-like response based on the candidate's background.
        Answer the following question using the resume info below:
    
        Resume Info:
        {retrieved_chunks[0]}
        {retrieved_chunks[1]}
    
        Question: {question}
        Answer:
        """

        response = model_gemini.generate_content(prompt)
        answer = response.text.strip()

        st.markdown("### Answer:")
        st.write(answer)
    else:
        st.write("Please enter a question to get started.")
