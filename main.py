import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
import google.generativeai as genai



genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=os.getenv("GOOGLE_API_KEY"),
                                              model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_document_data(docs):
    data = ""
    for pdf in docs:
        doc = PdfReader(pdf)
        for page in doc.pages:
            data += page.extract_text()
    return data


def main():
    st.header("Chat with Documents")

    docs = st.file_uploader("Upload Documents (PDF, PPT, Word)", accept_multiple_files=True)

    if st.button("Upload"):
        data = get_document_data(docs)
        chunks = get_chunks(data)
        get_vector_store(chunks)
        st.toast("Documents Uploaded Successfully")
        st.switch_page("pages/chat.py")


if __name__ == "__main__":
    main()
