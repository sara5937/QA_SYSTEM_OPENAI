import streamlit as st
import os
import asyncio
from QAWithPDF.embedding import download_embedding
from QAWithPDF.model_api import load_model
from QAWithPDF.retriever import redriever_model
from QAWithPDF.data_ingestion import Data_loading
from langchain_chroma import Chroma
from logger import logging

PERSIST_DIR = "vec_DB"
logging.info("Temp directory creating...")
def save_uploadedfile(uploadedfile):
    temp_dir = "tempDir"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    file_path = os.path.join(temp_dir, uploadedfile.name)
    with open(file_path, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return file_path

def load_vectordb(embedding_model):
    if os.path.exists(PERSIST_DIR):
        return Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)
    return None

def main():
    st.set_page_config(page_title="QA with Documents")
    
    # Initialize session state
    if "vectordb" not in st.session_state:
        Chat_model, Embedding_model = load_model()
        st.session_state.vectordb = load_vectordb(Embedding_model)
        st.session_state.Chat_model = Chat_model
        st.session_state.Embedding_model = Embedding_model

    doc = st.file_uploader("Upload your document", type=["pdf"])
    
    st.header("QA with Documents (Information Retrieval)")
    
    user_question = st.text_input("Ask your question")
    
    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            if st.session_state.vectordb:
                # Use existing vector database
                response = redriever_model(st.session_state.vectordb, st.session_state.Chat_model, user_question)
                st.write(response["result"])
            elif doc is not None:
                # Save the uploaded file
                file_path = save_uploadedfile(doc)
                
                try:
                    # Read the document using Data_loading
                    documents = asyncio.run(Data_loading(file_path))
                    
                    # Process the document with LangChain
                    vectordb = download_embedding(st.session_state.Embedding_model, documents)
                    st.session_state.vectordb = vectordb

                    response = redriever_model(vectordb, st.session_state.Chat_model, user_question)
                    st.write(response["result"])
                finally:
                    # Delete the uploaded file after processing
                    os.remove(file_path)
            else:
                st.warning("Please upload a document first.")

if __name__ == "__main__":
    main()
