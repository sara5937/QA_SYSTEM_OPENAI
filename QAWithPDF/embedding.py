from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma


from QAWithPDF.data_ingestion import Data_loading
from QAWithPDF.model_api import load_model

import sys
from exception import customexception
from logger import logging

def download_embedding(model,document):
   
    try:
        logging.info("Creating Chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100)
        document_chunk = text_splitter.split_documents(document)
        persist_dir = "vec_DB"
        vectorDB = Chroma.from_documents(embedding = model ,persist_directory=persist_dir ,documents = document_chunk)
        return vectorDB
    except Exception as e:
        raise customexception(e,sys)