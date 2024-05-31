from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader

from logger import logging
from exception import customexception
import sys
import os



def Data_loading(filename):
    #loading the PDF files
    try:
        logging.info("data loading started...")

       
        
    
        loader = PyPDFLoader(file_path=filename)
        documents=loader.aload()
        print("loaded documents")
        logging.info("data loading completed...")
        return documents
    except Exception as e:
        logging.info("exception in loading data...")
        raise customexception(e,sys)
    




