import os
from dotenv import load_dotenv
import sys

from langchain_openai import ChatOpenAI
from langchain.embeddings import SentenceTransformerEmbeddings

from exception import customexception
from logger import logging

load_dotenv()

OPEN_AI_API_KEY= os.getenv('OPEN_AI_API_KEY')



def load_model():
    

    try:
        logging.info("Initialize Embedding model")
        llm_embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        logging.info("Embedding Model Loaded")

        logging.info("Loading Chat model")
        Chat_model = ChatOpenAI(model_name = "gpt-3.5-turbo",openai_api_key  = OPEN_AI_API_KEY)
        return Chat_model,llm_embedding
    except Exception as e:
        raise customexception(e,sys)
        