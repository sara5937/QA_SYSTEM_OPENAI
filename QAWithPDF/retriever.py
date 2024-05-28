
from langchain.chains import RetrievalQA
from logger import logging
from exception import customexception
import sys

def redriever_model(vectordb,Chat_model,user_question):
    try:
  
        retriever = vectordb.as_retriever(search_kwargs={"k": 2})


        qa_chain = RetrievalQA.from_chain_type(llm=Chat_model,
                                    chain_type="stuff",
                                    retriever=retriever,
                                    return_source_documents=True)
        res = qa_chain.invoke(user_question)
        
        return res
    except Exception as e:
        raise customexception(e,sys)
    


