import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Import for streamlit purpose: the import path starts at the location of the streamlit entrypoint
#from DocumentKnowledgeMining.utils.retrievers import save_in_chroma, load_pdf, split_documents
from utils.retrievers import save_in_chroma, load_pdf, split_documents

def knownledge_answer(question:str, retriever, llm):
    prompt_template = """Your are an AI assistant, developed to answer user questions based on the most relevant information retrieved from
    a knowledge vector store. Those information are given to you through context section below.
    Don't be afraid to be sorry if you cannot answer the user question despite the provided context. Just kindly let them known you're sorry
    to not be able to answer their questions if they are out of the scope.
    
    User question: {question}

    Context: {context}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()} |
        prompt |
        llm |
        StrOutputParser()
    )

    response = chain.invoke(question)
    return response


if __name__ == "__main__":
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_TOKEN")
    filepath = "data/Fundamentals of expert systems.pdf"
    docs = load_pdf(filepath)
    docs_chunks = split_documents(docs, 'token')
    retriever = save_in_chroma(docs, openai_api_key)
    llm = llm = ChatOpenAI(api_key=openai_api_key)
    while True:
        question = input("How can I assist you ? \n> ")
        response = knownledge_answer(question, retriever, llm)
        print(response)