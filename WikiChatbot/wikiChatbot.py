from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

from WikiChatbot.utils.data_access import wikipedia_retriever

def wiki_answer(query, llm):
    prompt_template = """
            You're an AI assistant specialized on Wikipedia content that answer
            in a professional and user-friendly manner questions. 
            Use the provided context to answer the user's question. When answering question, also provide at the bottom of your response 
            URLs of all relevant Wikipedia pages you use to construct your response.
            In case, you cannot get relevant information from Wikipedia, kindly response to the user
            that you're sorry but no wikipedia content is related to their question.

            Context: {context}

            User question: {question}
        """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    wiki_chain = (
        {"context": RunnablePassthrough() | wikipedia_retriever,
         "question": RunnablePassthrough()} |
         prompt |
         llm |
         StrOutputParser()
    )
    response = wiki_chain.invoke({"question":query})
    return response

# For command line testing purpose
if __name__ == "__main__":
    print('Building Wikipedia chatbot...')
    load_dotenv()

    api_key = os.getenv("OPENAI_TOKEN")
    llm = ChatOpenAI(api_key=api_key)
    try:
        while True:
            print('-' * 50)
            print('Ask a question :')
            question = input('> ')
            print("--- Answer ---")
            response = wiki_answer(question, llm)
            print(response)
            print('\n')

    except KeyboardInterrupt:
        print("\nExiting...")
