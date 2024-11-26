import requests
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from lang_chain_openai import ChatOpenAPI
from dotenv import load_dotenv
import os

from utils.data_access import wikipedia_retriever

load_dotenv()


# PAGES = [
#     "Intelligence_artificielle_générative",
#     "Transformeur_génératif_préentraîné",
#     "Google_Gemini",
#     "Grand_modèle_de_langage",
#     "ChatGPT",
#     "LLaMA",
#     "Réseaux_antagonistes_génératifs",
#     "Apprentissage_auto-supervisé",
#     "Apprentissage_par_renforcement",
#     "DALL-E",
#     "Midjourney",
#     "Stable_Diffusion"
# ]
api_key = os.get("OPENAI_TOKEN")
llm = ChatOpenAPI(api_key=api_key)

def wiki_answer(query, llm):
    prompt = ChatPromptTemplate.from_template(
        """
            You're an AI assantant specialized on Wikipedia content that answer
            in a professional and user-friendly manner questions. 
            Use the provided context to answer the user's question.
            In case, you cannot get relevant information from Wikipedia, kindly response to the user
            that you're sorry but no wikipedia content is related to their question.

            Context: {context}

            Question: {question}
        """
    )
    wiki_chain = (
        {"context": RunnablePassthrough() | wikipedia_retriever,
         "question": RunnablePassthrough()} |
         prompt |
         llm |
         StrOutputParser()
    )
    response = wiki_chain.invoke(query)
    return response


if __name__ == "__main__":
    print('Building Wikipedia chatbot...')

    try:
        while True:
            print('-' * 50)
            print('Ask a question :')
            question = input('> ')
            print()
            wiki_answer(question)
            print('\n')

    except KeyboardInterrupt:
        print("\nExiting...")
