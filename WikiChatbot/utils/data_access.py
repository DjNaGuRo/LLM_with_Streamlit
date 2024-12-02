from langchain_community.retrievers import WikipediaRetriever
from langchain_community.utilities import wikipedia as wikipedia_utils
from typing import List

from WikiChatbot.fix.wikipedia import lazy_load

def format_docs(docs:List[str]):
    doc_string = "\n\n".join([
        doc.page_content for doc in docs
    ])
    return doc_string

def wikipedia_retriever(question: str):
    # Apply the modified lazy_load method to the WikipediaAPIWrapper class            
    wikipedia_utils.WikipediaAPIWrapper.lazy_load = lazy_load

    wiki_retriever = WikipediaRetriever()
    docs = wiki_retriever.invoke(question)
    context = format_docs(docs)
    return context


