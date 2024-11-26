import requests
from langchain_community.retrievers import WikipediaRetriever
from typing import List, Optional

def get_wikipedia_page(title: str) -> str:
    """
    Retrieve the full text content of a Wikipedia page.

    :param title: str - Title of the Wikipedia page.
    :return: str - Full text content of the page as raw string.
    """
    # Wikipedia API endpoint
    URL = "https://fr.wikipedia.org/w/api.php"

    # Parameters for the API request
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
    }

    # Custom User-Agent header to comply with Wikipedia's best practices
    headers = {"User-Agent": "RAG_project/0.0.1 (contact@datascientist.fr)"}

    response = requests.get(URL, params=params, headers=headers)
    data = response.json()

    # Extracting page content
    page = next(iter(data["query"]["pages"].values()))
    return page["extract"] if "extract" in page else None

def format_docs(docs:List[str]):
    doc_string = "\n\n".join([
        doc.page_content for doc in docs
    ])

def wikipedia_retriever(question: str):
    wiki_retriever = WikipediaRetriever()
    docs = wiki_retriever.invoke(question)
    context = format_docs(docs)
    return context


