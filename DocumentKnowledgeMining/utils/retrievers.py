from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from typing import Literal
import tiktoken

def load_csv(filepath):
    csv_loader = CSVLoader(filepath)
    documents = csv_loader.load()
    return documents

def load_pdf(filepath):
    pdf_loader = PyPDFLoader(filepath)
    documents = pdf_loader.load()
    return documents

def load_markdown(filepath):
    md_loader = UnstructuredMarkdownLoader(filepath)
    documents = md_loader.load()
    return documents

def split_documents(docs, type: Literal["recursive", "semantic", "token"] = "token", openai_api_key:str=None):
    if type == "recursive":
        splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    elif type == "semantic":
        embedding_model = OpenAIEmbeddings(
        api_key = openai_api_key,
        model = "text-embedding-3-small"
        )
        splitter = SemanticChunker(
            embeddings=embedding_model,
            breakpoint_threshold_type="gradient",
            breakpoint_threshold_amount=0.75
        )
    elif type == "token":
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        splitter = TokenTextSplitter(
            encoding_name=encoding.name,
            chunk_size=25,
            chunk_overlap=5
        )
    docs_chunks = splitter.split_documents(docs)
    return docs_chunks

def save_in_chroma(docs_chunks, openai_api_key):
    embedding_model = OpenAIEmbeddings(
        api_key = openai_api_key,
        model = "text-embedding-3-small"
    )

    vector_store = Chroma.from_documents(
        documents = docs_chunks,
        embedding = embedding_model
    )

    return vector_store.as_retriever(
        search_type="similarity", 
        search_kwargs = {"k": 5}
    )