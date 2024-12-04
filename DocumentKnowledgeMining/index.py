import streamlit as st
from langchain_openai import ChatOpenAI
import os


from utils.retrievers import (load_csv, load_markdown, load_pdf,
                              split_documents, save_in_chroma)
from documentKnowledgeMining import knownledge_answer

st.markdown("# Document Knowledge Mining AI Bot")
openai_api_key = st.text_input("Enter an OpenAI API key", type="password")
# Issue with using uploaded file from streamlit. Should be resolved later on.
# uploaded_file= st.file_uploader("Upload document(s) to store...",
#                             type=["pdf", "csv", "md"])
# if uploaded_file:
#     filename = uploaded_file.name
#     print(f"Filename: {filename}")
#     filepath = os.path.join("/tmp", filename)
#     # Create a temporary file to save the uploaded file and pass it forward to document loader
#     with open(filepath, "wb") as file:
#         file.write(uploaded_file.getvalue())
        
#     if uploaded_file.name.endswith(".pdf"):
#         docs = load_pdf(filepath)
#     elif uploaded_file.name.endswith(".csv"):
#         docs = load_csv(filepath)
#     elif uploaded_file.name.endswith(".md"):
#         docs = load_markdown(filepath)

internal_files = {
    "EU AI Act": "data/eu_ai_act.pdf",
    "Fundamentals of Expert Systems": "data/Fundamentals of expert systems.pdf",
    "EU Central Bank Report 2023": "data/eu_central_bank_report_2023.pdf",
    "Attention is all you need (LLM fundation paper)": "data/Attention is all you need.pdf"
}
file_options = internal_files.keys()
selected_file_option = st.selectbox(
    "Select a file for demo",
    options = file_options
)
if selected_file_option and openai_api_key:
    filepath = internal_files[selected_file_option]
    if filepath.endswith(".pdf"):
        docs = load_pdf(filepath)
    elif filepath.endswith(".csv"):
        docs = load_csv(filepath)
    elif filepath.endswith(".md"):
        docs = load_markdown(filepath)
    docs_chunks = split_documents(docs)
    retriever = save_in_chroma(docs_chunks, openai_api_key)
    llm = llm = ChatOpenAI(api_key=openai_api_key)
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role":"ai", "content":"""I'm a Document Knowledge Mining AI bot that answer questions 
                                         related to the document knowledge database I've access for, i.e. your uploaded documents."""}]
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if question := st.chat_input("Listing to your questions ..."):
        with st.chat_message("human"):
            st.markdown(question)
        st.session_state["messages"].append({"role":"human", "content": question})
        ai_response = knownledge_answer(question, retriever, llm)
        with st.chat_message("ai"):
            st.markdown(ai_response)
        st.session_state["messages"].append({"role":"ai", "content":ai_response})

