import streamlit as st
from langchain_openai import ChatOpenAI

#from dotenv import load_dotenv
#import os

from WikiChatbot.wikiChatbot import wiki_answer


#load_dotenv()
#api_key = os.getenv("OPENAI_TOKEN")

st.title("LLM Apps")
sidebar = st.sidebar
openai_api_key = sidebar.text_input("Enter an OpenAI API key", type="password")

# Check the OPENAI API key provided
if not(openai_api_key and openai_api_key.startswith("sk-")):
    st.warning("Please enter a valid OpenAI API key!", icon="⚠")
else:
    # Create an openai-based llm using langchain
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role":"ai", "content":"""I'm a AI bot dedicated to scroll Wikipedia content to answer your question.
                                      You should my answer with caution as an LLM-based chatbot, sometimes I can hallucination."""}]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input
    if query := st.chat_input("How can I assist you?"):
        # Add user message to chat history
        st.session_state.messages.append(
            {"role":"human", "content": query}
        )
        # Display user input
        with st.chat_message("human"):
            st.markdown(query)

        # Display assistant/ai response in chat message container
        with st.chat_message("ai"):
            response = wiki_answer(query, llm)
            st.markdown(response)
        
        # Add AI response in session state messages for history
        st.session_state.messages.append({
            "role": "ai",
            "content": response
        })
   

     

if __name__ == "__main__":
    pass
