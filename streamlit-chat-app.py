import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import os
import platform

# Get the Groq API key based on the operating system
if platform.system() == "Windows":
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY is not set in the environment variables. Please set it and restart the application.")
        st.stop()
else:
    # For non-Windows systems (including Streamlit Cloud), use st.secrets
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except KeyError:
        st.error("GROQ_API_KEY is not set in the Streamlit secrets. Please set it in the Streamlit Cloud dashboard.")
        st.stop()

# Create an instance of the Groq language model
llm = ChatGroq(
    api_key=groq_api_key,
    temperature=0.7,
    model_name="llama-3.1-70b-versatile",
)

# Define a template for the chatbot's responses
template = """
You are MediBot, a medical chatbot designed for preliminary symptom assessment. Your responses are brief and to the point. Follow these guidelines:
1. Ask concise questions about symptoms, duration, severity, and relevant history.
2. Gather essential information efficiently.
3. Once sufficient data is collected, provide a brief potential diagnosis.
4. Offer short, clear treatment suggestions or next steps.
5. Always include a concise disclaimer about seeking professional medical advice.
6. For severe symptoms, quickly advise emergency care.
Keep all responses brief and focused. Prioritize clarity and efficiency in your communication.
Current conversation:
{history}
Human: {input}
MediBot: 
"""

prompt = PromptTemplate(input_variables=["history", "input"], template=template)

# Initialize session state for memory
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

# Create the conversation chain
conversation = ConversationChain(
    llm=llm,
    memory=st.session_state.memory,
    prompt=prompt,
    verbose=False
)

# Streamlit app
st.title("MediBot - Medical Chat Assistant")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What are your symptoms?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get the response from the conversation chain
    response = conversation.predict(input=prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Debug: Display current memory contents
if st.checkbox("Show conversation memory"):
    st.write(st.session_state.memory.chat_memory.messages)

# Clear conversation button
if st.button("Clear Conversation"):
    st.session_state.messages = []
    st.session_state.memory.clear()
    st.experimental_rerun()