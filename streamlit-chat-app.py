import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain, LLMChain
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

# New function to generate potential diagnoses
def generate_potential_diagnoses(conversation_history):
    diagnosis_prompt = PromptTemplate(
        input_variables=["conversation"],
        template="""
Based on the following conversation between a patient and a medical chatbot, provide the top 3 most likely diagnoses. 
Give only the names of the conditions without any explanation or additional text.
Separate each diagnosis with a semicolon.

Conversation:
{conversation}

Top 3 potential diagnoses:"""
    )
    
    diagnosis_chain = LLMChain(
        llm=llm,
        prompt=diagnosis_prompt,
        verbose=False
    )
    
    diagnoses = diagnosis_chain.run(conversation=conversation_history)
    return diagnoses.strip().split(';')

# Initialize session state for memory and diagnoses
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
if 'diagnoses' not in st.session_state:
    st.session_state.diagnoses = ["No diagnosis yet", "No diagnosis yet", "No diagnosis yet"]

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

# Create a two-column layout
col1, col2 = st.columns([2, 1])

with col1:
    # Create a container for the chat messages
    chat_container = st.container()

    # Display chat messages from history on app rerun
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Create a form for user input
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Type your message here:", key="user_input")
        submit_button = st.form_submit_button("Send")

    if submit_button and user_input:
        # Display user message in chat message container
        with chat_container:
            st.chat_message("user").markdown(user_input)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Get the response from the conversation chain
        response = conversation.predict(input=user_input)
        
        # Display assistant response in chat message container
        with chat_container:
            with st.chat_message("assistant"):
                st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Update potential diagnoses
        conversation_history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
        st.session_state.diagnoses = generate_potential_diagnoses(conversation_history)

        # Rerun the app to update the chat display
        st.experimental_rerun()

with col2:
    st.subheader("Potential Diagnoses")
    for i, diagnosis in enumerate(st.session_state.diagnoses, 1):
        st.write(f"{i}. {diagnosis}")

# Debug: Display current memory contents
if st.checkbox("Show conversation memory"):
    st.write(st.session_state.memory.chat_memory.messages)

# Clear conversation button
if st.button("Clear Conversation"):
    st.session_state.messages = []
    st.session_state.memory.clear()
    st.session_state.diagnoses = ["No diagnosis yet", "No diagnosis yet", "No diagnosis yet"]
    st.experimental_rerun()

# Add centered credit with hyperlink at the bottom of the app
st.markdown(
    """
    <div style="position: fixed; left: 0; bottom: 10px; width: 100%; text-align: center;">
        <p>Developed by <a href="https://www.linkedin.com/in/mohamedfadlalla-ai/" target="_blank">Mohamed Fadlalla</a></p>
    </div>
    """,
    unsafe_allow_html=True
)