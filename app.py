import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# Retrieve the API keys from Streamlit secrets
langchain_api_key = st.secrets["langchain_api_key"]
groq_api_key = st.secrets["groq_api_key"]

# Check if the API keys are retrieved correctly
if not langchain_api_key or not groq_api_key:
    raise ValueError("API keys are not set. Please check your Streamlit Cloud secrets.")

# Set the environment variables explicitly (if necessary for any downstream libraries)
import os
os.environ['langchain_api_key'] = langchain_api_key
os.environ['groq_api_key'] = groq_api_key

# Define the prompt template with memory
conversation_history = []

# Function to generate a response
def generate_response(question, engine, temperature, max_token):
    llm = ChatGroq(model=engine)
    output_parser = StrOutputParser()
    
    # Append the user's question to the conversation history
    conversation_history.append(("user", question))
    
    # Add past conversation to the prompt
    prompt = ChatPromptTemplate.from_messages([("system", "You are a helpful Assistant")] + conversation_history)
    
    # Create the chain to process the prompt and model
    chain = prompt | llm | output_parser
    answer = chain.invoke({"question": question})
    
    # Append the assistant's answer to the conversation history
    conversation_history.append(("assistant", answer))
    
    return answer

# Build Streamlit app interface
st.title("QnA Chatbot with Memory")

# Sidebar selections
engine = st.sidebar.selectbox("Select model", [
    "gemma2-9b-it", 
    "llama3-groq-70b-8192-tool-use-preview", 
    "llama-3.1-8b-instant", 
    "lama3-groq-8b-8192-tool-use-preview"
])
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_token = st.sidebar.slider("Max Tokens", min_value=100, max_value=500, value=250)

# Display the conversation history
st.write("### Conversation History")
for i, (role, message) in enumerate(conversation_history):
    if role == "user":
        st.write(f"**You:** {message}")
    else:
        st.write(f"**Assistant:** {message}")

# Create a form at the bottom of the app for user input
with st.form(key='input_form', clear_on_submit=True):
    st.write("Go ahead and ask your question")
    user_input = st.text_input("You:")
    submit_button = st.form_submit_button(label="Submit")

# Process the user's input when the button is clicked
if submit_button and user_input:
    response = generate_response(user_input, engine, temperature, max_token)
    st.write(f"**Assistant:** {response}")
else:
    if submit_button:
        st.write("Please provide a question.")
