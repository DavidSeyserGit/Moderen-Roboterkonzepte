import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import requests
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from tools import *

@st.cache_data
def get_openrouter_models():
    url = "https://openrouter.ai/api/v1/models"
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_KEY']}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        models = [model['id'] for model in data['data']]
        return models
    else:
        return ["openai/gpt-4o-mini", "anthropic/claude-3-haiku:beta", "deepseek/deepseek-chat-v3.1:free"]  # Fallback

# Get available models
available_models = get_openrouter_models()

def initialize_agent(model):
    llm = ChatOpenAI(
        model=model,
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.environ["OPENROUTER_KEY"],
        temperature=0,
    )

    tools = [
        Tool.from_function(
            func=get_status,
            name="get_status",
            description="Check the current system status."
        ),
        Tool.from_function(
            func=retrieve_places,
            name="retrieve_places",
            description="Retrieve matching places from the knowledge base based on a query."
        ),
        Tool.from_function(
            func=execute_goal,
            name="execute_goal",
            description="sending the robot to a named location."
        ),
    ]

    system_prompt = """You are a helpful robot assistant for modern robotics concepts.
You always have to check the system status, retrieve places from the knowledge base, and execute goals to move the robot to locations."""

    agent = create_react_agent(llm, tools)
    return agent, system_prompt

# Initialize agent
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Chat")

# Model picker
default_index = available_models.index("deepseek/deepseek-chat-v3.1:free") if "deepseek/deepseek-chat-v3.1:free" in available_models else 0
selected_model = st.selectbox("Choose a model:", available_models, index=default_index)

if "selected_model" not in st.session_state or st.session_state.selected_model != selected_model:
    st.session_state.selected_model = selected_model
    st.session_state.agent, st.session_state.system_prompt = initialize_agent(selected_model)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Type your message..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = st.session_state.agent.invoke({
                "messages": [
                    SystemMessage(content=st.session_state.system_prompt),
                    HumanMessage(content=prompt)
                ]
            })
            response = result["messages"][-1].content
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
