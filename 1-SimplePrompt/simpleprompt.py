from autogen import AssistantAgent
import os
from dotenv import load_dotenv

def llm_config() :
    # Load variables from .env
    load_dotenv()

    # Access the key
    api_key = os.getenv("GROQ_API_KEY")
    # 1. Define the Groq configuration
    llm_config = {
        "config_list": [
            {
                "model": "llama-3.3-70b-versatile",  # Or your preferred Groq model
                "api_key": api_key,
                "base_url": "https://api.groq.com/openai/v1", # The crucial redirect
                "api_type": "openai",
            }
        ]
    }
    return llm_config

def init_agent(llm_config) :
    # 2. Initialize the agent
    assistant = AssistantAgent(
        name="solo_agent",
        llm_config=llm_config
    )
    return assistant

def prompt(assistant,prompt) :
    # 3. Manually get a reply
    reply = assistant.generate_reply(
        messages=[{"content": prompt, "role": "user"}]
    )
    print(reply)

p = str(input("Write a short prompt here : \n"))
c = llm_config()
a = init_agent(c)
prompt(a,p)