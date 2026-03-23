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

# ---AGENT INITIALIZATION WITH SYSTEM MESSAGE (TEMPLATE)---
def init_agent(llm_config,smsg) :
    # Function to initialize the agent with a system message
    assistant = AssistantAgent(
        name="templated_agent",
        system_message=smsg,
        llm_config=llm_config
    )
    return assistant

def prompt(assistant,prompt) :
    # Function to prompt the agent
    reply = assistant.generate_reply(
        messages=[{"content": prompt, "role": "user"}]
    )
    print(reply)

# ---TEMPLATE DEFINITION---
# Define the template using f-strings or external config files to keep instructions modular
AGENT_ROLE_TEMPLATE = """
You are a specialized {role} focusing on {topic}.
Your goal is to provide a {tone} tutorial to your student.
Please ensure any code you write follows {style_guide} standards.
"""
# ---FILLING THE TEMPLATE---
def populate(template) :
    # Function to fill the template with specific values
    custom_system_message = template.format(
        role="Senior Developer and Teacher",
        topic="Making Documentation Accessible to Beginners",
        tone="concise and easy to understand",
        style_guide="PEP8"
    )
    return custom_system_message

# ---PROMPT USING THE TEMPLATE---
def use(template) :
    # Function to initialize the agent with the populated template and prompt it
    p = str(input("Write a short prompt here : \n"))
    c = llm_config()
    a = init_agent(c,populate(template))
    prompt(a,p)