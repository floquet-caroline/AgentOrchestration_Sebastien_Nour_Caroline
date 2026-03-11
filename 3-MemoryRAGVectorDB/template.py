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

def init_agent(llm_config,smsg) :
    # 2. Initialize the agent
    assistant = AssistantAgent(
        name="templated_agent",
        system_message=smsg,
        llm_config=llm_config
    )
    return assistant

def prompt(assistant,prompt) :
    # 3. Manually get a reply
    reply = assistant.generate_reply(
        messages=[{"content": prompt, "role": "user"}]
    )
    print(reply)

# 1. Define the Template
# Using f-strings or external config files to keep instructions modular
AGENT_ROLE_TEMPLATE = """
You are a specialized {role} focusing on {topic}.
Your goal is to provide a {tone} tutorial to your student.
Please ensure any code you write follows {style_guide} standards.
"""

def populate(template) :
    # 2. Populate the Template
    custom_system_message = template.format(
        role="Senior Developer and Teacher",
        topic="Making Documentation Accessible to Beginners",
        tone="concise and easy to understand",
        style_guide="PEP8"
    )
    return custom_system_message

def use(template, my_prompt) :
    p = my_prompt
    c = llm_config()
    a = init_agent(c,populate(template))
    prompt(a,p)