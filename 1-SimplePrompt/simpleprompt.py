from autogen import AssistantAgent

def llm_config(apikey) :
    # 1. Define the Groq configuration
    llm_config = {
        "config_list": [
            {
                "model": "llama-3.3-70b-versatile",  # Or your preferred Groq model
                "api_key": apikey,
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
k = str(input("Groq API key : \n"))
c = llm_config(k)
a = init_agent(c)
prompt(a,p)