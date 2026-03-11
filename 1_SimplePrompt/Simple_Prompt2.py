import asyncio
import os
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage

async def main():
    # 1. Configuration du client (compatible Groq via l'interface OpenAI)
    # On utilise OpenAIChatCompletionClient car Groq utilise le même format d'API
    model_client = OpenAIChatCompletionClient(
        model="llama-3.3-70b-versatile",
        api_key="", # Remplacez par votre clé
        base_url="https://api.groq.com/openai/v1", # Indispensable pour Groq
        model_info={
        "vision": False,       # Le modèle supporte-t-1 l'image ?
        "function_calling": True, 
        "json_output": True,
        "family": "unknown"    # On peut mettre unknown ici
    }
    )

    # 2. Définition de l'agent
    agent = AssistantAgent(
        name="SoloAgent",
        model_client=model_client,
        system_message="You are a brilliant scientist who explains things simply.",
    )

    response = await agent.on_messages(
    [TextMessage(content="Why is the sky blue?", source="user")],
    cancellation_token=None
    )
    print(response.chat_message.content)

# Lancement de la boucle asynchrone
if __name__ == "__main__":
    asyncio.run(main())