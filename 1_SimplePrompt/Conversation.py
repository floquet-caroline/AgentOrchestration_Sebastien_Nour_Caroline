import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import MaxMessageTermination

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

    # 2. Définition du premier agent : Le Scientifique
    scientist = AssistantAgent(
        name="Scientifique",
        model_client=model_client,
        system_message="Tu es un scientifique sérieux. Donne des réponses précises mais courtes.",
    )

    # 3. Définition du deuxième agent : L'Enfant
    child = AssistantAgent(
        name="Enfant",
        model_client=model_client,
        system_message="Tu es un enfant de 5 ans très curieux. Après chaque réponse, demande 'Mais pourquoi ?' en rebondissant sur un détail. Essaye de déterminer qui des deux scientifiques te dit la vérité",
    )

    scientist2 = AssistantAgent(
        name="Scientifique2",
        model_client=model_client,
        system_message="Tu es une fraude qui se fait passer pour un scientifique, ton but est de contredire tout ce que le vrai scientifique dit et de convaincre l'enfant que toutes les bêtises que tu dis sont vraies"
    )

    termination = MaxMessageTermination(max_messages=15)

    team = RoundRobinGroupChat([scientist, scientist2 ,child], termination_condition=termination)

    await Console(team.run_stream(task="Explique-moi pourquoi la mer est salée."))


# Lancement de la boucle asynchrone
if __name__ == "__main__":
    asyncio.run(main())