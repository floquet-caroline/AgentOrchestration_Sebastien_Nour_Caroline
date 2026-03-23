from autogen_agentchat.agents import AssistantAgent
from config import get_model
from tools import find_urls, extract_content


def get_research_team(db):
    # Modèle léger pour Scout et Reader (tâches simples)
    model_fast = get_model("llama-3.1-8b-instant", max_tokens=1000)
    # Modèle puissant pour le Writer (rédaction longue)
    model_stable = get_model("llama-3.3-70b-versatile", max_tokens=2000)

    scout = AssistantAgent(
        name="Scout",
        model_client=model_fast,
        tools=[find_urls],
        system_message="""Tu es un chercheur d'informations web.
        MISSION : Utilise 'find_urls' pour trouver 2 ou 3 sources textuelles (articles, documentation).
        RESTRICTIONS :
        - Ignore YouTube et les réseaux sociaux.
        - Retourne uniquement la liste des URLs trouvées, sans commentaire superflu."""
    )

    reader = AssistantAgent(
        name="Reader",
        model_client=model_fast,
        tools=[extract_content],
        system_message="""Tu es un analyste technique.
        MISSION : Pour chaque URL reçue, utilise 'extract_content' pour en lire le contenu.
        Produis ensuite une synthèse technique structurée des points clés.
        CONTRAINTES : 
        - Sois factuel et précis.
        - Limite ta synthèse à 800 mots maximum pour rester concis."""
    )

    writer = AssistantAgent(
        name="Writer",
        model_client=model_stable,
        tools=[db.index_segments],
        system_message="""Tu es un Rédacteur Technique Senior.
        MISSION :
        1. Appelle 'index_segments' avec le contenu du tutoriel pour l'indexer en base.
        2. Rédige un tutoriel PROFESSIONNEL en Markdown.

        STRUCTURE OBLIGATOIRE :
        # [Titre du sujet]
        ## Introduction
        ## Architecture et Concepts clés
        ## Guide d'implémentation (avec exemples de code Python)
        ## Conclusion

        STYLE : Sois direct et technique. Pas de formules de politesse ni de bavardage inutile."""
    )

    return scout, reader, writer  # On retourne un tuple pour un accès nommé dans main.py


def get_professor(db):
    return AssistantAgent(
        name="Professor",
        model_client=get_model("llama-3.3-70b-versatile", max_tokens=1500),
        tools=[db.query_kb],
        system_message="""Tu es un professeur pédagogue.
        Pour répondre à l'étudiant, interroge TOUJOURS la base de connaissances via 'query_kb'.
        Utilise les informations récupérées pour expliquer les concepts clairement et simplement."""
    )