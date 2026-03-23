from autogen_agentchat.agents import AssistantAgent
from config import get_model
from tools import find_urls, extract_content


def get_research_team(db):
    # Modèle léger pour Scout et Reader (tâches simples)
    model_fast = get_model("llama-3.1-8b-instant", max_tokens=1000)
    # Modèle puissant pour Writer et Critic (rédaction + évaluation)
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
        system_message="""Tu es un analyste technique senior.
        MISSION : Pour chaque URL reçue, utilise 'extract_content' pour en lire le contenu.
        Produis ensuite une synthèse structurée EXACTEMENT selon ce format :

        ## CONCEPTS CLÉS
        - Liste des concepts fondamentaux avec définitions concises

        ## ARCHITECTURE
        - Description des composants et de leurs interactions

        ## POINTS D'IMPLÉMENTATION
        - Éléments concrets et techniques utiles pour coder (patterns, étapes, pièges courants)

        ## SNIPPETS & EXEMPLES
        - Tout extrait de code ou pseudo-code trouvé dans les sources

        CONTRAINTES :
        - Sois factuel, précis, sans reformulation inutile.
        - 600 mots maximum.
        - Si le Writer te demande des précisions sur un point, réponds uniquement sur ce point."""
    )

    writer = AssistantAgent(
        name="Writer",
        model_client=model_stable,
        tools=[extract_content],  # index_segments géré par main.py après approbation du Critic
        system_message="""Tu es un Rédacteur Technique Senior.
        MISSION : Rédige un tutoriel PROFESSIONNEL en Markdown à partir de la synthèse fournie.
        Si la synthèse manque de détails sur un point précis, utilise 'extract_content'
        sur une URL pertinente pour compléter — ou indique [BESOIN_INFO: <point>]
        pour que le Reader te réponde.
        NE PAS appeler index_segments — l'indexation est gérée en dehors.

        STRUCTURE OBLIGATOIRE :
        # [Titre du sujet]
        ## Introduction
        > Pourquoi ce sujet est-il important ? Quel problème résout-il ?

        ## Architecture et Concepts clés
        > Schéma textuel des composants si pertinent (ex: `[Query] → [Retriever] → [LLM]`)

        ## Guide d'implémentation
        > OBLIGATOIRE : au moins 2 blocs de code Python commentés et fonctionnels.
        > Couvre : installation, code minimal, cas d'usage réel.

        ## Bonnes pratiques & Pièges courants
        > Au moins 3 points concrets issus de la synthèse.

        ## Conclusion
        > Résumé en 3 bullet points + ressources pour aller plus loin.

        STYLE : Direct, technique, sans bavardage. Pas de formules de politesse."""
    )

    critic = AssistantAgent(
        name="Critic",
        model_client=model_stable,
        system_message="""Tu es un Reviewer Technique exigeant.
        MISSION : Évaluer le tutoriel produit par le Writer et lui fournir un feedback actionnable.

        GRILLE D'ÉVALUATION (note chaque critère /5) :
        - CLARTÉ : Le texte est-il clair et bien structuré ?
        - CODE : Les exemples Python sont-ils présents, corrects et commentés ?
        - COMPLÉTUDE : Les sections obligatoires sont-elles toutes présentes ?
        - PROFONDEUR : Le contenu va-t-il au-delà du survol superficiel ?

        FORMAT DE RÉPONSE OBLIGATOIRE :
        ### SCORES
        | Critère | Score | Commentaire |
        |---|---|---|
        | Clarté | X/5 | ... |
        | Code | X/5 | ... |
        | Complétude | X/5 | ... |
        | Profondeur | X/5 | ... |
        | **TOTAL** | **X/20** | |

        ### VERDICT
        - Si TOTAL >= 15/20 → écris exactement : ✅ APPROVED
        - Si TOTAL < 15/20  → écris exactement : ❌ REVISION NEEDED

        ### CORRECTIONS DEMANDÉES (si REVISION NEEDED)
        Liste numérotée des corrections précises que le Writer doit apporter.
        Sois spécifique : "Ajoute un exemple de code pour X" plutôt que "améliore le code".

        IMPORTANT : Sois impartial et rigoureux. Un tutoriel sans code Python fonctionnel
        ne peut pas dépasser 8/20."""
    )

    return scout, reader, writer, critic


def get_professor(db):
    return AssistantAgent(
        name="Professor",
        model_client=get_model("llama-3.3-70b-versatile", max_tokens=1500),
        tools=[db.query_kb],
        system_message="""Tu es un professeur pédagogue.
        Pour répondre à l'étudiant, interroge TOUJOURS la base de connaissances via 'query_kb'.
        Utilise les informations récupérées pour expliquer les concepts clairement et simplement."""
    )