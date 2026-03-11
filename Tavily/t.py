from tavily import TavilyClient
import asyncio
import sqlite3
import datetime
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import MaxMessageTermination

DB_PATH = "Tavily/conversations.db"

#--------------------------------------------------------------
# BASE DE DONNÉES
def init_db():
    """Initialise la base SQLite et crée les tables si elles n'existent pas."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL,
            tutorial_content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)
    con.commit()
    con.close()

def create_session(topic: str, tutorial_content: str) -> int:
    """Crée une nouvelle session et retourne son id."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO sessions (topic, tutorial_content, created_at) VALUES (?, ?, ?)",
        (topic, tutorial_content, datetime.datetime.now().isoformat())
    )
    session_id = cur.lastrowid
    con.commit()
    con.close()
    return session_id

def save_message(session_id: int, role: str, content: str):
    """Enregistre un message (question ou réponse) dans la session."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (session_id, role, content, datetime.datetime.now().isoformat())
    )
    con.commit()
    con.close()

def load_session(session_id: int) -> dict:
    """Charge une session existante avec son historique de messages."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT topic, tutorial_content, created_at FROM sessions WHERE id = ?", (session_id,))
    row = cur.fetchone()
    if not row:
        con.close()
        return None
    topic, tutorial_content, created_at = row
    cur.execute(
        "SELECT role, content, timestamp FROM messages WHERE session_id = ? ORDER BY id",
        (session_id,)
    )
    messages = [{"role": r, "content": c, "timestamp": t} for r, c, t in cur.fetchall()]
    con.close()
    return {"id": session_id, "topic": topic, "tutorial_content": tutorial_content,
            "created_at": created_at, "messages": messages}

def list_sessions() -> list:
    """Retourne toutes les sessions enregistrées."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT id, topic, created_at FROM sessions ORDER BY id DESC")
    rows = [{"id": r[0], "topic": r[1], "created_at": r[2]} for r in cur.fetchall()]
    con.close()
    return rows

model_client_large = OpenAIChatCompletionClient(
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

model_client_search = OpenAIChatCompletionClient(
        model="llama-3.1-8b-instant",
        api_key="", # Remplacez par votre clé
        base_url="https://api.groq.com/openai/v1", # Indispensable pour Groq
        model_info={
        "vision": False,       # Le modèle supporte-t-1 l'image ?
        "function_calling": True, 
        "json_output": True,
        "family": "unknown"    # On peut mettre unknown ici
    }   
)

tavily_client = TavilyClient(api_key="")

#--------------------------------------------------------------
# TOOLS
def find_urls(query: str):
    """Trouve les 5 meilleures URLs pour un sujet donné, avec un résumé de chaque page."""
    search_result = tavily_client.search(query=query, max_results=5)
    return [
        {"url": r["url"], "summary": r.get("content", "")[:300]}
        for r in search_result["results"]
    ]


def extract_page_content(url: str):
    """Extrait le contenu d'une page web (tronqué à 2500 caractères)."""
    extraction = tavily_client.extract(urls=[url])
    content = extraction['results'][0]['raw_content']
    return content[:2500] + "..." if len(content) > 2500 else content
#--------------------------------------------------------------
# AGENTS
scout = AssistantAgent(
    "Scout",
    model_client=model_client_search,
    tools=[find_urls], # <--- Vérifie que le NOM ici est celui de ta fonction
    system_message="""Tu es un expert en recherche. 
    Pour chercher sur le web, tu as l'interdiction stricte d'utiliser 'brave_search'. 
    Tu DOIS utiliser uniquement l'outil 'find_urls'.
    Si tu as besoin de chercher quelque chose, appelle 'find_urls' avec ta requête.
    IMPORTANT : privilégie toujours les sources les plus récentes (documentation officielle, guides à jour, changelogs récents). Évite les articles obsolètes ou anciens."""
)

reader = AssistantAgent(
    name="Reader",
    model_client=model_client_large,
    tools=[extract_page_content],
    system_message="""Tu lis et synthétises le contenu des pages web.
    Utilise 'extract_page_content' sur les 2 URLs les plus pertinentes parmi celles fournies par le Scout.
    Tu as l'interdiction stricte d'utiliser 'brave_search'.
    IMPORTANT : vérifie que les pages lues correspondent bien à la documentation la plus récente.
    Après extraction, produis une synthèse structurée et détaillée du contenu pertinent pour la tâche demandée.""",
)

fact_checker = AssistantAgent(
    name="FactChecker",
    model_client=model_client_large,
    tools=[find_urls, extract_page_content],
    system_message="""Tu es un vérificateur de faits technique rigoureux.
    Tu reçois la synthèse produite par le Reader et tu dois la valider en croisant avec d'autres sources.
    Ta procédure :
    1. Identifie les affirmations clés : numéros de version, noms d'API, exemples de code, comportements décrits.
    2. Utilise 'find_urls' pour chercher une source de croisement différente (ex: changelog officiel, GitHub releases, doc officielle).
    3. Utilise 'extract_page_content' sur l'URL la plus pertinente pour vérifier.
    4. Produis un rapport structuré avec :
       - ✅ Points confirmés par la source croisée
       - ⚠️ Points incertains ou potentiellement obsolètes (avec justification)
       - ❌ Erreurs détectées ou hallucinations potentielles
       - La synthèse corrigée/validée à transmettre au Writer.
    Tu as l'interdiction stricte d'utiliser 'brave_search'.""",
)

writer = AssistantAgent(
    name="Writer",
    model_client=model_client_large,
    system_message="""Tu es un rédacteur technique expert. Tu crées des tutoriels complets et approfondis à partir du contenu fourni par le Lecteur.
    Tu as l'interdiction stricte d'utiliser 'brave_search'.
    IMPORTANT : assure-toi que tous les exemples de code utilisent la syntaxe et les APIs les plus récentes. Mentionne la version concernée si possible.
    FORMAT : rédige TOUJOURS en Markdown valide avec :
    - Un titre principal et une introduction motivante
    - Une section Prérequis
    - Des étapes numérotées et détaillées
    - Plusieurs blocs de code annotés (```)
    - Une section pièges courants / bonnes pratiques
    - Une section ressources / pour aller plus loin
    Sois exhaustif : un débutant doit pouvoir suivre le tuto de bout en bout sans chercher ailleurs.""",
)

"""groupchat = GroupChat(agents=[user_proxy, scout, reader, writer], messages=[], max_round=12)
manager = GroupChatManager(groupchat=groupchat, llm_config={"config_list": config_list})

user_proxy.initiate_chat(
    manager,
    message="Trouve comment utiliser les 'Zustand slices' en TypeScript et fais-moi un tuto."
)"""

termination = MaxMessageTermination(max_messages=5)  # task + Scout + Reader + FactChecker + Writer
team = RoundRobinGroupChat(
    [scout, reader, fact_checker, writer],
    termination_condition=termination,
)


async def interactive_qa(tutorial_content: str, session_id: int, history: list = None):
    """Phase interactive : l'utilisateur pose des questions, le Professeur répond."""
    context = tutorial_content[:3000] + "..." if len(tutorial_content) > 3000 else tutorial_content

    # Injecte l'historique passé dans le system message si reprise de session
    history_text = ""
    if history:
        lines = []
        for msg in history:
            prefix = "Étudiant" if msg["role"] == "user" else "Professeur"
            lines.append(f"**{prefix} [{msg['timestamp'][:10]}]** : {msg['content']}")
        history_text = "\n\n--- HISTORIQUE DE LA SESSION PRÉCÉDENTE ---\n" + "\n\n".join(lines) + "\n--- FIN DE L'HISTORIQUE ---"

    professor = AssistantAgent(
        name="Professor",
        model_client=model_client_large,
        system_message=f"""Tu es un professeur expert et pédagogue. Un étudiant va te poser des questions sur le tutoriel suivant.
        Réponds de façon claire, précise et détaillée, comme un professeur qui explique à un étudiant.
        Utilise des analogies, des exemples concrets et des reformulations si nécessaire.
        Si la question dépasse le cadre du tutoriel, tu peux y répondre en restant dans le domaine technique.
        Réponds toujours en Markdown avec des blocs de code si pertinent.

        --- TUTORIEL DE RÉFÉRENCE ---
        {context}
        --- FIN DU TUTORIEL ---
        {history_text}""",
    )

    qa_termination = MaxMessageTermination(max_messages=2)

    print("\n" + "="*60)
    print("Mode Question-Réponse — Pose tes questions sur le tutoriel")
    print("(tape 'quit' ou 'exit' pour terminer)")
    print("="*60 + "\n")

    while True:
        question = input("Ta question : ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Fin de la session. Bonne continuation !")
            break

        save_message(session_id, "user", question)

        qa_team = RoundRobinGroupChat([professor], termination_condition=qa_termination)
        result = await qa_team.run(task=question)

        for msg in result.messages:
            if getattr(msg, "source", "") == "Professor":
                answer = msg.content
                print(f"\nProfesseur :\n{answer}\n")
                save_message(session_id, "professor", answer)
                break


async def main():
    init_db()

    # --- Reprise ou nouvelle session ---
    sessions = list_sessions()
    session_id = None
    tutorial_content = None
    history = []

    if sessions:
        print("\n=== Sessions existantes ===")
        for s in sessions:
            print(f"  [{s['id']}] {s['topic'][:80]} — {s['created_at'][:10]}")
        choice = input("\nReprise d'une session (entre l'id) ou nouvelle session (appuie sur Entrée) : ").strip()
        if choice.isdigit():
            data = load_session(int(choice))
            if data:
                session_id = data["id"]
                tutorial_content = data["tutorial_content"]
                history = data["messages"]
                print(f"\n=> Session '{data['topic'][:60]}' reprise ({len(history)} messages précédents).")
                print("=> Tutoriel rechargé depuis la base de données.")
                await interactive_qa(tutorial_content, session_id, history)
                return
            else:
                print("Session introuvable, démarrage d'une nouvelle session.")

    # --- Nouvelle session : génération du tutoriel ---
    task = "Explique ce qu'est un RAG, et a quoi il sert, hésite pas a aller dans les détails sans prendre d'exemple tres spécifique"

    result = await team.run(task=task)

    fact_checker_message = None
    writer_message = None
    for message in result.messages:
        if getattr(message, "source", "") == "FactChecker":
            fact_checker_message = getattr(message, "content", None)
        if getattr(message, "source", "") == "Writer":
            writer_message = getattr(message, "content", None)

    if fact_checker_message:
        print("\n=== Rapport du FactChecker ===\n")
        print(fact_checker_message)

    if writer_message:
        print("\n=== Réponse du Writer ===\n")
        print(writer_message)
        with open("Tavily/tutorial_output.md", "w", encoding="utf-8") as f:
            f.write(writer_message)
        print("\n=> Tutoriel sauvegardé dans Tavily/tutorial_output.md")

        session_id = create_session(task, writer_message)
        print(f"=> Session #{session_id} sauvegardée dans la base de données.")

        await interactive_qa(writer_message, session_id)
    else:
        print("Aucune réponse du Writer trouvée.")


if __name__ == "__main__":
    asyncio.run(main())