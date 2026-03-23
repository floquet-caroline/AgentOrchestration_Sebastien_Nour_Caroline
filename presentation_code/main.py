import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from datetime import datetime

from tools import VectorDB
from agents import get_research_team, get_professor
from autogen_agentchat.teams import SelectorGroupChat, RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination
from config import get_model

console = Console()


async def main():
    db = VectorDB(reset=True)
    task = "Explique le RAG (Retrieval-Augmented Generation) en détail pour un développeur senior avec exemples Python."

    scout, reader, writer, critic = get_research_team(db)

    # ── SelectorGroupChat ────────────────────────────────────────────────────
    # Un LLM lit l'historique complet à chaque tour et choisit quel agent parle.
    # La conversation s'arrête dès que le Critic écrit "APPROVED".
    team = SelectorGroupChat(
        participants=[scout, reader, writer, critic],
        model_client=get_model("llama-3.3-70b-versatile"),
        termination_condition=TextMentionTermination("APPROVED"),
        selector_prompt=(
            "Tu orchestre une équipe de recherche. "
            "Voici les agents disponibles et leur rôle :\n"
            "- Scout : trouve des URLs sur le web\n"
            "- Reader : lit les URLs et produit une synthèse structurée. "
            "Si le Writer pose une question avec [BESOIN_INFO:...], le Reader doit répondre.\n"
            "- Writer : rédige le tutoriel Markdown à partir de la synthèse. "
            "Peut poser des questions au Reader avec [BESOIN_INFO: <point>].\n"
            "- Critic : évalue le tutoriel. Écrit 'APPROVED' si le tutoriel est bon, "
            "sinon liste les corrections pour le Writer.\n\n"
            "Historique de la conversation :\n{history}\n\n"
            "Quel agent doit parler maintenant ? Réponds uniquement avec le nom de l'agent."
        )
    )

    console.print(Rule("[bold cyan]🤖 AutoGen Research Pipeline[/bold cyan]"))
    console.print(Panel(task, title="[bold]Sujet[/bold]", border_style="cyan"))

    # ── Lancement du pipeline ────────────────────────────────────────────────
    tutorial = ""
    async for message in team.run_stream(task=task):
        source = getattr(message, "source", "")
        content = getattr(message, "content", "")
        if not content or not isinstance(content, str):
            continue

        # Affichage en temps réel avec couleur par agent
        colors = {"Scout": "blue", "Reader": "yellow", "Writer": "green", "Critic": "red"}
        color = colors.get(source, "dim")
        preview = content[:300] + ("..." if len(content) > 300 else "")
        console.print(f"[{color}][{source}][/{color}] [dim]{preview}[/dim]")

        if source == "Writer":
            tutorial = content  # On garde la dernière version du Writer

    # ── Indexation du tutoriel final ─────────────────────────────────────────
    if not tutorial or len(tutorial) < 300:
        console.print("[red]❌ Tutoriel insuffisant. Arrêt.[/red]")
        return

    console.print("[dim]→ Indexation en base vectorielle...[/dim]")
    console.print(f"[dim]{db.index_segments(tutorial)}[/dim]")

    filename = f"tuto_{datetime.now().strftime('%H%M%S')}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(tutorial)

    console.print(Rule())
    console.print(f"[bold green]✅ Tutoriel sauvegardé → [cyan]{filename}[/cyan] ({len(tutorial)} caractères)[/bold green]")

    # ── Mode Professeur ──────────────────────────────────────────────────────
    console.print(Rule("[bold magenta]🎓 Mode Professeur[/bold magenta]"))
    console.print("[dim]Posez vos questions. Tapez 'exit' pour quitter.[/dim]\n")

    prof = get_professor(db)
    qa_team = RoundRobinGroupChat([prof], termination_condition=MaxMessageTermination(2))

    while True:
        question = input("Vous : ").strip()
        if question.lower() in ("exit", "quit", ""):
            break
        result = await qa_team.run(task=question)
        for msg in result.messages:
            if msg.source == "Professor":
                console.print(Panel(msg.content, title="[bold magenta]Professeur[/bold magenta]", border_style="magenta"))


if __name__ == "__main__":
    asyncio.run(main())