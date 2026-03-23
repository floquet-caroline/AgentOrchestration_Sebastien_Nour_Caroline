import asyncio
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from tools import VectorDB
from agents import get_research_team, get_professor
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination

console = Console()


async def run_agent_step(agent, task: str, step_label: str) -> str:
    """Lance un agent en solo pour une seule étape et retourne son dernier message."""
    team = RoundRobinGroupChat([agent], termination_condition=MaxMessageTermination(2))

    # run_stream permet de voir l'agent "réfléchir" en temps réel
    async for message in team.run_stream(task=task):
        content = getattr(message, "content", "")
        source = getattr(message, "source", "")
        if content and isinstance(content, str) and source == agent.name:
            # Affiche un aperçu du travail en cours
            preview = content[:300] + ("..." if len(content) > 300 else "")
            console.print(f"[dim]{preview}[/dim]")

    # Récupère le résultat final proprement
    result = await team.run(task=task)
    for msg in reversed(result.messages):
        if msg.source == agent.name and getattr(msg, "content", ""):
            return msg.content
    return ""


async def main():
    db = VectorDB()
    task = "Explique le RAG (Retrieval-Augmented Generation) en détail pour un développeur senior avec exemples Python."

    console.print(Rule("[bold cyan]🤖 AutoGen Research Pipeline[/bold cyan]"))

    # ─── ÉTAPE 1 : Scout cherche les URLs ────────────────────────────────────
    console.print(Panel("Recherche de sources sur le web...", title="[bold blue]🔍 Scout[/bold blue]", border_style="blue"))
    scout, reader, writer = get_research_team(db)

    urls_content = await run_agent_step(scout, task, "Scout")
    if not urls_content:
        console.print("[red]❌ Scout n'a pas trouvé d'URLs. Arrêt.[/red]")
        return

    # ─── ÉTAPE 2 : Reader extrait et synthétise ──────────────────────────────
    console.print(Panel("Lecture et synthèse des sources...", title="[bold yellow]📖 Reader[/bold yellow]", border_style="yellow"))

    synthesis = await run_agent_step(reader, urls_content, "Reader")
    if not synthesis:
        console.print("[red]❌ Reader n'a pas produit de synthèse. Arrêt.[/red]")
        return

    # ─── ÉTAPE 3 : Writer rédige le tutoriel ─────────────────────────────────
    console.print(Panel("Rédaction du tutoriel Markdown...", title="[bold green]✍️  Writer[/bold green]", border_style="green"))

    tutorial = await run_agent_step(writer, synthesis, "Writer")
    if not tutorial or len(tutorial) < 300:
        console.print("[red]❌ Writer n'a pas produit de contenu suffisant.[/red]")
        return

    # ─── Sauvegarde horodatée ────────────────────────────────────────────────
    filename = f"tuto_{datetime.now().strftime('%H%M%S')}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(tutorial)

    console.print(Rule())
    console.print(f"[bold green]✅ Tutoriel sauvegardé dans '[cyan]{filename}[/cyan]' ({len(tutorial)} caractères)[/bold green]")

    # ─── PHASE 2 : Mode Professeur ───────────────────────────────────────────
    console.print(Rule("[bold magenta]🎓 Mode Professeur[/bold magenta]"))
    console.print("[dim]Posez vos questions sur le tutoriel. Tapez 'exit' pour quitter.[/dim]\n")

    prof = get_professor(db)
    qa_team = RoundRobinGroupChat([prof], termination_condition=MaxMessageTermination(2))

    while True:
        question = input("Vous : ").strip()
        if question.lower() in ["exit", "quit", ""]:
            break

        result = await qa_team.run(task=question)
        for msg in result.messages:
            if msg.source == "Professor":
                console.print(Panel(msg.content, title="[bold magenta]Professeur[/bold magenta]", border_style="magenta"))


if __name__ == "__main__":
    asyncio.run(main())