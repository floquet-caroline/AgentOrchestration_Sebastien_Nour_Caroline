import datetime
from tavily import TavilyClient
import chromadb
from sentence_transformers import SentenceTransformer
from config import TAVILY_API_KEY
import os

os.environ["HF_TOKEN"] = ""

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

_embed_model = None

def get_embedding_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embed_model


class VectorDB:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(name="tutoriels")

    def index_segments(self, text_content: str) -> str:
        """
        Indexe un bloc de texte dans la base de données vectorielle.
        Argument: text_content (str): Le contenu textuel complet à indexer.
        """
        if not text_content.strip():
            return "Erreur : Contenu vide."

        # Découpage en chunks de 800 chars avec chevauchement de 200
        chunks = [text_content[i:i+1000] for i in range(0, len(text_content), 800)]

        ids = [f"id_{datetime.datetime.now().timestamp()}_{i}" for i in range(len(chunks))]
        embeddings = get_embedding_model().encode(chunks).tolist()

        self.collection.add(documents=chunks, ids=ids, embeddings=embeddings)
        return f"Succès : {len(chunks)} segments indexés."

    def query_kb(self, question: str) -> str:
        """
        Interroge la base de connaissances avec une question en langage naturel.
        Argument: question (str): La question à poser.
        """
        query_embedding = get_embedding_model().encode(question).tolist()
        results = self.collection.query(query_embeddings=[query_embedding], n_results=2)
        return "\n".join(results['documents'][0]) if results['documents'] else "Aucune info trouvée."


def find_urls(query: str) -> list:
    """
    Recherche des URLs pertinentes sur le web pour une requête donnée.
    Argument: query (str): Le sujet à rechercher.
    Retourne une liste de dicts avec 'url' et 'summary'.
    """
    results = tavily_client.search(query=query, max_results=3)["results"]  # 5 → 3 pour limiter les tokens
    return [
        {"url": r["url"], "summary": r.get("content", "")[:200]}  # 300 → 200 chars
        for r in results
    ]


def extract_content(url: str) -> list:
    """
    Extrait le contenu textuel d'une URL.
    Argument: url (str): L'URL de la page à lire.
    Retourne une liste avec le contenu extrait.
    """
    try:
        raw = tavily_client.extract(urls=[url])['results'][0]['raw_content']
        clean = " ".join(raw.split())[:1500]  # 3000 → 1500 chars pour rester sous les limites Groq
        return [{"source": url, "content": clean}]
    except Exception as e:
        return [{"source": url, "content": f"Erreur d'extraction : {e}"}]