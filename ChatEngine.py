import requests
import os
import time
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

if not GROQ_API_KEY:
    raise ValueError("⚠️ GROQ_API_KEY not set! Please add it to environment variables.")

GROQ_RATE_LIMIT_UNTIL = 0

def get_last_two_exchanges(chat_history):
    if not chat_history or len(chat_history) < 4:
        return ""

    pairs = []
    i = 0
    while i < len(chat_history) - 1:
        if chat_history[i]["role"] == "user" and chat_history[i + 1]["role"] == "assistant":
            pairs.append((
                chat_history[i]["content"],
                chat_history[i + 1]["content"][:250]  # أقصر
            ))
            i += 2
        else:
            i += 1

    last_pairs = pairs[-2:]

    # بدون labels
    return "\n".join(
        f"Q: {q}\nA: {a}"
        for q, a in last_pairs
    )


def get_embedding_function():
    """Initialize embedding function for multilingual support"""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="intfloat/multilingual-e5-large"
    )


def get_system_prompt(language):
    """Generate system prompt based on language"""
    if language == "de":
        return """Du bist ein präziser und professioneller Assistent für den Master Biomedical Engineering (MBE) Studiengang an der Hochschule Anhalt.

WICHTIGE REGELN:

1. QUELLENBASIERTE ANTWORTEN:
   - Antworte NUR basierend auf bereitgestellten Dokumenten oder Gesprächshistorie
   - Bei Folgefragen: Nutze die Gesprächshistorie zusammen mit Dokumenten
   - Bei Dokumentzusammenfassungen: Priorisiere DOKUMENTE über Gesprächshistorie
   - Ohne relevante Informationen: Sage klar "Keine ausreichenden Informationen in den verfügbaren Dokumenten"

2. SPRACHE & STIL:
   - Antworte auf Deutsch
   - Sei prägnant, klar und professionell
   - Verwende Aufzählungen oder Nummerierungen für Listen
   - Zitiere immer Quellen (z.B. "Laut SPO MBE 2024, Seite X...")

3. INHALTLICHE ANFORDERUNGEN:
   - NIEMALS halluzinieren oder externes Wissen hinzufügen
   - Bei Zählungen: Sei präzise und vollständig

4. KONTEXT-HANDLING:
   - Die bereitgestellte Historie zeigt die letzten 2 Gesprächspaare
   - Nutze diese Info wenn die aktuelle Frage darauf Bezug nimmt
   - WICHTIG: Wenn nach Dokumentzusammenfassung gefragt wird, konzentriere dich hauptsächlich auf die Dokumentquellen, nicht auf die Gesprächshistorie"""
    
    else:
        return """You are a precise and professional assistant for the Master Biomedical Engineering (MBE) program at Hochschule Anhalt.

IMPORTANT RULES:

1. SOURCE-BASED ANSWERS:
   - Answer ONLY based on provided documents or conversation history
   - For follow-up questions: Use conversation history together with documents
   - For document-level summaries: Prioritize DOCUMENTS over conversation history
   - Without relevant information: Clearly state "No sufficient information in the available documents"

2. LANGUAGE & STYLE:
   - Answer in English only
   - Be concise, clear, and professional
   - Use bullet points or numbering for lists
   - Always cite sources (e.g., "According to SPO MBE 2024, page X...")

3. CONTENT REQUIREMENTS:
   - NEVER hallucinate or add external knowledge
   - For counting/lists: Be precise and complete

4. CONTEXT HANDLING:
   - The provided history shows the last 2 conversation pairs
   - Use this info when the current question refers to it
   - IMPORTANT: If the current question is a document-level summary, prioritize document sources over conversation history
   
   """

def answer_question_with_groq(query, relevant_chunks, chat_history=None, user_language="en"):
    """Generate answer using Groq API with context from documents and chat history"""
    global GROQ_RATE_LIMIT_UNTIL

    # Check rate limit
    now = time.time()
    if now < GROQ_RATE_LIMIT_UNTIL:
        wait_seconds = int(GROQ_RATE_LIMIT_UNTIL - now)
        if user_language == "de":
            return (
                f"⏳ Groq Rate-Limit erreicht. Bitte warte {wait_seconds} Sekunden vor der nächsten Anfrage.",
                []
            )
        else:
            return (
                f"⏳ Groq rate limit reached. Please wait {wait_seconds} seconds before sending a new request.",
                []
            )

    # Build context from relevant chunks
    context_parts = []
    used_chunks = []
    
    for i, chunk in enumerate(relevant_chunks[:8], 1):
        source = chunk["metadata"].get("source", "Unknown")
        page = chunk["metadata"].get("page", "N/A")
        content = chunk["content"]
        chunk_type = chunk["metadata"].get("type", "text")
        table_num = chunk["metadata"].get("table_number", "N/A")

        # Format table chunks specially
        if chunk_type == "table_with_context" and table_num != "N/A":
            context_parts.append(f"[Table {table_num} | Source: {source} | Page: {page}]\n{content}")
        else:
            context_parts.append(f"[Source: {source} | Page: {page}]\n{content}")

        used_chunks.append({
            "source": source,
            "page": page,
            "content": content
        })

    context = "\n\n---\n\n".join(context_parts)

    # Get conversation history only if at least 4 messages exist (2 complete pairs)
    conversation_summary = get_last_two_exchanges(chat_history)

    # Get system prompt
    system_prompt = get_system_prompt(user_language)

    # Build request
    data = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
"content": f"""{conversation_summary}

Sources:
{context}

Question:
{query}"""
            }
        ],
        "temperature": 0.05,
        "max_tokens": 1000,
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json=data,
            timeout=60
        )
        response.raise_for_status()

        answer = response.json()["choices"][0]["message"]["content"].strip()
        return answer, used_chunks

    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 429:
            retry_after = e.response.headers.get("Retry-After")
            wait_time = int(retry_after) if retry_after else 60

            GROQ_RATE_LIMIT_UNTIL = time.time() + wait_time

            if user_language == "de":
                return (
                    f"⛔ Groq Rate-Limit erreicht.\n"
                    f"⏳ Bitte warte {wait_time} Sekunden vor einem erneuten Versuch.",
                    []
                )
            else:
                return (
                    f"⛔ Groq rate limit reached.\n"
                    f"⏳ Please wait {wait_time} seconds before trying again.",
                    []
                )

        error_msg = f"❌ HTTP Fehler: {str(e)}" if user_language == "de" else f"❌ HTTP Error: {str(e)}"
        return error_msg, []

    except Exception as e:
        error_msg = f"❌ Fehler: {str(e)}" if user_language == "de" else f"❌ Error: {str(e)}"
        return error_msg, []
