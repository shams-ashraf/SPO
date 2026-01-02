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
        return """Du bist ein Assistent für den Masterstudiengang Biomedical Engineering (MBE) an der Hochschule Anhalt.

Regeln:
- Antworte ausschließlich auf Basis der bereitgestellten Dokumente und des letzten Gesprächskontexts.
- Kein externes Wissen oder Annahmen.
- Fehlende Informationen klar benennen.
- Kurz, sachlich und präzise antworten.
- Quellen und Seitenzahlen immer angeben.
- Bei Folgefragen auf das zuletzt genannte Thema beziehen.

   """
    
    else:
        return """You are an assistant for the Master Biomedical Engineering (MBE) program at Hochschule Anhalt.

Rules:
- Answer ONLY using the provided documents and recent conversation.
- Do NOT add external knowledge or assumptions.
- If information is missing, say so clearly.
- Be concise and factual.
- Always cite the document source and page.
- For follow-up questions, refer to the last specific topic mentioned.

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