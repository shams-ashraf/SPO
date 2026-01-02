import streamlit as st
import uuid
import chromadb
import os
import re
import time
import shutil 
from styles import load_custom_css

from DocumentProcessor import (
    get_files_from_folder,
    get_file_hash,
    load_cache,
    extract_pdf_detailed,
    save_cache
)
from ChatEngine import get_embedding_function, answer_question_with_groq

st.set_page_config(
    page_title="Biomedical Document Chatbot",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_custom_css()
DOCS_FOLDER = "/mount/src/spo/documents"
CACHE_FOLDER = os.getenv("CACHE_FOLDER", "./cache")
CHROMA_FOLDER = "./chroma_db"

os.makedirs(DOCS_FOLDER, exist_ok=True)
os.makedirs(CACHE_FOLDER, exist_ok=True)
os.makedirs(CHROMA_FOLDER, exist_ok=True)

if "collection" not in st.session_state:
    st.session_state.collection = None

if "chats" not in st.session_state:
    st.session_state.chats = {}
    st.session_state.active_chat = None

if "user_language" not in st.session_state:
    st.session_state.user_language = "en"

# ===== DEBUG ADDITION (1) =====
if "all_chunks_debug" not in st.session_state:
    st.session_state.all_chunks_debug = []

if "show_chunks" not in st.session_state:
    st.session_state.show_chunks = False
# ==============================

client = chromadb.PersistentClient(path=CHROMA_FOLDER)  

collections = client.list_collections()

if collections:
    collection = client.get_collection(
        name=collections[0].name,
        embedding_function=get_embedding_function()
    )
    st.session_state.collection = collection
else:
    with st.spinner("📚 Processing documents... This may take a while for the first time."):
        files = get_files_from_folder()
        st.write("🔍 Searching for documents in:")
        st.code(DOCS_FOLDER)
        st.write(f"Found {len(files)} files:")
        if files:
            for f in files:
                st.write(f"- {f}")
        else:
            st.error("🚨 No files found! Check the folder path and file extensions.")
            st.stop()
        if not files:
            st.error("No documents found in the documents folder!")
            st.stop()

        collection = client.create_collection(
            name="biomed_docs2",
            embedding_function=get_embedding_function(),
            metadata={"hnsw:space": "cosine"}  
        )

        all_chunks = []
        all_meta = {}
        all_ids = []

        processed_files = []  
        processed_count = 0  

        for idx, path in enumerate(files):
            name = os.path.basename(path)
            ext = name.split(".")[-1].lower()
            key = f"{get_file_hash(path)}_{ext}"
            cached = load_cache(key)

            if cached:
                info = cached
                st.success(f"✅ Loaded from cache: {name}") 
            else:
                st.info(f"📄 Processing: {name} ...") 
                if ext == "pdf":
                    info, error = extract_pdf_detailed(path)
                else:
                    st.warning(f"⚠️ Skipped unsupported file: {name}")
                    continue
                if error:
                    st.warning(f"⚠️ Error in {name}: {error}")
                    continue
                save_cache(key, info)

            st.success(f"✅ Processed successfully: {name}")
            processed_files.append(name)
            processed_count += 1

            for c in info["chunks"]:
                all_chunks.append(c["content"])
                all_meta[len(all_chunks) - 1] = c["metadata"]
                all_ids.append(f"chunk_{idx}_{len(all_chunks)}")

                # ===== DEBUG ADDITION (2) =====
                st.session_state.all_chunks_debug.append(c)
                # ==============================

        if processed_count > 0:
            st.success(f"🎉 All done! Processed {processed_count} documents successfully!")
            st.write("**Processed documents:**")
            for file_name in processed_files:
                st.write(f"- {file_name}")
        else:
            st.error("❌ No documents were processed!")

        batch_size = 300
        for i in range(0, len(all_chunks), batch_size):
            collection.add(
                documents=all_chunks[i:i+batch_size],
                metadatas=[all_meta[j] for j in range(i, min(i+batch_size, len(all_chunks)))],
                ids=all_ids[i:i+batch_size]
            )

        st.session_state.collection = collection
        st.success(f"✅ Processed {len(files)} documents successfully!")

if not st.session_state.chats:
    cid = f"chat_{uuid.uuid4().hex[:6]}"
    st.session_state.chats[cid] = {
        "title": "New Chat",
        "messages": [],
        "context": []
    }
    st.session_state.active_chat = cid

st.markdown("""
<div class="main-card">
    <h1 style='text-align:center;margin:0;'>🧬 Biomedical Document Chatbot</h1>
</div>

<div class="main-card">
    <p style="text-align:center; font-size:1.1rem;">Answers <strong>only</strong> from official documents • Supports English & German • Remembers conversation</p>
    <h3 style="color:#00d9ff;">Try these examples:</h3>
    <ul style="font-size:1.05rem;">
        <li>What are the requirements for registering the master's thesis?</li>
        <li>Was sind die Regelungen für die Masterarbeit?</li>
    </ul>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    
    st.markdown("# 🧬 BioMed Chat")
    
    st.markdown("### 🌐 Select Your Language")
    language_option = st.selectbox(
        "Choose your question language:",
        options=["English", "Deutsch"],
        index=0 if st.session_state.user_language == "en" else 1,
        key="language_selector"
    )
    st.session_state.user_language = "en" if language_option == "English" else "de"
    
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        cid = f"chat_{uuid.uuid4().hex[:6]}"
        st.session_state.chats[cid] = {
            "title": "New Chat",
            "messages": [],
            "context": []
        }
        st.session_state.active_chat = cid
        st.rerun()

    st.markdown("### ⚙️ System")

    # ===== DEBUG ADDITION (3) =====
    if st.button("🧾 Show all extracted chunks", use_container_width=True):
        st.session_state.show_chunks = not st.session_state.show_chunks
    # ==============================

    if st.button("🧹 Reset all documents & index", use_container_width=True):
        try:
            if os.path.exists(CHROMA_FOLDER):
                shutil.rmtree(CHROMA_FOLDER)

            if os.path.exists(CACHE_FOLDER):
                shutil.rmtree(CACHE_FOLDER)

            st.success("✅ Index & cache deleted. Rebuilding...")
            time.sleep(1)
            st.rerun()

        except Exception as e:
            st.error(f"❌ Reset failed: {e}")

    st.markdown("### 💬 Your Chats")

    for cid in reversed(list(st.session_state.chats.keys())):   
        chat = st.session_state.chats[cid]
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(f"💬 {chat['title'][:35]}...", key=f"open_{cid}", use_container_width=True):
                st.session_state.active_chat = cid
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{cid}"):
                del st.session_state.chats[cid]
                if st.session_state.active_chat == cid:
                    st.session_state.active_chat = next(iter(st.session_state.chats), None)
                st.rerun()

chat = st.session_state.chats[st.session_state.active_chat]
for m in chat["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if query := st.chat_input("Ask anything about the MBE program documents..."):
    chat["messages"].append({"role": "user", "content": query})
    if chat["title"] == "New Chat":
        chat["title"] = query[:40] + "..." if len(query) > 40 else query

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents & thinking..."):
            chunks = []

            res = st.session_state.collection.query(
                query_texts=[query],
                n_results=8  # fewer chunks → fewer tokens
            )

            for docs, metas in zip(res["documents"], res["metadatas"]):
                for d, m in zip(docs, metas):
                    chunks.append({
                        "content": d,
                        "metadata": m
                    })

            answer, used_chunks = answer_question_with_groq(
                query, 
                chunks, 
                chat["messages"],
                user_language=st.session_state.user_language
            )
            st.markdown(answer)

            match = re.search(r'wait (\d+) seconds', answer.lower())
            if match:
                remaining = int(match.group(1))
                countdown = st.empty()
                while remaining > 0:
                    countdown.warning(f"⏳ Please wait {remaining} seconds before sending a new request...")
                    time.sleep(1)
                    remaining -= 1
                countdown.success("✅ You can now send a new question.")

            st.markdown("### 📚 Answer was based on the following document excerpts:")
            for i, ch in enumerate(used_chunks, 1):
                with st.expander(f"📄 {ch['source']} — Page {ch['page']}"):
                    st.markdown(ch["content"])

    chat["messages"].append({"role": "assistant", "content": answer})
    chat["context"] = chunks

# ===== DEBUG ADDITION (4) =====
if st.session_state.show_chunks:
    st.markdown("## 🧾 All Extracted Chunks (Debug View)")

    for i, ch in enumerate(st.session_state.all_chunks_debug, 1):
        meta = ch.get("metadata", {})
        with st.expander(f"Chunk {i} | {meta.get('type')} | Page {meta.get('page')}"):
            st.json(meta)
            st.markdown(ch["content"])
# ==============================
