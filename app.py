from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
import json
from langchain_core.globals import set_debug, set_verbose, set_llm_cache
import streamlit as st
import os
import pickle
import faiss
import numpy as np
import pandas as pd
import plotly.express as px
import re
import pytesseract
from pdf2image import convert_from_path
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
from sklearn.decomposition import PCA
import langchain
import time
import tempfile
from pyvis.network import Network
from dotenv import load_dotenv

# Load environment variables from a .env file (if it exists)
load_dotenv()

# Fix missing attributes in langchain module
missing_attrs = ["verbose", "debug", "llm_cache"]
for attr in missing_attrs:
    if not hasattr(langchain, attr):
        setattr(langchain, attr, None)
        # Disable langchain debug/verbose globally

set_debug(False)
set_verbose(False)
set_llm_cache(None)

# 0. SYSTEM CONFIG (OCR & PATHS)
# SPEED BOOST: Allow Tesseract to use 4 CPU cores
os.environ["OMP_THREAD_LIMIT"] = "4"

# 1. Tesseract Configuration
# Defaults to standard Windows path, but can be overridden in .env
TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# 2. Poppler Configuration
# Can be left blank if Poppler is in the system PATH
POPPLER_BIN_PATH = os.getenv("POPPLER_PATH", None)

# 3. API Key Security (Used ONLY for Knowledge Graph extraction now)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.warning("⚠️ GOOGLE_API_KEY is not set. Knowledge Graph generation will fail. Please add it to your .env file.")

# 1. PATHS & SETUP
BASE_DIR = Path(__file__).parent
PDF_DIR = BASE_DIR / "PDF"
VECTOR_DB_DIR = BASE_DIR / "vector_db"

# Ensure directories exist
PDF_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = VECTOR_DB_DIR / "faiss.index"
DOCS_PATH = VECTOR_DB_DIR / "docs.pkl"
CHECKPOINT_PATH = VECTOR_DB_DIR / "checkpoint.pkl"
LOG_PATH = VECTOR_DB_DIR / "train.log"

# LOCAL EMBEDDING MODEL (384 Dimensions)
embedder = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}, 
    encode_kwargs={'normalize_embeddings': True}
)

st.set_page_config(page_title="VectorGraph-Engine", layout="wide")

# 2. SIDEBAR CONFIGURATION
with st.sidebar:
    st.header("⚙️ Configuration")
    st.info("Advanced Chunking Active (200-300 words target)")

    CHUNK_SIZE = st.number_input(
        "Chunk Size (Chars)", min_value=500, max_value=3000, value=1800, step=100
    )
    CHUNK_OVERLAP = st.number_input(
        "Chunk Overlap (Chars)", min_value=0, max_value=500, value=300, step=50
    )

    st.divider()
    st.write("📁 **File Uploader**")
    uploaded_files = st.file_uploader(
        "Add PDFs to Root", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_path = PDF_DIR / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"Saved {len(uploaded_files)} files!")
        st.rerun()

# 3. CORE HELPER FUNCTIONS
def append_log_to_file(message):
    """Writes logs to file with UTF-8 encoding and forces save."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {message}\n"

    if not LOG_PATH.exists():
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            f.write("")

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(entry)
        f.flush()
        os.fsync(f.fileno())

def save_checkpoint(last_file_id, last_chunk):
    with open(CHECKPOINT_PATH, "wb") as f:
        pickle.dump({"file_id": last_file_id, "chunk": last_chunk}, f)

def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, "rb") as f:
            return pickle.load(f)
    return None

def get_index(vector_dim=384): # Changed to 384 for MiniLM
    if INDEX_PATH.exists():
        return faiss.read_index(str(INDEX_PATH))
    else:
        index = faiss.IndexHNSWFlat(vector_dim, 32)
        faiss.write_index(index, str(INDEX_PATH))
        return index

def clean_legal_ocr(text):
    """Cleans noisy OCR text."""
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    text = re.sub(r'LUG.*?pies\.\]\s*[-.,A\s]+', ' ', text, flags=re.DOTALL)
    text = text.replace("inillions", "millions")
    text = text.replace("I'Eead Cornrnissio~ler", "Head Commissioner")
    text = text.replace("Gove~liment", "Government")
    text = " ".join(text.split())
    return text

def format_chunk(text, chunk_id, source_doc):
    """Rule 4 & 6: Formatting & Metadata Injection"""
    section_match = re.search(
        r'((Section|Article|Clause|CHAPTER)\s+[\d\w\.]+)', text, re.IGNORECASE
    )
    section_title = section_match.group(1) if section_match else "General Context"

    formatted_text = (
        f"[CHUNK ID: {chunk_id}]\n"
        f"[SOURCE: {source_doc} | SECTION: {section_title}]\n"
        f"Text:\n"
        f"“{text.strip()}”"
    )
    return formatted_text

def advanced_clean_text(text):
    """Rule 5: Cleaning & Noise Removal"""
    text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
    text = re.sub(r'(Page\s+\d+\s+of\s+\d+|Page\s+\d+)',
                  '', text, flags=re.IGNORECASE)
    text = re.sub(r'LUG.*?pies\.\]\s*[-.,A\s]+', ' ', text, flags=re.DOTALL)
    text = text.replace("inillions", "millions")
    text = text.replace("Gove~liment", "Government")
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()

def read_pdf_hybrid(file_path, logger_func=None, force_ocr=False):
    """Reads PDF with Real-Time Logging Support."""
    def local_log(msg):
        if logger_func:
            logger_func(msg)
        else:
            append_log_to_file(msg)

    full_text = ""
    is_scanned = False

    if not force_ocr:
        try:
            reader = PdfReader(file_path)
            if len(reader.pages) > 0:
                first_page = reader.pages[0].extract_text()
                if not first_page or len(first_page.strip()) < 50:
                    is_scanned = True

            if not is_scanned:
                temp_text = ""
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        temp_text += extracted + "\n"
                full_text = temp_text
                local_log(f"⚡ Fast Read: {file_path.name}")
        except Exception:
            is_scanned = True
    else:
        is_scanned = True

    if is_scanned:
        local_log(f"📷 OCR Started: {file_path.name} (Optimized Mode)")
        try:
            pdf_kwargs = {"dpi": 150}
            if POPPLER_BIN_PATH:
                pdf_kwargs["poppler_path"] = POPPLER_BIN_PATH
                
            images = convert_from_path(str(file_path), **pdf_kwargs)
            total_pages = len(images)
            for i, img in enumerate(images):
                if i % 5 == 0:
                    local_log(f"   ...Processing Page {i+1}/{total_pages}")
                img = img.convert('L')
                ocr_text = pytesseract.image_to_string(img)
                full_text += ocr_text + "\n"
        except Exception as e:
            local_log(f"❌ OCR Failed for {file_path.name}. Error: {e}")
            return []

    cleaned_text = advanced_clean_text(full_text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", ".\n", "\n", ". ", " ", ""],
        length_function=len
    )

    raw_chunks = splitter.split_text(cleaned_text)

    formatted_chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        if len(chunk_text) < 50:
            continue
        formatted_str = format_chunk(chunk_text, i, file_path.name)
        formatted_chunks.append(formatted_str)

    return formatted_chunks

# 4. RECURSIVE TRAINING PIPELINE
def train_vectors(restart_fresh=False):
    st.divider()
    st.subheader("🚦 Processing Terminal")
    log_placeholder = st.empty()
    prog_bar = st.progress(0)
    status_text = st.empty()
    recent_logs = []

    def realtime_logger(message):
        append_log_to_file(message)
        timestamp = datetime.now().strftime("%H:%M:%S")
        recent_logs.append(f"[{timestamp}] {message}")
        if len(recent_logs) > 8:
            recent_logs.pop(0)
        log_placeholder.code("\n".join(recent_logs), language="log")

    if restart_fresh:
        realtime_logger("🗑️ Deleting old data for fresh start...")
        if INDEX_PATH.exists():
            INDEX_PATH.unlink()
        if DOCS_PATH.exists():
            DOCS_PATH.unlink()
        if CHECKPOINT_PATH.exists():
            CHECKPOINT_PATH.unlink()
        
        index = faiss.IndexHNSWFlat(384, 32)
        faiss.write_index(index, str(INDEX_PATH))
        documents = []
        checkpoint = None
        realtime_logger("✅ Data Reset Complete. Starting Training...")
    else:
        checkpoint = load_checkpoint()
        index = get_index()
        documents = []
        if DOCS_PATH.exists():
            with open(DOCS_PATH, "rb") as f:
                documents = pickle.load(f)

    pdf_files = sorted(list(PDF_DIR.rglob("*.pdf")))

    if not pdf_files:
        st.error("No PDFs found in data/pdfs or its subfolders!")
        return

    resume_mode = False
    should_process_file = True

    if checkpoint and not restart_fresh:
        st.info(f"🔄 Resuming from: `{checkpoint['file_id']}`")
        resume_mode = True
        should_process_file = False

    total_pdfs = len(pdf_files)

    for pdf_i, pdf_file in enumerate(pdf_files):

        try:
            rel_path = pdf_file.relative_to(PDF_DIR)
            unique_id = str(rel_path)
            path_parts = rel_path.parts
            if len(path_parts) >= 2:
                source_website = path_parts[0]
                doc_type = path_parts[1]
            else:
                source_website = "root"
                doc_type = "unknown"
        except Exception:
            unique_id = pdf_file.name
            source_website = "unknown"
            doc_type = "unknown"

        if resume_mode:
            if unique_id == checkpoint["file_id"]:
                should_process_file = True
            if not should_process_file:
                continue

        status_text.markdown(f"**Processing:** `{unique_id}`")

        chunks = read_pdf_hybrid(pdf_file, logger_func=realtime_logger)
        total_chunks = len(chunks)

        if total_chunks == 0:
            realtime_logger(f"⚠️ Warning: No text found in {pdf_file.name}")

        for chunk_i, chunk_text in enumerate(chunks):
            if resume_mode and unique_id == checkpoint["file_id"] and chunk_i < checkpoint["chunk"]:
                continue

            try:
                vec = np.array(embedder.embed_query(chunk_text), dtype=np.float32)
                index.add(vec.reshape(1, -1))

                documents.append({
                    "text": chunk_text,
                    "pdf": pdf_file.name,
                    "source": source_website,
                    "type": doc_type,
                    "file_path": str(rel_path),
                    "timestamp": datetime.now(),
                    "chunk_id": chunk_i
                })

                save_checkpoint(unique_id, chunk_i)

                if chunk_i % 10 == 0:
                    faiss.write_index(index, str(INDEX_PATH))
                    with open(DOCS_PATH, "wb") as f:
                        pickle.dump(documents, f)

            except Exception as e:
                realtime_logger(f"Error embedding chunk {chunk_i}: {e}")

        prog_bar.progress((pdf_i + 1) / total_pdfs)
        resume_mode = False

        save_checkpoint(unique_id, total_chunks)
        faiss.write_index(index, str(INDEX_PATH))
        with open(DOCS_PATH, "wb") as f:
            pickle.dump(documents, f)

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    realtime_logger("🎉 Process Complete!")
    st.success("🎉 Training Completed Successfully!")
    st.balloons()

# 5. UI TABS & LAYOUT
st.title("VectorGraph-Engine")
st.caption("Recursive Scraper • OCR Support • Semantic Chunking")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏃 Pipeline",
    "🔎 Search",
    "📖 Inspector",
    "📉 Analytics",
    "📝 History Log",
    "🧬 KG Generator",
    "🌐 KG Visualizer"
])

# --- TAB 1: PIPELINE ---
with tab1:
    st.write("##### Control Panel")

    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("▶️ Resume Training", use_container_width=True, type="primary"):
            train_vectors(restart_fresh=False)

    with c2:
        if st.button("🔄 Start from Beginning", use_container_width=True):
            if st.checkbox("Confirm Reset? (This deletes old data)"):
                train_vectors(restart_fresh=True)

    with c3:
        if st.button("🛑 Stop / Pause", use_container_width=True, type="secondary"):
            st.warning("To STOP: Click the 'Stop' button in the top-right corner of the browser.")
            st.stop()

    st.info("💡 **Note:** The system auto-saves progress after **every chunk**.")

# --- TAB 2: SEARCH PLAYGROUND ---
with tab2:
    st.subheader("Test your Index")
    query = st.text_input("Enter a query to test retrieval:")

    if query and INDEX_PATH.exists() and DOCS_PATH.exists():
        index = faiss.read_index(str(INDEX_PATH))
        with open(DOCS_PATH, "rb") as f:
            docs = pickle.load(f)

        with st.spinner("Searching..."):
            q_vec = np.array(embedder.embed_query(query), dtype=np.float32).reshape(1, -1)
            distances, indices = index.search(q_vec, k=3)

        st.write("### Top Results")
        for i, idx in enumerate(indices[0]):
            if idx < len(docs) and idx != -1:
                doc = docs[idx]
                score = distances[0][i]
                with st.container():
                    st.markdown(f"**Result {i+1}** (Distance: {score:.4f})")
                    st.code(doc['text'], language="markdown")
                    st.caption(f"📂 Source: **{doc.get('source', 'N/A')}** | Type: **{doc.get('type', 'N/A')}**")
                    st.divider()
    elif query:
        st.warning("No index found. Please train the model first.")

# --- TAB 3: DATA INSPECTOR ---
with tab3:
    st.subheader("Browse Database Content")
    if DOCS_PATH.exists():
        with open(DOCS_PATH, "rb") as f:
            docs = pickle.load(f)
        df = pd.DataFrame(docs)

        c1, c2 = st.columns(2)
        with c1:
            all_sources = df['source'].unique() if 'source' in df.columns else []
            selected_source = st.selectbox("Filter by Source", ["All"] + list(all_sources))
        with c2:
            all_files = df['pdf'].unique()
            selected_file = st.selectbox("Filter by File", ["All"] + list(all_files))

        if selected_source != "All":
            df = df[df['source'] == selected_source]
        if selected_file != "All":
            df = df[df['pdf'] == selected_file]

        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "text": st.column_config.TextColumn("Content", width="large"),
                "source": st.column_config.TextColumn("Website", width="small"),
            }
        )
        st.metric("Total Chunks", len(df))
    else:
        st.info("No data available.")

# --- TAB 4: ADVANCED ANALYTICS ---
with tab4:
    st.header("🔬 Deep Vector Analysis")

    if not INDEX_PATH.exists() or not DOCS_PATH.exists():
        st.warning("⚠️ No vector index found.")
    else:
        index = faiss.read_index(str(INDEX_PATH))
        with open(DOCS_PATH, "rb") as f:
            docs = pickle.load(f)
        df = pd.DataFrame(docs)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📚 Total Chunks", index.ntotal)
        m2.metric("🌐 Sources", df['source'].nunique() if 'source' in df.columns else 0)
        m3.metric("🧠 Vector Dims", index.d)
        avg_len = int(df['text'].str.len().mean())
        m4.metric("📏 Avg Chunk Size", f"{avg_len} chars")

        st.divider()

        if st.button("Generate 2D Vector Map"):
            with st.spinner("Calculating PCA reduction..."):
                try:
                    total_vectors = index.ntotal
                    if hasattr(index, "reconstruct_n"):
                        vecs = index.reconstruct_n(0, total_vectors)
                        pca = PCA(n_components=2)
                        vecs_2d = pca.fit_transform(vecs)
                        df['x'] = vecs_2d[:, 0]
                        df['y'] = vecs_2d[:, 1]

                        color_col = 'source' if 'source' in df.columns else 'pdf'

                        fig = px.scatter(
                            df, x='x', y='y', color=color_col,
                            hover_data=['pdf', 'type'],
                            title="Knowledge Graph Clusters (By Source)",
                            template="plotly_dark", height=600
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Index type does not support direct reconstruction.")
                except Exception as e:
                    st.error(f"Visualization Error: {e}")

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Distribution by Source**")
            if 'source' in df.columns:
                st.bar_chart(df['source'].value_counts())
        with c2:
            st.write("**Scan vs Text Distribution**")
            if 'type' in df.columns:
                st.bar_chart(df['type'].value_counts())

# --- TAB 5: HISTORICAL LOGS ---
with tab5:
    st.header("📝 Full Log History")
    if st.button("Refresh Full Logs"):
        st.rerun()

    if LOG_PATH.exists():
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
            lines.reverse()
            st.code("".join(lines[:500]), language="log")
    else:
        st.info("No logs found yet.")

# 6. KNOWLEDGE GRAPH GENERATOR (TAB 6)
KG_JSON_PATH = VECTOR_DB_DIR / "kg.json"
KG_LOG_PATH = VECTOR_DB_DIR / "kg_log.txt"
KG_STATE_PATH = VECTOR_DB_DIR / "kg_state.json"
KG_MEMORY_PATH = VECTOR_DB_DIR / "kg_memory.json"

def load_memory():
    if KG_MEMORY_PATH.exists():
        try:
            with open(KG_MEMORY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return {"nodes": [], "edges": []}

def save_memory(memory_obj):
    with open(KG_MEMORY_PATH, "w", encoding="utf-8", buffering=1) as f:
        json.dump(memory_obj, f, indent=4)
        f.flush()

KG_MEMORY = load_memory()

def write_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(KG_LOG_PATH, "a", encoding="utf-8", buffering=1) as f:
        f.write(f"[{timestamp}] {message}\n")
        f.flush()

def read_state():
    if KG_STATE_PATH.exists():
        try:
            with open(KG_STATE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {"last_index": 0, "running": False}
    return {"last_index": 0, "running": False}

def save_state(last_index, running):
    with open(KG_STATE_PATH, "w", encoding="utf-8", buffering=1) as f:
        json.dump({"last_index": last_index, "running": running}, f)
        f.flush()

def extract_kg_from_chunk(chunk_text):
    prompt = f"""
    Extract ALL possible Knowledge Graph triples from the text.

    RULES:
    - Output MUST be a JSON list of objects:
      [
        {{"subject": "A", "predicate": "B", "object": "C"}}
      ]
    - NO text outside JSON.
    - NO duplicates.
    - Use consistent naming.
    - Link related concepts for connectivity.

    TEXT:
    {chunk_text}
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    chain = model.bind(generation_config={"response_mime_type": "application/json"})
    response = chain.invoke(prompt)

    try:
        triples = json.loads(response.content)
        return triples if isinstance(triples, list) else []
    except:
        return []

def append_kg_to_file(new_triples):
    existing = []
    if KG_JSON_PATH.exists():
        try:
            with open(KG_JSON_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except:
            existing = []

    for t in new_triples:
        s = t.get("subject")
        p = t.get("predicate")
        o = t.get("object")
        if not (s and p and o):
            continue

        if s not in KG_MEMORY["nodes"]:
            KG_MEMORY["nodes"].append(s)
        if o not in KG_MEMORY["nodes"]:
            KG_MEMORY["nodes"].append(o)

        edge_key = f"{s}|{p}|{o}"
        if edge_key not in KG_MEMORY["edges"]:
            KG_MEMORY["edges"].append(edge_key)
            existing.append(t)

    save_memory(KG_MEMORY)
    with open(KG_JSON_PATH, "w", encoding="utf-8", buffering=1) as f:
        json.dump(existing, f, indent=4, ensure_ascii=False)
        f.flush()

with tab6:
    st.header("🧬 Automated Knowledge Graph Generator")

    state = read_state()
    last_index = state["last_index"]
    running = state["running"]

    if not DOCS_PATH.exists():
        st.warning("No chunks found. Run the pipeline first.")
        st.stop()

    with open(DOCS_PATH, "rb") as f:
        docs = pickle.load(f)

    st.info(f"Total Chunks: **{len(docs)}**")
    st.info(f"Last Processed Chunk: **{last_index}**")

    col1, col2 = st.columns(2)
    start_clicked = col1.button("▶ START / RESUME", type="primary", use_container_width=True)
    stop_clicked = col2.button("⛔ STOP", type="secondary", use_container_width=True)

    if start_clicked:
        save_state(last_index, True)
        write_log("🚀 KG Processing Started.")
        running = True
        st.success("Processing started.")

    if stop_clicked:
        save_state(last_index, False)
        write_log("🛑 KG Processing Stopped by User.")
        running = False
        st.warning("Processing stopped.")

    st.subheader("📜 Real-Time Logs")
    log_box = st.empty()

    def update_logs():
        if KG_LOG_PATH.exists():
            with open(KG_LOG_PATH, "r") as f:
                log_box.text_area("Logs", f.read(), height=200)

    update_logs()

    if running:
        progress = st.progress(0)
        status = st.empty()

        for i in range(last_index, len(docs)):
            current_state = read_state()
            if not current_state["running"]:
                write_log("🛑 Processing Paused by User")
                st.warning("Processing stopped.")
                break

            save_state(i, True)
            chunk = docs[i].get("text", "")
            status.write(f"Processing chunk {i+1}/{len(docs)} ...")
            write_log(f"Processing chunk {i+1}")

            triples = extract_kg_from_chunk(chunk)
            append_kg_to_file(triples)
            write_log(f"Extracted {len(triples)} triples.")
            progress.progress((i + 1) / len(docs))

            update_logs()
            time.sleep(0.1)

        write_log("🎉 KG Extraction Completed Fully!")
        save_state(len(docs), False)
        st.success("🎉 KG Generation Complete!")
        st.balloons()


# --- TAB 7: KG VISUALIZATION ---
with tab7:
    st.header("📌 Knowledge Graph Visualization")

    st.write("Upload KG JSON, or load it from `vector_db/kg.json` using the button below.")

    kg_file = st.file_uploader("Upload KG JSON", type=["json"])

    kg_data = None

    load_btn = st.button("🟢 Load Knowledge Graph")
    refresh_btn = st.button("🔄 Refresh Graph")

    if "kg_data" not in st.session_state:
        st.session_state.kg_data = None

    if load_btn or refresh_btn:
        if kg_file is not None:
            try:
                kg_data = json.load(kg_file)
                st.session_state.kg_data = kg_data
                st.success("KG Loaded Successfully from Upload!")
            except Exception as e:
                st.error(f"Invalid JSON file: {e}")
                st.stop()
        else:
            internal_path = "vector_db/kg.json"
            if os.path.exists(internal_path):
                try:
                    with open(internal_path, "r", encoding="utf-8") as f:
                        kg_data = json.load(f)
                        st.session_state.kg_data = kg_data
                        st.success("KG Loaded Successfully from Internal File!")
                except Exception as e:
                    st.error(f"Error loading internal KG: {e}")
                    st.stop()
            else:
                st.warning("No file uploaded and internal KG file not found.")
                st.stop()
    else:
        kg_data = st.session_state.kg_data

    if kg_data is not None:
        st.subheader("🔍 Search Nodes or Edges")
        search_text = st.text_input("Enter any keyword to highlight (subject / object / predicate):")

        matched_nodes = set()
        matched_edges = []

        if search_text.strip() != "":
            for triple in kg_data:
                s = triple.get("subject", "")
                p = triple.get("predicate", "")
                o = triple.get("object", "")

                if search_text.lower() in s.lower():
                    matched_nodes.add(s)
                    matched_edges.append(triple)
                if search_text.lower() in o.lower():
                    matched_nodes.add(o)
                    matched_edges.append(triple)
                if search_text.lower() in p.lower():
                    matched_nodes.add(s)
                    matched_nodes.add(o)
                    matched_edges.append(triple)

        try:
            net = Network(height="600px", width="100%", directed=True)
            net.toggle_physics(True)

            for triple in kg_data:
                subject = triple.get("subject")
                predicate = triple.get("predicate")
                object_ = triple.get("object")

                node_color = "#97c2fc"

                if subject in matched_nodes or object_ in matched_nodes:
                    node_color = "#f5426c"

                if subject:
                    net.add_node(subject, label=subject, color=node_color)
                if object_:
                    net.add_node(object_, label=object_, color=node_color)

                edge_color = "gray"
                if triple in matched_edges:
                    edge_color = "red"

                if subject and object_:
                    net.add_edge(subject, object_, label=predicate, color=edge_color)

            with tempfile.TemporaryDirectory() as tmp:
                graph_path = os.path.join(tmp, "kg_graph.html")
                net.write_html(graph_path)

                with open(graph_path, "r", encoding="utf-8") as f:
                    html = f.read()

                st.components.v1.html(html, height=650, scrolling=True)

        except Exception as e:
            st.error(f"Error generating Knowledge Graph: {e}")
    else:
        st.info("Click 'Load Knowledge Graph' to display the graph.")