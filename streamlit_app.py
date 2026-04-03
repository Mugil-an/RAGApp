import asyncio
from pathlib import Path
import time
import os
import requests

import streamlit as st
import inngest
from dotenv import load_dotenv

# --- Initial Setup ---
load_dotenv()
st.set_page_config(page_title="RAG App", page_icon="📄", layout="centered")

# --- Inngest and API Helper Functions ---

@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    """Returns a cached Inngest client instance."""
    return inngest.Inngest(app_id="rag_app", is_production=False)

def save_uploaded_pdf(file) -> Path:
    """Saves the uploaded PDF to a local 'uploads' directory."""
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_path.write_bytes(file.getbuffer())
    return file_path

async def send_rag_ingest_event(pdf_path: Path) -> None:
    """Sends an event to Inngest to trigger the PDF ingestion workflow."""
    client = get_inngest_client()
    await client.send(
        inngest.Event(
            name="rag/inngest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
            },
        )
    )

async def send_rag_delete_event(source_id : str) -> None:
    client = get_inngest_client()
    await client.send(
        inngest.Event(name = "rag/delete",data = {"source_id" : source_id}))

async def send_rag_query_event(question: str, top_k: int) -> str:
    """Sends a query event to Inngest and returns the event ID."""
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/query",  # Matches the function ID in main.py
            data={
                "question": question,
                "top_k": top_k,
            },
        )
    )
    return result[0]

def _inngest_api_base() -> str:
    """Returns the Inngest API base URL from environment variables."""
    return os.getenv("INNGEST_API_BASE_URL", "http://127.0.0.1:8288/v1")

def fetch_runs(event_id: str) -> list[dict]:
    """Fetches the runs for a given event ID from the Inngest API."""
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", [])
    except requests.RequestException as e:
        st.error(f"Failed to connect to Inngest API: {e}")
        return []

def wait_for_run_output(event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 0.5) -> dict:
    """Polls the Inngest API for the output of a function run."""
    start_time = time.time()
    last_status = "Pending"
    while time.time() - start_time < timeout_s:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status
            if status in ("Completed", "Succeeded"):
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Function run failed with status: {status}")
        time.sleep(poll_interval_s)
    raise TimeoutError(f"Timed out waiting for run output. Last known status: {last_status}")

def human_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.0f} {unit}" if unit == "B" else f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def get_all_uploaded_files() -> list[os.DirEntry]:
    uploads_dir = Path("uploads")
    if not uploads_dir.exists():
        return []
    return [entry for entry in os.scandir(uploads_dir) if entry.is_file()]
# --- Streamlit UI ---

st.title("📄 RAG Application with Inngest")

tab1, tab2 = st.tabs(["Upload Document", "Query Documents"])

# --- Upload Tab ---
with tab1:
    st.header("Ingest a PDF Document")
    st.markdown(
        "Upload a PDF document below. The system will process and index its content, "
        "making it available for querying."
    )
    uploaded_file = st.file_uploader(
        "Choose a PDF file", type=["pdf"], accept_multiple_files=False
    )

    if uploaded_file is not None:
        with st.spinner("Uploading and processing..."):
            path = save_uploaded_pdf(uploaded_file)
            asyncio.run(send_rag_ingest_event(path))
            time.sleep(0.5)
        st.success(f"✅ Successfully triggered ingestion for: `{path.name}`")
        st.info("Switch to the 'Query Documents' tab to ask questions.")

    st.header("Files Uploaded")
    files = sorted(
        get_all_uploaded_files(),
        key=lambda entry: entry.stat().st_mtime,
        reverse=True,
    )
    if not files:
        st.info("No uploads yet.")
    else:
        for entry in files:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"{entry.name} ({human_size(entry.stat().st_size)})")
            with col2:
                if st.button("Delete", key=f"del-{entry.name}"):
                    asyncio.run(send_rag_delete_event(entry.name))
                    Path(entry.path).unlink(missing_ok=True)
                    st.success(f"Deleted {entry.name}")
                    st.rerun()
# --- Query Tab ---
with tab2:
    st.header("Ask a Question")
    st.markdown(
        "Ask a question about the documents you've uploaded. The system will retrieve relevant context and generate an answer."
    )

    # Sidebar for configuration
    with st.sidebar:
        st.header("Query Configuration")
        top_k = st.slider(
            "Number of chunks to retrieve:",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            help="The number of relevant text chunks to use for generating the answer.",
        )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("View Sources"):
                    for source in message["sources"]:
                        st.info(source)

    # React to user input
    if prompt := st.chat_input("Ask your question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            try:
                event_id = asyncio.run(send_rag_query_event(prompt, int(top_k)))
                output = wait_for_run_output(event_id)

                answer = output.get("answer", "Sorry, I couldn't find an answer.")
                sources = output.get("sources", [])

                message_placeholder.markdown(answer)
                if sources:
                    with st.expander("View Sources"):
                        for source in sources:
                            st.info(source)
                
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )

            except Exception as e:
                error_message = f"An error occurred: {e}"
                message_placeholder.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )