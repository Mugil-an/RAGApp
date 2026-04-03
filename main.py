import logging
import os
import datetime
import uuid

#inngest
import inngest
import inngest.fast_api
import google.generativeai as genai


from fastapi import FastAPI
from dotenv import load_dotenv
from data_loader import load_and_chunk_pdf,embed_texts
from vector_db import QdrantStorage
from custom_types import RAGChunkAndSrc,RAGQueryResult,RAGSearchResult,RAGUpsertResult

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

inngest_client = inngest.Inngest(
    app_id="rag-app",
    logger = logging.getLogger('uvicorn'),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)


@inngest_client.create_function(
    fn_id="RAG : Inngest PDF",
    trigger=inngest.TriggerEvent(event="rag/inngest_pdf"),
)
async def rag_inngest_pdf(ctx:inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        # Use 768 dimensions for Google AI Studio embeddings
        QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))


    chunks_and_src = await ctx.step.run("load-and-chunk",lambda : _load(ctx),output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert",lambda : _upsert(chunks_and_src),output_type=RAGUpsertResult)
    return ingested.model_dump()


@inngest_client.create_function(
        fn_id="RAG : Delete",
        trigger = inngest.TriggerEvent(event = '/rag/delete'),
)
async def rag_delete(ctx : inngest.Context):
    source_id = ctx.event.data["source_id"]
    QdrantStorage().delete_by_source(source_id=source_id)
    return {'deleted':True,"source_id":source_id}


@inngest_client.create_function(
    fn_id="RAG : Query",
    trigger=inngest.TriggerEvent(event="rag/query"),
)
async def rag_query_pdf_ai(ctx: inngest.Context) -> RAGQueryResult:
    # --- Step 1: Search for relevant context ---
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vec, top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    # --- Step 2: Generate an answer using the context ---
    def _generate_answer(question: str, contexts: list[str]) -> str:
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set.")

        context_block = "\n\n".join(f"- {c}" for c in contexts)
        prompt = (
            "Use the following context to answer the question. if the question not in the context then provide a message that this question is irrelevent to the document\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n"
            "Answer clearly and deeply using only the context above ."
        )

        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text.strip()


    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    # Run Step 1 durably
    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult)

    # Run Step 2 durably
    answer = await ctx.step.run("generate-answer", lambda: _generate_answer(question, found.contexts))

    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}


# --- FastAPI Server Setup ---
app = FastAPI()
inngest.fast_api.serve(app,inngest_client,functions=[rag_inngest_pdf, rag_query_pdf_ai,rag_delete])