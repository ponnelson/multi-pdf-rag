import os
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

st.set_page_config(page_title="Multi-PDF RAG App", page_icon="📚", layout="wide")
st.title("Multi-PDF RAG Question Answering")
st.caption("Ask questions across multiple uploaded PDFs with file and page citations.")

api_key = os.getenv("OPENAI_API_KEY", "").strip()
if not api_key:
    st.error("`OPENAI_API_KEY` environment variable is not set. Set it before starting Streamlit.")
    st.stop()


def index_uploaded_pdfs(
    uploaded_files: list[Any],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Save uploaded PDFs temporarily, load pages with PyPDFLoader, split into chunks, and keep metadata."""
    safe_overlap = min(chunk_overlap, max(0, chunk_size - 1))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=safe_overlap,
        add_start_index=True,
    )
    all_chunks: list[Document] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for uploaded_file in uploaded_files:
            temp_file_path = Path(tmp_dir) / uploaded_file.name
            temp_file_path.write_bytes(uploaded_file.getvalue())

            pages = PyPDFLoader(str(temp_file_path)).load()
            for page_doc in pages:
                raw_page = page_doc.metadata.get("page", 0)
                page_num = int(raw_page) + 1 if isinstance(raw_page, (int, float)) else 1
                page_doc.metadata["source_file"] = uploaded_file.name
                page_doc.metadata["page_number"] = page_num

            file_chunks = splitter.split_documents(pages)
            all_chunks.extend(file_chunks)

    return all_chunks


def format_chunks_for_prompt(docs: list[Document]) -> str:
    """Attach explicit file and page labels to every retrieved chunk."""
    blocks = []
    for i, doc in enumerate(docs, start=1):
        fname = doc.metadata.get("source_file", "unknown.pdf")
        page = doc.metadata.get("page_number", "?")
        blocks.append(f"[Chunk {i} | File: {fname} | Page: {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def answer_question(
    store: InMemoryVectorStore,
    question: str,
    k: int = 5,
    model_name: str = "gpt-4o-mini",
) -> tuple[str, list[Document]]:
    """Retrieve matching chunks across all PDFs and generate a grounded answer."""
    retriever = store.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(question)

    llm = ChatOpenAI(model=model_name, temperature=0, api_key=api_key)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You answer questions strictly from the provided document chunks. "
            "If two or more documents or sections disagree, explicitly point out the conflict and attribute each position to its file and page. "
            "If the context does not contain the answer, state that clearly. "
            "Do not invent facts.",
        ),
        (
            "human",
            "Retrieved Chunks:\n{context}\n\nQuestion:\n{question}",
        ),
    ])

    chain = (
        {
            "context": lambda _: format_chunks_for_prompt(docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    answer = chain.invoke(question)
    return answer, docs


with st.sidebar:
    st.header("Index & Retrieval Settings")
    model_name = st.selectbox("OpenAI Model", ["gpt-4o-mini", "gpt-4o"], index=0)
    chunk_size = st.slider("Chunk Size", min_value=300, max_value=2000, value=1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=400, value=150, step=25)
    top_k = st.slider("Top-K Chunks", min_value=2, max_value=10, value=5, step=1)

uploaded_files = st.file_uploader(
    "Upload multiple PDFs",
    type=["pdf"],
    accept_multiple_files=True,
)

# Rebuild the single vector store whenever files or chunking settings change
upload_signature = (
    (
        tuple((f.name, f.size) for f in uploaded_files),
        chunk_size,
        chunk_overlap,
    )
    if uploaded_files
    else ()
)

if uploaded_files and st.session_state.get("upload_signature") != upload_signature:
    with st.spinner("Chunking and indexing PDFs into vector store..."):
        chunks = index_uploaded_pdfs(
            uploaded_files=uploaded_files,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
        st.session_state["store"] = InMemoryVectorStore.from_documents(chunks, embeddings)
        st.session_state["chunk_count"] = len(chunks)
        st.session_state["file_names"] = [f.name for f in uploaded_files]
        st.session_state["upload_signature"] = upload_signature

if "store" in st.session_state:
    st.caption(f"Uploaded: {', '.join(st.session_state['file_names'])}")
    st.success(f"Indexed {st.session_state['chunk_count']} chunks.")

question = st.text_input("Ask a question across your uploaded PDFs")

if question:
    if "store" not in st.session_state:
        st.warning("Upload at least one PDF first.")
    else:
        with st.spinner("Searching across all indexed documents..."):
            answer, matched_docs = answer_question(
                store=st.session_state["store"],
                question=question,
                k=top_k,
                model_name=model_name,
            )

        st.markdown(f"**A:** {answer}")

        seen: set[tuple[str, Any]] = set()
        source_labels: list[str] = []
        for doc in matched_docs:
            fname = doc.metadata.get("source_file", "unknown.pdf")
            page_num = doc.metadata.get("page_number", "?")
            key = (fname, page_num)
            if key not in seen:
                seen.add(key)
                source_labels.append(f"{fname} page {page_num}")

        st.markdown("**Sources:** " + " | ".join(source_labels) if source_labels else "**Sources:** None")

        with st.expander("Inspect retrieved chunks"):
            for doc in matched_docs:
                fname = doc.metadata.get("source_file", "unknown.pdf")
                page_num = doc.metadata.get("page_number", "?")
                st.caption(f"{fname} page {page_num}")
                st.write(doc.page_content)
                st.divider()