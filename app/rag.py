"""Leaflets (public) and check-in notes (one patient each) in a vector store, behind one tool."""
from langchain.tools import ToolRuntime, tool
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.context import DATA, CareContext


def load_documents():
    docs = [Document(p.read_text(), metadata={"source": p.name, "scope": "public"})
            for p in sorted((DATA / "leaflets").glob("*.md"))]
    docs += [Document(p.read_text(), metadata={"source": p.name, "scope": "private", "patient_token": p.stem})
             for p in sorted((DATA / "notes").glob("*.md"))]
    return RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)


_vector_store = None


def vector_store():
    """Built on first use, so importing the app doesn't call the embeddings API."""
    global _vector_store
    if _vector_store is None:
        _vector_store = InMemoryVectorStore.from_documents(
            load_documents(), OpenAIEmbeddings(model="text-embedding-3-small"))
    return _vector_store


def allowed_for(patient_token):
    """Public leaflets + only this patient's notes. Built from context, never from the model."""
    return lambda doc: doc.metadata["scope"] == "public" or doc.metadata.get("patient_token") == patient_token


@tool(response_format="content_and_artifact")
def search_documents(query: str, runtime: ToolRuntime[CareContext]):
    """Search approved patient leaflets and this patient's own check-in notes."""
    found = vector_store().similarity_search(query, k=3, filter=allowed_for(runtime.context.patient_token))
    text = "\n\n".join(f"[{d.metadata['source']}] {d.page_content}" for d in found)
    return text or "Nothing found in the documents.", [d.metadata for d in found]
