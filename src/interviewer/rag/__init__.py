from .chroma_store import ResumeRAG

_rag_singleton: ResumeRAG | None = None


def get_rag() -> ResumeRAG:
    global _rag_singleton
    if _rag_singleton is None:
        _rag_singleton = ResumeRAG()
    return _rag_singleton


def index_resume_to_chroma(user_id: str, resume_text: str):
    return get_rag().index_resume(user_id=user_id, resume_text=resume_text)


def retrieve_resume_context(user_id: str, query: str, k: int = 6):
    return get_rag().retrieve(user_id=user_id, query=query, k=k)
