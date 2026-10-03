FALLBACK_MESSAGE = (
    "The approved customer success documents do not contain enough information "
    "to answer that question. Review the relevant policy or contact a supervisor "
    "before acting on this request."
)


def format_sources(context_chunks):
    """Return source metadata for retrieved context chunks."""
    sources = []
    seen = set()

    for chunk in context_chunks or []:
        source = {
            "id": chunk.get("source_id"),
            "title": chunk.get("title"),
            "category": chunk.get("category"),
            "section": chunk.get("section"),
            "chunk_id": chunk.get("id"),
        }
        key = (source["id"], source["chunk_id"])
        if key in seen:
            continue
        seen.add(key)
        sources.append(source)

    return sources


def format_success_response(answer, sources):
    """Return the successful RAG API response body."""
    return {
        "answer": (answer or "").strip(),
        "sources": list(sources or []),
    }


def format_fallback_response(question=None):
    """Return a safe response when there is not enough approved context."""
    return {"answer": FALLBACK_MESSAGE, "sources": []}


def format_error_response(error, message):
    """Return a standard error response body."""
    return {"error": error, "message": message}
