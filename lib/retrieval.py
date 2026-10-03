CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "customer_success_knowledge"
DEFAULT_TOP_K = 3


class RetrievalError(Exception):
    """Raised when the vector store cannot be queried."""


def get_chroma_collection(path=CHROMA_PATH, collection_name=COLLECTION_NAME):
    """Return a persistent Chroma collection for manual local testing."""
    import chromadb

    client = chromadb.PersistentClient(path=path)
    return client.get_or_create_collection(collection_name)


def _first_batch(results, key):
    """Return the first query's list for a key in a Chroma result dict."""
    values = (results or {}).get(key) or []
    if values and isinstance(values[0], (list, tuple)):
        return list(values[0])
    return list(values)


def format_chroma_results(results):
    """Normalize Chroma query results into context chunk dictionaries.

    Chroma query results often look like:
        {
            "ids": [["chunk-1"]],
            "documents": [["Text"]],
            "metadatas": [[{"source_id": "SRC-1"}]],
            "distances": [[0.12]]
        }
    """
    # Unwrap the first query's batch from each nested result list.
    ids = _first_batch(results, "ids")
    documents = _first_batch(results, "documents")
    metadatas = _first_batch(results, "metadatas")
    distances = _first_batch(results, "distances")

    chunks = []
    for index, document in enumerate(documents):
        # Skip documents that are missing, non-string, or blank.
        if not isinstance(document, str) or not document.strip():
            continue

        # Tolerate missing metadata entries by falling back to an empty dict.
        metadata = metadatas[index] if index < len(metadatas) else None
        metadata = metadata or {}

        # Tolerate missing ids and distances by using None.
        chunks.append(
            {
                "id": ids[index] if index < len(ids) else None,
                "text": document.strip(),
                "source_id": metadata.get("source_id"),
                "title": metadata.get("title"),
                "category": metadata.get("category"),
                "section": metadata.get("section"),
                "distance": distances[index] if index < len(distances) else None,
            }
        )

    return chunks


def retrieve_context(question, collection=None, top_k=DEFAULT_TOP_K):
    """Retrieve context chunks for a user question.

    Tests may pass a fake collection. Manual use should call Chroma.

    Raises:
        ValueError: If top_k is not a positive integer.
        RetrievalError: If the collection cannot be opened or queried.
    """
    if not isinstance(question, str) or not question.strip():
        return []
    question = question.strip()

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    try:
        if collection is None:
            collection = get_chroma_collection()

        results = collection.query(
            query_texts=[question],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        raise RetrievalError(f"Context retrieval failed: {exc}") from exc

    return format_chroma_results(results)
