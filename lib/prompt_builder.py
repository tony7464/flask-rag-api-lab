def format_context_chunk(chunk):
    """Format one retrieved chunk for the prompt context block."""
    return (
        f"[Source: {chunk.get('source_id') or 'Unknown'}]\n"
        f"Title: {chunk.get('title') or 'Unknown'}\n"
        f"Category: {chunk.get('category') or 'Unknown'}\n"
        f"Section: {chunk.get('section') or 'Unknown'}\n"
        f"Text: {(chunk.get('text') or '').strip()}"
    )


def build_rag_prompt(question, context_chunks):
    """Build a structured RAG prompt from a question and retrieved context."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    usable_chunks = [
        chunk
        for chunk in (context_chunks or [])
        if isinstance(chunk, dict)
        and isinstance(chunk.get("text"), str)
        and chunk["text"].strip()
    ]
    if not usable_chunks:
        raise ValueError("Context must include at least one chunk with text.")

    context_block = "\n\n".join(format_context_chunk(chunk) for chunk in usable_chunks)

    return (
        "Instructions:\n"
        "You are a customer success assistant helping support representatives. "
        "Use only the approved context below to answer the question. "
        "Do not invent policies, numbers, timelines, or other details that are "
        "not supported by the approved context.\n\n"
        "Context:\n"
        f"{context_block}\n\n"
        "Question:\n"
        f"{question.strip()}\n\n"
        "Response Requirements:\n"
        "- Answer concisely in plain language a support representative can use.\n"
        "- Base every statement on the approved context.\n"
        "- If the context is incomplete, clearly state what information is missing.\n"
        "- Reference source IDs (for example, [SUB-101]) when helpful.\n"
    )
