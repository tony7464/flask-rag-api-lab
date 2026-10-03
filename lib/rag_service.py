from lib.model_client import ModelClientError, generate_answer
from lib.prompt_builder import build_rag_prompt
from lib.response_formatter import (
    format_fallback_response,
    format_sources,
    format_success_response,
)
from lib.retrieval import DEFAULT_TOP_K, retrieve_context


def answer_question(
    question,
    retriever=None,
    prompt_builder=None,
    model_client=None,
    top_k=DEFAULT_TOP_K,
):
    """Run the full RAG workflow for a validated question.

    Flow: retrieve context chunks, return the fallback response if no usable
    context was found, build the prompt, call the model, and format the
    answer with its sources.

    retriever, prompt_builder, and model_client are injectable for tests and
    default to retrieve_context, build_rag_prompt, and generate_answer.
    """
    retriever = retriever or retrieve_context
    prompt_builder = prompt_builder or build_rag_prompt
    model_client = model_client or generate_answer

    # Retrieve candidate context for the question.
    context_chunks = retriever(question, top_k=top_k) or []

    # Keep only chunks with non-blank text.
    usable_chunks = [
        chunk
        for chunk in context_chunks
        if isinstance(chunk, dict)
        and isinstance(chunk.get("text"), str)
        and chunk["text"].strip()
    ]

    # Fall back without calling the model when there is no usable context.
    if not usable_chunks:
        return format_fallback_response(question)

    # Build the grounded prompt and ask the model.
    prompt = prompt_builder(question, usable_chunks)
    answer = model_client(prompt)

    if not isinstance(answer, str) or not answer.strip():
        raise ModelClientError("Model returned a blank or unusable answer.")

    # Format the answer with the sources that were used.
    return format_success_response(answer, format_sources(usable_chunks))
