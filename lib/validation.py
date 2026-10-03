MIN_QUESTION_LENGTH = 3


def _error(error, message):
    return None, {"error": error, "message": message}


def validate_question_payload(payload):
    """Validate the JSON body for POST /api/ask.

    Return:
        (question, None) when valid
        (None, error_dict) when invalid
    """
    if not isinstance(payload, dict):
        return _error("invalid_request", "Request body must be a JSON object.")

    if "question" not in payload:
        return _error("missing_question", "The 'question' field is required.")

    question = payload["question"]
    if not isinstance(question, str):
        return _error("invalid_question", "The 'question' field must be a string.")

    question = question.strip()
    if not question:
        return _error("empty_question", "The 'question' field must not be blank.")

    if len(question) < MIN_QUESTION_LENGTH:
        return _error(
            "short_question",
            f"The 'question' field must be at least {MIN_QUESTION_LENGTH} characters long.",
        )

    return question, None
