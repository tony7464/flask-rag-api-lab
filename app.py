from flask import Flask, jsonify, request

from lib.model_client import ModelClientError
from lib.rag_service import answer_question
from lib.response_formatter import format_error_response
from lib.retrieval import RetrievalError
from lib.validation import validate_question_payload


def create_app():
    app = Flask(__name__)

    @app.post("/api/ask")
    def ask():
        """Answer a customer success question using retrieved context.

        Returns 400 for an invalid payload, 502 when retrieval or the model
        service fails, and 200 with the answer and sources otherwise.
        """
        payload = request.get_json(silent=True)
        question, error = validate_question_payload(payload)
        if error:
            return jsonify(error), 400

        try:
            result = answer_question(question)
        except ModelClientError as exc:
            return jsonify(format_error_response("model_service_error", str(exc))), 502
        except RetrievalError as exc:
            return jsonify(format_error_response("retrieval_service_error", str(exc))), 502

        return jsonify(result), 200

    return app


app = create_app()
