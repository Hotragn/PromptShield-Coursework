import pytest

from backend import _coerce_quiz_structure, _evaluate_quiz, _extract_json


def test_extract_json_from_clean_response():
    content = '{"title":"Quiz","topic":"Test Topic","questions":[]}'
    result = _extract_json(content)

    assert result["title"] == "Quiz"
    assert result["topic"] == "Test Topic"
    assert result["questions"] == []


def test_extract_json_from_markdown_block():
    content = '```json\n{"title":"Quiz","topic":"Test Topic","questions":[]}\n```'
    result = _extract_json(content)

    assert result["title"] == "Quiz"
    assert result["topic"] == "Test Topic"


def test_extract_json_raises_on_invalid_text():
    with pytest.raises(ValueError):
        _extract_json("This is not valid JSON")


def test_coerce_quiz_structure_fills_defaults_for_mcq():
    quiz = {
        "questions": [
            {
                "type": "mcq",
                "question": "What is prompt engineering?",
                "options": ["A", "B"],
                "correct_answer": "A",
            }
        ]
    }

    result = _coerce_quiz_structure(quiz, "Prompt Engineering")

    assert "title" in result
    assert result["topic"] == "Prompt Engineering"
    assert len(result["questions"]) == 1

    q = result["questions"][0]
    assert q["id"] == 1
    assert q["type"] == "mcq"
    assert len(q["options"]) == 4
    assert q["correct_answer"] in q["options"]
    assert q["acceptable_answers"] == []


def test_coerce_quiz_structure_fills_defaults_for_short_answer():
    quiz = {
        "questions": [
            {
                "type": "short_answer",
                "question": "Define RAG",
                "correct_answer": "Retrieval-Augmented Generation",
            }
        ]
    }

    result = _coerce_quiz_structure(quiz, "RAG")

    q = result["questions"][0]
    assert q["type"] == "short_answer"
    assert q["options"] == []
    assert "Retrieval-Augmented Generation" in q["acceptable_answers"]


def test_evaluate_quiz_returns_metric_keys():
    quiz = {
        "questions": [
            {
                "type": "mcq",
                "question": "What is prompt sensitivity?",
                "options": ["A", "B", "C", "D"],
                "correct_answer": "A",
                "acceptable_answers": [],
                "explanation": "Because the wording changes output behavior.",
                "source_anchor": "prompt sensitivity",
            }
        ]
    }

    sources = [
        {
            "source": "lecture.pdf",
            "preview": "This lecture explains prompt sensitivity and how small wording changes affect model outputs.",
        }
    ]

    result = _evaluate_quiz(quiz, sources, "V2")

    assert "format_score" in result
    assert "groundedness_score" in result
    assert "coverage_score" in result
    assert "consistency_score" in result
    assert "notes" in result


def test_evaluate_quiz_detects_empty_quiz():
    quiz = {"questions": []}
    sources = []

    result = _evaluate_quiz(quiz, sources, "V2")

    assert result["format_score"] == 0
    assert result["groundedness_score"] == 0
    assert result["coverage_score"] == 0
    assert result["consistency_score"] == 0