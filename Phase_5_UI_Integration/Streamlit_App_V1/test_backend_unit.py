import pytest

from backend import (
    _extract_json,
    _coerce_quiz_structure,
    _evaluate_quiz,
    _normalize_question_type,
    _get_topic_prompt,
    _get_format_instructions,
     _evaluate_quiz
)


def test_extract_json_from_clean_response():
    content = '{"title":"Quiz","topic":"Test Topic","questions":[]}'
    result = _extract_json(content)
    assert result["title"] == "Quiz"
    assert result["topic"] == "Test Topic"


def test_extract_json_from_markdown_block():
    content = '```json\n{"title":"Quiz","topic":"Test Topic","questions":[]}\n```'
    result = _extract_json(content)
    assert result["title"] == "Quiz"


def test_extract_json_from_wrapped_text():
    content = 'Here is your quiz:\n{"title":"Quiz","topic":"Test Topic","questions":[]}\nThanks!'
    result = _extract_json(content)
    assert result["topic"] == "Test Topic"


def test_extract_json_raises_on_empty():
    with pytest.raises(ValueError):
        _extract_json("")


def test_extract_json_raises_on_invalid_text():
    with pytest.raises(ValueError):
        _extract_json("not valid json")


def test_normalize_question_type_mcq():
    assert _normalize_question_type("MCQ") == "MCQ"


def test_normalize_question_type_short_answer():
    assert _normalize_question_type("Short answer") == "Short answer"


def test_get_topic_prompt_original():
    assert _get_topic_prompt("Prompt Engineering", "Original") == "Prompt Engineering"


def test_get_topic_prompt_exam_prep():
    result = _get_topic_prompt("Prompt Engineering", "Exam prep style")
    assert "Prompt Engineering" in result
    assert "exam-prep" in result.lower()


def test_coerce_mcq_adds_missing_options():
    quiz = {
        "questions": [
            {
                "type": "mcq",
                "question": "What is RAG?",
                "options": ["A", "B"],
                "correct_answer": "A",
            }
        ]
    }
    result = _coerce_quiz_structure(quiz, "RAG")
    q = result["questions"][0]
    assert len(q["options"]) == 4
    assert q["correct_answer"] in q["options"]
    assert q["acceptable_answers"] == []


def test_coerce_short_answer_adds_acceptable_answers():
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
    assert q["options"] == []
    assert "Retrieval-Augmented Generation" in q["acceptable_answers"]


def test_coerce_invalid_type_defaults_to_mcq():
    quiz = {
        "questions": [
            {
                "type": "essay",
                "question": "Explain prompt engineering",
                "options": ["A"],
                "correct_answer": "A",
            }
        ]
    }
    result = _coerce_quiz_structure(quiz, "Prompt Engineering")
    assert result["questions"][0]["type"] == "mcq"


def test_evaluate_quiz_returns_metric_keys():
    quiz = {
        "questions": [
            {
                "type": "mcq",
                "question": "What is prompt sensitivity?",
                "options": ["A", "B", "C", "D"],
                "correct_answer": "A",
                "acceptable_answers": [],
                "explanation": "Because small wording changes affect outputs.",
                "source_anchor": "prompt sensitivity",
            }
        ]
    }
    sources = [
        {
            "source": "lecture.pdf",
            "preview": "This lecture explains prompt sensitivity and how wording changes affect outputs.",
        }
    ]
    result = _evaluate_quiz(quiz, sources, "V2")
    assert "format_score" in result
    assert "groundedness_score" in result
    assert "coverage_score" in result
    assert "consistency_score" in result
    assert "notes" in result


def test_evaluate_quiz_empty_quiz():
    result = _evaluate_quiz({"questions": []}, [], "V2")
    assert result["format_score"] == 0
    assert result["groundedness_score"] == 0


def test_evaluate_quiz_exact_anchor_match_improves_groundedness():
    quiz = {
        "questions": [
            {
                "type": "mcq",
                "question": "Q1",
                "options": ["A", "B", "C", "D"],
                "correct_answer": "A",
                "acceptable_answers": [],
                "explanation": "Explanation",
                "source_anchor": "prompt sensitivity",
            }
        ]
    }
    sources = [{"source": "slides.pdf", "preview": "The module covers prompt sensitivity in detail."}]
    result = _evaluate_quiz(quiz, sources, "V2")
    assert result["groundedness_score"] > 0


def test_evaluate_quiz_repeated_anchor_lowers_coverage():
    quiz = {
        "questions": [
            {
                "type": "mcq",
                "question": "Q1",
                "options": ["A", "B", "C", "D"],
                "correct_answer": "A",
                "acceptable_answers": [],
                "explanation": "Explanation",
                "source_anchor": "same anchor",
            },
            {
                "type": "mcq",
                "question": "Q2",
                "options": ["A", "B", "C", "D"],
                "correct_answer": "A",
                "acceptable_answers": [],
                "explanation": "Explanation",
                "source_anchor": "same anchor",
            },
        ]
    }
    sources = [{"source": "slides.pdf", "preview": "same anchor appears here"}]
    result = _evaluate_quiz(quiz, sources, "V2")
    assert result["coverage_score"] < 5


def test_evaluate_quiz_v1_consistency_lower_than_v2():
    quiz = {
        "questions": [
            {
                "type": "mcq",
                "question": "Q1",
                "options": ["A", "B", "C", "D"],
                "correct_answer": "A",
                "acceptable_answers": [],
                "explanation": "Explanation",
                "source_anchor": "anchor",
            }
        ]
    }
    sources = [{"source": "slides.pdf", "preview": "anchor"}]
    v1 = _evaluate_quiz(quiz, sources, "V1")
    v2 = _evaluate_quiz(quiz, sources, "V2")
    assert v1["consistency_score"] < v2["consistency_score"]

def test_get_format_instructions_mcq():
    result = _get_format_instructions("MCQ")
    assert "all 6 must be of type mcq" in result.lower()


def test_get_format_instructions_short_answer():
    result = _get_format_instructions("Short answer")
    assert "all 6 must be of type short_answer" in result.lower()


def test_get_format_instructions_mixed():
    result = _get_format_instructions("Mixed")
    assert "3 mcq, 2 short_answer, and 1 application" in result.lower()


def test_evaluate_quiz_uses_full_text_not_just_preview():
    quiz = {
        "questions": [
            {
                "type": "mcq",
                "question": "Q1",
                "options": ["A", "B", "C", "D"],
                "correct_answer": "A",
                "acceptable_answers": [],
                "explanation": "Because.",
                "source_anchor": "buried anchor phrase deep in the chunk",
            }
        ]
    }

    sources = [
        {
            "source": "slides.pdf",
            "preview": "This preview does not contain the anchor.",
            "full_text": "Some beginning text. More filler text. buried anchor phrase deep in the chunk appears here.",
        }
    ]

    result = _evaluate_quiz(quiz, sources, "V2")
    assert result["groundedness_score"] > 0


def test_evaluate_quiz_falls_back_to_preview_if_full_text_missing():
    quiz = {
        "questions": [
            {
                "type": "mcq",
                "question": "Q1",
                "options": ["A", "B", "C", "D"],
                "correct_answer": "A",
                "acceptable_answers": [],
                "explanation": "Because.",
                "source_anchor": "prompt sensitivity",
            }
        ]
    }

    sources = [
        {
            "source": "slides.pdf",
            "preview": "This preview contains prompt sensitivity in the text.",
        }
    ]

    result = _evaluate_quiz(quiz, sources, "V2")
    assert result["groundedness_score"] > 0