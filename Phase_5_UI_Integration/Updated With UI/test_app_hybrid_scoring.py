import pytest
import app


def make_short_question(correct="Easy Event Discovery", acceptable=None):
    return {
        "type": "short_answer",
        "question": "What is a key need for users when discovering tech events?",
        "correct_answer": correct,
        "acceptable_answers": acceptable or [correct],
        "explanation": "Users need a simple way to find tech events without searching across multiple platforms.",
        "source_anchor": "Users need a simple way to find tech events without searching across multiple platforms.",
    }


def make_application_question(
    correct="Provide personalized event recommendations that align with career goals."
):
    return {
        "type": "application",
        "question": "How would you improve event recommendations for early-career professionals?",
        "correct_answer": correct,
        "acceptable_answers": [correct],
        "explanation": "Recommendations should match career goals and professional growth needs.",
        "source_anchor": "focus on upskilling, networking, and events that align with their career path",
    }


def make_mcq_question():
    return {
        "type": "mcq",
        "question": "Which group is part of EVNTURE's target audience?",
        "options": ["Students", "Retirees", "Children", "Doctors"],
        "correct_answer": "Students",
        "acceptable_answers": [],
        "explanation": "EVNTURE is designed for students and early-career professionals.",
        "source_anchor": "students and early-career professionals",
    }


def test_normalize_text_collapses_spaces_and_case():
    assert app.normalize_text("  Easy   Event Discovery  ") == "easy event discovery"


def test_cosine_similarity_identity_is_one():
    assert app.cosine_similarity([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)


def test_cosine_similarity_zero_vector_safe():
    assert app.cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


def test_get_answer_match_info_mcq_correct():
    q = make_mcq_question()
    result = app.get_answer_match_info(q, "Students")
    assert result["correct"] is True
    assert result["method"] == "mcq"
    assert result["score"] == 1.0


def test_get_answer_match_info_mcq_incorrect():
    q = make_mcq_question()
    result = app.get_answer_match_info(q, "Doctors")
    assert result["correct"] is False
    assert result["method"] == "mcq"


def test_get_answer_match_info_empty_answer():
    q = make_short_question()
    result = app.get_answer_match_info(q, "")
    assert result["correct"] is False
    assert result["method"] == "none"


def test_short_answer_exact_match():
    q = make_short_question()
    result = app.get_answer_match_info(q, "Easy Event Discovery")
    assert result["correct"] is True
    assert result["method"] == "exact"


def test_short_answer_case_insensitive_exact_match():
    q = make_short_question()
    result = app.get_answer_match_info(q, "easy event discovery")
    assert result["correct"] is True
    assert result["method"] == "exact"


def test_short_answer_semantic_high(monkeypatch):
    q = make_short_question()
    monkeypatch.setattr(app, "semantic_similarity_score", lambda answer, expected: 0.91)
    result = app.get_answer_match_info(q, "Users want an easy way to find relevant events.")
    assert result["correct"] is True
    assert result["method"] == "semantic_high"
    assert result["score"] == pytest.approx(0.91)


def test_short_answer_semantic_low(monkeypatch):
    q = make_short_question()
    monkeypatch.setattr(app, "semantic_similarity_score", lambda answer, expected: 0.22)
    result = app.get_answer_match_info(q, "Users want more advertisements.")
    assert result["correct"] is False
    assert result["method"] == "semantic_low"
    assert result["score"] == pytest.approx(0.22)


def test_short_answer_borderline_calls_llm_judge_accept(monkeypatch):
    q = make_short_question()
    monkeypatch.setattr(app, "semantic_similarity_score", lambda answer, expected: 0.56)
    monkeypatch.setattr(app, "llm_judge_answer", lambda question, answer, expected: (True, "Grounded paraphrase."))
    result = app.get_answer_match_info(
        q,
        "Students need an easier way to discover relevant tech events without checking multiple platforms."
    )
    assert result["correct"] is True
    assert result["method"] == "llm_judge"
    assert "Grounded paraphrase" in result["reason"]


def test_short_answer_borderline_calls_llm_judge_reject(monkeypatch):
    q = make_short_question()
    monkeypatch.setattr(app, "semantic_similarity_score", lambda answer, expected: 0.58)
    monkeypatch.setattr(app, "llm_judge_answer", lambda question, answer, expected: (False, "Wrong concept."))
    result = app.get_answer_match_info(
        q,
        "The app should suggest events aligned with career goals and networking."
    )
    assert result["correct"] is False
    assert result["method"] == "llm_judge"
    assert "Wrong concept" in result["reason"]


def test_application_question_semantic_high(monkeypatch):
    q = make_application_question()
    monkeypatch.setattr(app, "semantic_similarity_score", lambda answer, expected: 0.88)
    result = app.get_answer_match_info(
        q,
        "The system should suggest events that match the user's career path and growth goals."
    )
    assert result["correct"] is True
    assert result["method"] == "semantic_high"


def test_application_question_llm_judge_path(monkeypatch):
    q = make_application_question()
    monkeypatch.setattr(app, "semantic_similarity_score", lambda answer, expected: 0.61)
    monkeypatch.setattr(app, "llm_judge_answer", lambda question, answer, expected: (True, "Equivalent application insight."))
    result = app.get_answer_match_info(
        q,
        "Recommend events based on what supports the user's professional direction."
    )
    assert result["correct"] is True
    assert result["method"] == "llm_judge"


def test_is_correct_answer_wrapper_uses_match_info(monkeypatch):
    q = make_short_question()
    monkeypatch.setattr(app, "get_answer_match_info", lambda question, answer: {
        "correct": True, "method": "semantic_high", "score": 0.9, "reason": "ok"
    })
    assert app.is_correct_answer(q, "paraphrase") is True


def test_submit_quiz_scores_mixed_questions(monkeypatch):
    q1 = make_mcq_question()
    q2 = make_short_question()
    app.st.session_state.result = {"quiz_data": {"questions": [q1, q2]}}
    app.st.session_state.answer_1 = "Students"
    app.st.session_state.answer_2 = "Easy Event Discovery"

    monkeypatch.setattr(app, "is_correct_answer", lambda q, ans: ans in ["Students", "Easy Event Discovery"])

    app.submit_quiz()
    assert app.st.session_state.score == 2
    assert app.st.session_state.submitted is True


def test_submit_quiz_partial_score(monkeypatch):
    q1 = make_mcq_question()
    q2 = make_short_question()
    app.st.session_state.result = {"quiz_data": {"questions": [q1, q2]}}
    app.st.session_state.answer_1 = "Doctors"
    app.st.session_state.answer_2 = "Easy Event Discovery"

    monkeypatch.setattr(app, "is_correct_answer", lambda q, ans: ans == "Easy Event Discovery")

    app.submit_quiz()
    assert app.st.session_state.score == 1
    assert app.st.session_state.submitted is True


def test_reset_quiz_state_clears_answers():
    app.st.session_state.result = {"quiz_data": {"questions": []}}
    app.st.session_state.comparison_results = [{"variant": "V1"}]
    app.st.session_state.submitted = True
    app.st.session_state.score = 3
    app.st.session_state.score_notified = True
    app.st.session_state.answer_1 = "A"
    app.st.session_state.answer_2 = "B"

    app.reset_quiz_state()

    assert app.st.session_state.result is None
    assert app.st.session_state.comparison_results is None
    assert app.st.session_state.submitted is False
    assert app.st.session_state.score == 0
    assert "answer_1" not in app.st.session_state
    assert "answer_2" not in app.st.session_state
