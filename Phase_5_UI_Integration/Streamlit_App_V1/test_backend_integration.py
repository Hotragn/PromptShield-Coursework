from types import SimpleNamespace
from unittest.mock import patch

from backend import generate_quiz_experiment, generate_variant_comparison


class DummyLLMResponse:
    def __init__(self, content):
        self.content = content


class DummyLLM:
    def __init__(self, responses):
        self.responses = responses
        self.index = 0

    def invoke(self, prompt):
        response = self.responses[self.index]
        self.index += 1
        return DummyLLMResponse(response)


def test_generate_quiz_experiment_v2_with_mocks():
    fake_docs = [SimpleNamespace(page_content="Prompt sensitivity and RAG.", metadata={"source": "slides.pdf"})]
    fake_json = '{"title":"Quiz","topic":"Prompt Engineering","questions":[{"id":1,"type":"mcq","question":"Q1","options":["A","B","C","D"],"correct_answer":"A","acceptable_answers":[],"explanation":"Because.","source_anchor":"Prompt sensitivity"}]}'

    with patch("backend.load_uploaded_pdfs", return_value=fake_docs), \
         patch("backend._build_context_and_sources", return_value=("Prompt sensitivity and RAG.", [{"source": "slides.pdf", "preview": "Prompt sensitivity and RAG.", "full_text": "Prompt sensitivity and RAG."}])), \
         patch("backend._get_llm", return_value=DummyLLM([fake_json])):
        result = generate_quiz_experiment(
            uploaded_files=["fake.pdf"],
            topic="Prompt Engineering",
            difficulty="Medium",
            question_type="Mixed",
            variant="V2",
            mode="rag",
        )

    assert result["variant"] == "V2"
    assert "quiz_data" in result
    assert len(result["quiz_data"]["questions"]) == 1


def test_generate_quiz_experiment_v3_returns_concepts():
    fake_docs = [SimpleNamespace(page_content="Decomposition and concept extraction.", metadata={"source": "slides.pdf"})]
    concept_json = '{"topic":"Prompt Engineering","concepts":[{"name":"Decomposition","why_it_matters":"Improves auditability.","source_anchor":"Decomposition"}]}'
    quiz_json = '{"title":"Quiz","topic":"Prompt Engineering","questions":[{"id":1,"type":"mcq","question":"Q1","options":["A","B","C","D"],"correct_answer":"A","acceptable_answers":[],"explanation":"Because.","source_anchor":"Decomposition"}]}'

    with patch("backend._build_context_and_sources", return_value=("Decomposition and concept extraction.", [{"source": "slides.pdf", "preview": "Decomposition and concept extraction."}])), \
         patch("backend._get_llm", return_value=DummyLLM([concept_json, quiz_json])):
        result = generate_quiz_experiment(
            uploaded_files=["fake.pdf"],
            topic="Prompt Engineering",
            difficulty="Medium",
            question_type="Mixed",
            variant="V3",
            mode="experimental",
        )

    assert result["variant"] == "V3"
    assert len(result["concepts"]) == 1


def test_generate_variant_comparison_returns_three_results():
    with patch("backend.generate_quiz_experiment") as mock_generate:
        mock_generate.side_effect = [
            {"variant": "V1", "evaluation": {}},
            {"variant": "V2", "evaluation": {}},
            {"variant": "V3", "evaluation": {}},
        ]
        results = generate_variant_comparison(
            uploaded_files=["fake.pdf"],
            topic="Prompt Engineering",
        )

    assert len(results) == 3
    assert results[0]["variant"] == "V1"
    assert results[1]["variant"] == "V2"
    assert results[2]["variant"] == "V3"

def test_generate_quiz_experiment_v2_mcq_format(monkeypatch):
    fake_json = """{
        "title": "Quiz",
        "topic": "Prompt Engineering",
        "questions": [
            {
                "id": 1,
                "type": "mcq",
                "question": "Q1",
                "options": ["A","B","C","D"],
                "correct_answer": "A",
                "acceptable_answers": [],
                "explanation": "Because.",
                "source_anchor": "Prompt sensitivity"
            }
        ]
    }"""

    with patch("backend._build_context_and_sources", return_value=(
        "Prompt sensitivity and RAG.",
        [{"source": "slides.pdf", "preview": "Prompt sensitivity", "full_text": "Prompt sensitivity and RAG."}]
    )), patch("backend._get_llm", return_value=DummyLLM([fake_json])):
        result = generate_quiz_experiment(
            uploaded_files=["fake.pdf"],
            topic="Prompt Engineering",
            difficulty="Medium",
            question_type="MCQ",
            variant="V2",
            mode="rag",
        )

    assert result["question_type"] == "MCQ"
    assert all(q["type"] == "mcq" for q in result["quiz_data"]["questions"])


def test_generate_quiz_experiment_v2_short_answer_format(monkeypatch):
    fake_json = """{
        "title": "Quiz",
        "topic": "Prompt Engineering",
        "questions": [
            {
                "id": 1,
                "type": "short_answer",
                "question": "Q1",
                "options": [],
                "correct_answer": "Answer 1",
                "acceptable_answers": ["Answer 1", "Alt 1"],
                "explanation": "Because.",
                "source_anchor": "Prompt sensitivity"
            }
        ]
    }"""

    with patch("backend._build_context_and_sources", return_value=(
        "Prompt sensitivity and RAG.",
        [{"source": "slides.pdf", "preview": "Prompt sensitivity", "full_text": "Prompt sensitivity and RAG."}]
    )), patch("backend._get_llm", return_value=DummyLLM([fake_json])):
        result = generate_quiz_experiment(
            uploaded_files=["fake.pdf"],
            topic="Prompt Engineering",
            difficulty="Medium",
            question_type="Short answer",
            variant="V2",
            mode="rag",
        )

    assert result["question_type"] == "Short answer"
    assert all(q["type"] == "short_answer" for q in result["quiz_data"]["questions"])


def test_generate_quiz_experiment_v3_receives_selected_format(monkeypatch):
    concept_json = """{
        "topic": "Prompt Engineering",
        "concepts": [
            {
                "name": "Decomposition",
                "why_it_matters": "Improves auditability.",
                "source_anchor": "Decomposition"
            }
        ]
    }"""

    quiz_json = """{
        "title": "Quiz",
        "topic": "Prompt Engineering",
        "questions": [
            {
                "id": 1,
                "type": "short_answer",
                "question": "Q1",
                "options": [],
                "correct_answer": "Answer 1",
                "acceptable_answers": ["Answer 1"],
                "explanation": "Because.",
                "source_anchor": "Decomposition"
            }
        ]
    }"""

    with patch("backend._build_context_and_sources", return_value=(
        "Decomposition and concept extraction.",
        [{"source": "slides.pdf", "preview": "Decomposition", "full_text": "Decomposition and concept extraction."}]
    )), patch("backend._get_llm", return_value=DummyLLM([concept_json, quiz_json])):
        result = generate_quiz_experiment(
            uploaded_files=["fake.pdf"],
            topic="Prompt Engineering",
            difficulty="Medium",
            question_type="Short answer",
            variant="V3",
            mode="experimental",
        )

    assert result["variant"] == "V3"
    assert all(q["type"] == "short_answer" for q in result["quiz_data"]["questions"])