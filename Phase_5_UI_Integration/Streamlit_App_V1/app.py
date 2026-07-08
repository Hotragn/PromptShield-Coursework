import json
import math
import re

import pandas as pd
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from backend import generate_quiz_experiment, generate_variant_comparison


st.set_page_config(
    page_title="QuizLab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


SEMANTIC_HIGH_THRESHOLD = 0.85
SEMANTIC_LOW_THRESHOLD = 0.45
JUDGE_MODEL = "gpt-4o-mini"


def init_state():
    defaults = {
        "result": None,
        "comparison_results": None,
        "submitted": False,
        "score": 0,
        "score_notified": False,
        "mode_used": "fine_tuned",
        "variant_used": "FT",
        "view_mode_used": "Test mode",
        "topic_used": "",
        "difficulty_used": "",
        "question_type_used": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def normalize_text(value):
    return re.sub(r"\s+", " ", str(value).strip().lower())


def cosine_similarity(vec1, vec2):
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot / (norm1 * norm2)


@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")


@st.cache_resource
def get_judge_llm():
    return ChatOpenAI(model=JUDGE_MODEL, temperature=0)


def semantic_similarity_score(answer, expected_answers):
    if not answer or not expected_answers:
        return 0.0

    embeddings = get_embeddings()
    answer_vec = embeddings.embed_query(answer)
    expected_vecs = embeddings.embed_documents(expected_answers)

    best_score = 0.0
    for expected_vec in expected_vecs:
        score = cosine_similarity(answer_vec, expected_vec)
        if score > best_score:
            best_score = score

    return best_score

def llm_judge_answer(question, answer, expected_answers):
    if not answer or not expected_answers:
        return False, "No answer or expected answers provided."

    judge = get_judge_llm()

    source_anchor = question.get("source_anchor", "")
    explanation = question.get("explanation", "")
    correct_answer = question.get("correct_answer", "")

    prompt = f"""
You are grading a student's short-answer response for a study quiz.

Your task is to decide whether the student's answer is semantically correct.

Judge the answer using:
1. the question
2. the expected answer(s)
3. the correct answer
4. the source anchor
5. the explanation

Important rules:
- Accept paraphrases and differently worded answers if they preserve the same meaning.
- Accept answers that are grounded in the same source idea even if they do not closely match the expected wording.
- Reject answers that are unrelated, vague, contradictory, or only partially correct.
- Be reasonably flexible for educational short-answer grading.

Return valid JSON only in this exact format:
{{
  "is_correct": true,
  "reason": "short explanation"
}}

Question:
{question.get("question", "")}

Correct Answer:
{correct_answer}

Acceptable Answers:
{json.dumps(expected_answers, ensure_ascii=False)}

Source Anchor:
{source_anchor}

Explanation:
{explanation}

Student Answer:
{answer}
""".strip()

    response = judge.invoke(prompt)
    content = (response.content or "").strip()

    try:
        if content.startswith("```"):
            lines = content.split("\n")
            lines = lines[1:]
            content = "\n".join(lines)
            if content.endswith("```"):
                content = content[:-3].strip()

        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if match:
            content = match.group(0)

        data = json.loads(content)
        return bool(data.get("is_correct", False)), data.get("reason", "LLM judge used.")
    except Exception:
        return False, "LLM judge response could not be parsed."


def get_answer_match_info(question, answer):
    """
    Returns:
    {
        "correct": bool,
        "method": "mcq" | "exact" | "semantic_high" | "llm_judge" | "semantic_low" | "none",
        "score": float,
        "reason": str
    }
    """
    if answer is None or str(answer).strip() == "":
        return {"correct": False, "method": "none", "score": 0.0, "reason": "No answer provided."}

    if question["type"] == "mcq":
        correct = answer == question["correct_answer"]
        return {
            "correct": correct,
            "method": "mcq",
            "score": 1.0 if correct else 0.0,
            "reason": "Exact option match." if correct else "Selected option does not match the correct answer.",
        }

    expected_answers = question.get("acceptable_answers") or [question.get("correct_answer", "")]
    normalized_answer = normalize_text(answer)

    for expected in expected_answers:
        if normalize_text(expected) == normalized_answer:
            return {
                "correct": True,
                "method": "exact",
                "score": 1.0,
                "reason": "Exact normalized match with an acceptable answer.",
            }

    similarity = semantic_similarity_score(answer, expected_answers)

    if similarity >= SEMANTIC_HIGH_THRESHOLD:
        return {
            "correct": True,
            "method": "semantic_high",
            "score": similarity,
            "reason": f"Accepted by semantic similarity (score: {similarity:.2f}).",
        }

    if similarity < SEMANTIC_LOW_THRESHOLD:
        return {
            "correct": False,
            "method": "semantic_low",
            "score": similarity,
            "reason": f"Semantic similarity too low (score: {similarity:.2f}).",
        }

    judge_correct, judge_reason = llm_judge_answer(question, answer, expected_answers)
    return {
        "correct": judge_correct,
        "method": "llm_judge",
        "score": similarity,
        "reason": f"Borderline semantic similarity ({similarity:.2f}); LLM judge decision: {judge_reason}",
    }


def is_correct_answer(question, answer):
    return get_answer_match_info(question, answer)["correct"]


def reset_quiz_state():
    st.session_state.result = None
    st.session_state.comparison_results = None
    st.session_state.submitted = False
    st.session_state.score = 0
    st.session_state.score_notified = False
    for key in list(st.session_state.keys()):
        if key.startswith("answer_"):
            del st.session_state[key]


def start_single_run(uploaded_files, topic, difficulty, question_type, mode, variant, temperature, paraphrase_style):
    result = generate_quiz_experiment(
        uploaded_files=uploaded_files,
        topic=topic,
        difficulty=difficulty,
        question_type=question_type,
        mode=mode,
        variant=variant,
        temperature=temperature,
        paraphrase_style=paraphrase_style,
    )

    st.session_state.result = result
    st.session_state.comparison_results = None
    st.session_state.submitted = False
    st.session_state.score = 0
    st.session_state.score_notified = False
    st.session_state.mode_used = result["mode"]
    st.session_state.variant_used = result["variant"]
    st.session_state.topic_used = topic
    st.session_state.difficulty_used = difficulty
    st.session_state.question_type_used = question_type

    for question in result["quiz_data"]["questions"]:
        st.session_state[f"answer_{question['id']}"] = ""


def start_comparison_run(uploaded_files, topic, difficulty, question_type, temperature, paraphrase_style):
    comparison_results = generate_variant_comparison(
        uploaded_files=uploaded_files,
        topic=topic,
        difficulty=difficulty,
        question_type=question_type,
        temperature=temperature,
        paraphrase_style=paraphrase_style,
    )
    st.session_state.result = None
    st.session_state.comparison_results = comparison_results
    st.session_state.submitted = False
    st.session_state.score = 0
    st.session_state.score_notified = False


def submit_quiz():
    if not st.session_state.result:
        return

    score = 0
    for idx, question in enumerate(st.session_state.result["quiz_data"]["questions"], start=1):
        qid = question.get("id", idx)
        answer = st.session_state.get(f"answer_{qid}", "")

        if is_correct_answer(question, answer):
            score += 1

    st.session_state.score = score
    st.session_state.submitted = True
    st.session_state.score_notified = False


def render_sources(sources):
    if not sources:
        st.caption("No sources available.")
        return

    for item in sources:
        with st.container(border=True):
            st.write(f"**{item['source']}**")
            st.caption(item["preview"])


def render_concepts(concepts):
    if not concepts:
        st.caption("No extracted concepts for this run.")
        return

    for idx, concept in enumerate(concepts, start=1):
        with st.container(border=True):
            st.write(f"**{idx}. {concept.get('name', 'Concept')}**")
            st.write(concept.get("why_it_matters", ""))
            if concept.get("source_anchor"):
                st.caption(f"Anchor: {concept['source_anchor']}")


def render_evaluation(evaluation):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Format", f"{evaluation['format_score']}/5")
    col2.metric("Groundedness", f"{evaluation['groundedness_score']}/5")
    col3.metric("Coverage", f"{evaluation['coverage_score']}/5")
    col4.metric("Consistency", f"{evaluation['consistency_score']}/5")
    for note in evaluation.get("notes", []):
        st.caption(f"• {note}")


init_state()

with st.sidebar:
    st.title("QuizLab")
    st.caption("AI Smart Learning Companion")

    uploaded_files = st.file_uploader(
        "Lecture slides",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF lecture files.",
    )

    topic = st.text_input(
        "Topic",
        placeholder="Prompt sensitivity, LangChain retrieval, decomposition...",
    )
    difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1)
    question_type = st.selectbox("Format", ["MCQ", "Short answer", "Mixed"], index=2)
    view_mode = st.radio("Quiz mode", ["Test mode", "Study mode"], index=0)
    st.session_state.view_mode_used = view_mode

    st.divider()
    st.subheader("Generation mode")
    app_mode = st.radio(
        "Choose mode",
        options=["Fine-tuned", "Grounded RAG", "Experimental"],
        index=0,
        help="Fine-tuned is the default product mode. Grounded RAG emphasizes retrieved context. Experimental exposes V1/V2/V3.",
    )

    with st.expander("Advanced options"):
        temperature = st.select_slider("Temperature", options=[0.2, 0.5, 0.8], value=0.2)
        paraphrase_style = st.selectbox(
            "Prompt wording style",
            ["Original", "Study guide style", "Exam prep style", "Concept focus style"],
            index=0,
        )
        comparison_mode = st.checkbox("Compare V1 / V2 / V3", value=False)
        experimental_variant = st.selectbox("Experimental variant", ["V1", "V2", "V3"], index=1)

    generate_clicked = st.button("Generate Quiz", type="primary", use_container_width=True)
    reset_clicked = st.button("Reset", use_container_width=True)

    if reset_clicked:
        reset_quiz_state()
        st.rerun()

    st.divider()
    st.subheader("Files")
    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.write(f"- {uploaded_file.name}")
    else:
        st.caption("No PDFs uploaded yet.")

if generate_clicked:
    if not uploaded_files:
        st.warning("Upload at least one PDF before generating a quiz.")
    elif not topic.strip():
        st.warning("Enter a topic before generating a quiz.")
    else:
        try:
            with st.spinner("Generating your quiz..."):
                if comparison_mode and app_mode == "Experimental":
                    start_comparison_run(
                        uploaded_files=uploaded_files,
                        topic=topic,
                        difficulty=difficulty,
                        question_type=question_type,
                        temperature=temperature,
                        paraphrase_style=paraphrase_style,
                    )
                else:
                    mode_map = {
                        "Fine-tuned": "fine_tuned",
                        "Grounded RAG": "rag",
                        "Experimental": "experimental",
                    }
                    variant = "V2"
                    if app_mode == "Experimental":
                        variant = experimental_variant

                    start_single_run(
                        uploaded_files=uploaded_files,
                        topic=topic,
                        difficulty=difficulty,
                        question_type=question_type,
                        mode=mode_map[app_mode],
                        variant=variant,
                        temperature=temperature,
                        paraphrase_style=paraphrase_style,
                    )
            st.rerun()
        except Exception as exc:
            st.error(f"Quiz generation failed: {exc}")

st.title("Quiz Workspace")
st.caption("Polished app flow by default, deeper prompt-engineering controls when needed.")

if st.session_state.comparison_results:
    st.subheader("Variant comparison")
    rows = []
    for item in st.session_state.comparison_results:
        ev = item["evaluation"]
        rows.append(
            {
                "Variant": item["variant"],
                "Format": ev["format_score"],
                "Groundedness": ev["groundedness_score"],
                "Coverage": ev["coverage_score"],
                "Consistency": ev["consistency_score"],
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    compare_tabs = st.tabs([item["variant"] for item in st.session_state.comparison_results])
    for tab, item in zip(compare_tabs, st.session_state.comparison_results):
        with tab:
            st.write(f"**Topic:** {item['topic']}")
            render_evaluation(item["evaluation"])
            if item["variant"] == "V3":
                st.markdown("### Extracted Concepts")
                render_concepts(item["concepts"])
            st.markdown("### Retrieved Sources")
            render_sources(item["sources"])

elif st.session_state.result:
    result = st.session_state.result
    quiz_data = result["quiz_data"]
    total_questions = len(quiz_data["questions"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mode", result["mode"].replace("_", " ").title())
    c2.metric("Variant", result["variant"])
    c3.metric("Questions", total_questions)
    c4.metric("Quiz View", st.session_state.view_mode_used)

    st.divider()

    left, right = st.columns([1.6, 1])

    with left:
        with st.container(border=True):
            st.subheader(quiz_data.get("title", "Generated Quiz"))
            st.write(f"**Topic:** {quiz_data.get('topic', result['topic'])}")
            st.write(f"**Difficulty:** {result['difficulty']}")
            st.write(f"**Format:** {result['question_type']}")

    with right:
        if st.session_state.submitted:
            with st.container(border=True):
                score = st.session_state.score
                percentage = int((score / total_questions) * 100) if total_questions else 0
                st.success(f"Score: {score}/{total_questions}")
                st.success(f"{percentage}% correct")
                if not st.session_state.score_notified:
                    st.toast(f"Quiz submitted. Score: {score}/{total_questions}")
                    st.session_state.score_notified = True
        elif st.session_state.view_mode_used != "Study mode":
            with st.container(border=True):
                st.info("Submit to see your score.")

    tab_labels = ["Quiz", "Sources", "Quality"]
    if result["variant"] == "V3":
        tab_labels.append("Concepts")
    tabs = st.tabs(tab_labels)
    quiz_tab, sources_tab, quality_tab = tabs[0], tabs[1], tabs[2]
    concepts_tab = tabs[3] if result["variant"] == "V3" else None

    with quiz_tab:
        is_study_mode = st.session_state.view_mode_used == "Study mode"

        def render_questions():
            for index, question in enumerate(quiz_data["questions"], start=1):
                st.markdown(f"### Question {question['id']}")
                st.caption(question["type"].replace("_", " ").title())
                st.write(question["question"])

                if question["type"] == "mcq":
                    options = question.get("options", [])
                    current_value = st.session_state.get(f"answer_{question['id']}", "")
                    default_index = options.index(current_value) if current_value in options else None
                    st.radio(
                        f"Select an answer for question {question['id']}",
                        options,
                        index=default_index,
                        key=f"answer_{question['id']}",
                        label_visibility="collapsed",
                        disabled=st.session_state.submitted,
                    )
                else:
                    st.text_input(
                        f"Type your answer for question {question['id']}",
                        key=f"answer_{question['id']}",
                        label_visibility="collapsed",
                        placeholder="Type your answer here",
                        disabled=is_study_mode or st.session_state.submitted,
                    )

                show_review = is_study_mode or st.session_state.submitted
                if show_review:
                    answer = st.session_state.get(f"answer_{question['id']}", "")
                    match_info = get_answer_match_info(question, answer)
                    correct = match_info["correct"]

                    if st.session_state.submitted:
                        if correct:
                            message = question["explanation"]
                            if question["type"] != "mcq":
                                message += f"\n\nScoring method: {match_info['method']}"
                                if match_info["method"] in {"semantic_high", "llm_judge"}:
                                    message += f"\nSimilarity score: {match_info['score']:.2f}"
                                if match_info["reason"]:
                                    message += f"\n{match_info['reason']}"
                            st.success(message, icon="✅")
                        else:
                            message = f"Correct answer: {question['correct_answer']}\n\n{question['explanation']}"
                            if question["type"] != "mcq":
                                message += f"\n\nScoring method: {match_info['method']}"
                                message += f"\nSimilarity score: {match_info['score']:.2f}"
                                if match_info["reason"]:
                                    message += f"\n{match_info['reason']}"
                            st.error(message, icon="❌")
                    else:
                        st.info(
                            f"Answer preview: {question['correct_answer']}\n\n{question['explanation']}",
                            icon="📘",
                        )

                    if question.get("source_anchor"):
                        st.caption(f"Anchor: {question['source_anchor']}")

                if index != total_questions:
                    st.divider()

        with st.container(border=True):
            if is_study_mode or st.session_state.submitted:
                render_questions()
            else:
                with st.form("quiz_form"):
                    render_questions()
                    st.form_submit_button(
                        "Submit Quiz",
                        type="primary",
                        use_container_width=True,
                        on_click=submit_quiz,
                    )

    with sources_tab:
        render_sources(result["sources"])

    with quality_tab:
        render_evaluation(result["evaluation"])

    if concepts_tab is not None:
        with concepts_tab:
            render_concepts(result["concepts"])

else:
    st.info("Upload your slides, choose a topic, and generate a quiz.")