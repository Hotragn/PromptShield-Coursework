"""
app.py — QuizLab Streamlit Frontend
Assignment 7: Prompt Hacking & Security (updated from A6)

New features vs A6:
  - Question count slider (3–10 questions)
  - Security scan on uploaded PDFs (injection detection + document type check)
  - Topic field sanitization before quiz generation
  - Topic relevance validation with inline warning
  - Security status panel visible in the sidebar
"""

import json
import math
import re

import pandas as pd
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from backend import (
    DEFAULT_NUM_QUESTIONS,
    generate_quiz_experiment,
    generate_variant_comparison,
    scan_documents_for_security,
    load_from_url,
)
from security import sanitize_topic


st.set_page_config(
    page_title="QuizLab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

SEMANTIC_HIGH_THRESHOLD = 0.85
SEMANTIC_LOW_THRESHOLD = 0.45
JUDGE_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

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
        "num_questions_used": DEFAULT_NUM_QUESTIONS,
        # Security state
        "doc_scan_result": None,
        "topic_validation": None,
        "security_warnings": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# Answer scoring (hybrid pipeline — unchanged from A6)
# ---------------------------------------------------------------------------

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
    return max(cosine_similarity(answer_vec, ev) for ev in expected_vecs)


def llm_judge_answer(question, answer, expected_answers):
    if not answer or not expected_answers:
        return False, "No answer or expected answers provided."

    judge = get_judge_llm()
    prompt = f"""
You are grading a student's short-answer response for a study quiz.

Judge the answer using:
1. the question
2. the expected answer(s)
3. the correct answer
4. the source anchor
5. the explanation

Important rules:
- Accept paraphrases that preserve the same meaning.
- Accept answers grounded in the same source idea even if worded differently.
- Reject answers that are unrelated, vague, contradictory, or only partially correct.

Return valid JSON only:
{{
  "is_correct": true,
  "reason": "short explanation"
}}

Question: {question.get("question", "")}
Correct Answer: {question.get("correct_answer", "")}
Acceptable Answers: {json.dumps(expected_answers, ensure_ascii=False)}
Source Anchor: {question.get("source_anchor", "")}
Explanation: {question.get("explanation", "")}
Student Answer: {answer}
""".strip()

    response = judge.invoke(prompt)
    content = (response.content or "").strip()
    try:
        if content.startswith("```"):
            content = "\n".join(content.split("\n")[1:])
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
    if answer is None or str(answer).strip() == "":
        return {"correct": False, "method": "none", "score": 0.0, "reason": "No answer provided."}

    if question["type"] == "mcq":
        correct = answer == question["correct_answer"]
        return {
            "correct": correct,
            "method": "mcq",
            "score": 1.0 if correct else 0.0,
            "reason": "Exact option match." if correct else "Selected option does not match.",
        }

    expected_answers = question.get("acceptable_answers") or [question.get("correct_answer", "")]
    normalized = normalize_text(answer)

    for expected in expected_answers:
        if normalize_text(expected) == normalized:
            return {"correct": True, "method": "exact", "score": 1.0,
                    "reason": "Exact normalized match with an acceptable answer."}

    similarity = semantic_similarity_score(answer, expected_answers)

    if similarity >= SEMANTIC_HIGH_THRESHOLD:
        return {"correct": True, "method": "semantic_high", "score": similarity,
                "reason": f"Accepted by semantic similarity (score: {similarity:.2f})."}

    if similarity < SEMANTIC_LOW_THRESHOLD:
        return {"correct": False, "method": "semantic_low", "score": similarity,
                "reason": f"Semantic similarity too low (score: {similarity:.2f})."}

    judge_correct, judge_reason = llm_judge_answer(question, answer, expected_answers)
    return {
        "correct": judge_correct,
        "method": "llm_judge",
        "score": similarity,
        "reason": f"Borderline similarity ({similarity:.2f}); LLM judge: {judge_reason}",
    }


def is_correct_answer(question, answer):
    return get_answer_match_info(question, answer)["correct"]


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def reset_quiz_state():
    st.session_state.result = None
    st.session_state.comparison_results = None
    st.session_state.submitted = False
    st.session_state.score = 0
    st.session_state.score_notified = False
    st.session_state.doc_scan_result = None
    st.session_state.topic_validation = None
    st.session_state.security_warnings = []
    for key in list(st.session_state.keys()):
        if key.startswith("answer_"):
            del st.session_state[key]


def start_single_run(
    uploaded_files, topic, difficulty, question_type, mode,
    variant, temperature, paraphrase_style, num_questions, source_url="",
):
    result = generate_quiz_experiment(
        uploaded_files=uploaded_files,
        topic=topic,
        difficulty=difficulty,
        question_type=question_type,
        mode=mode,
        variant=variant,
        temperature=temperature,
        paraphrase_style=paraphrase_style,
        num_questions=num_questions,
        run_topic_validation=True,
        source_url=source_url,
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
    st.session_state.num_questions_used = num_questions
    st.session_state.topic_validation = result.get("topic_validation", {})

    for question in result["quiz_data"]["questions"]:
        st.session_state[f"answer_{question['id']}"] = ""


def start_comparison_run(
    uploaded_files, topic, difficulty, question_type,
    temperature, paraphrase_style, num_questions, source_url="",
):
    comparison_results = generate_variant_comparison(
        uploaded_files=uploaded_files,
        topic=topic,
        difficulty=difficulty,
        question_type=question_type,
        temperature=temperature,
        paraphrase_style=paraphrase_style,
        num_questions=num_questions,
        source_url=source_url,
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


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def render_sources(sources):
    if not sources:
        st.caption("No sources available.")
        return
    for i, item in enumerate(sources, start=1):
        src = item["source"]
        # Show just the filename, not the full path/URL
        label = src.split("/")[-1] if "/" in src else src
        snippet = item["preview"][:120].replace("\n", " ").strip()
        if len(item["preview"]) > 120:
            snippet += "…"
        with st.expander(f"Source {i} — {label}"):
            st.caption(snippet)


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


def render_security_status(doc_scan_result, topic_validation):
    """Show a user-friendly content check summary."""
    if not doc_scan_result and not topic_validation:
        st.caption("No scan results yet. Generate a quiz to see the content check.")
        return

    if doc_scan_result:
        inj = doc_scan_result.get("injection_scan", {})
        doc_cls = doc_scan_result.get("document_classification", {})
        doc_type = doc_cls.get("document_type", "unknown")

        type_label = {
            "slides": "lecture slides / PDF",
            "website": "webpage printout",
            "other": "article or mixed content",
            "unknown": "document (very short — check that it has readable text)",
        }.get(doc_type, doc_type)

        if inj.get("risk_level") == "high":
            st.error(
                "Your document was blocked. It contains instructions that could "
                "interfere with quiz generation. Please upload your actual lecture material."
            )
        elif inj.get("detected"):
            st.warning(
                f"Your document ({type_label}) was scanned and some unusual content was detected. "
                "The quiz was generated using only the lecture material portions."
            )
        else:
            st.success(
                f"Your document looks good. It was identified as {type_label} "
                "and no problematic content was found."
            )

        if doc_cls.get("warning") and doc_type in ("website", "other"):
            st.info(
                "For the best quiz quality, upload lecture slides or a textbook PDF. "
                "Webpages and articles may include navigation text or ads that reduce accuracy."
            )

    if topic_validation:
        coverage = topic_validation.get("coverage_estimate", 0)
        reason = topic_validation.get("reason", "")
        if coverage >= 50:
            st.success(
                f"Your topic appears to be well covered in the uploaded material "
                f"(estimated {coverage}% coverage). Questions should be well grounded."
            )
        elif coverage >= 20:
            st.warning(
                f"Your topic has partial coverage in the material (estimated {coverage}%). "
                f"{reason} Some questions may be less specific to your slides."
            )
        else:
            st.warning(
                f"Your topic may not be well covered in this material (estimated {coverage}%). "
                f"{reason} Consider uploading slides that focus on this topic."
            )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

init_state()

with st.sidebar:
    st.title("QuizLab")
    st.caption("AI Smart Learning Companion")

    uploaded_files = st.file_uploader(
        "Lecture slides (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF lecture files.",
    )

    source_url = st.text_input(
        "Or paste a webpage URL",
        placeholder="https://...",
        help="Paste a public webpage URL instead of (or alongside) a PDF.",
    ).strip()

    topic = st.text_input(
        "Topic",
        placeholder="Prompt sensitivity, LangChain retrieval, decomposition...",
    )
    difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1)
    question_type = st.selectbox("Format", ["MCQ", "Short answer", "Mixed"], index=2)

    # ---- Question count (A7 addition) ------------------------------------
    num_questions = st.slider(
        "Number of questions",
        min_value=3,
        max_value=10,
        value=DEFAULT_NUM_QUESTIONS,
        step=1,
        help="Choose how many questions to generate (3–10).",
    )

    view_mode = st.radio("Quiz mode", ["Test mode", "Study mode"], index=0)
    st.session_state.view_mode_used = view_mode

    st.divider()
    st.subheader("Generation mode")
    app_mode = st.radio(
        "Choose mode",
        options=["Fine-tuned", "Grounded RAG", "Experimental"],
        index=0,
        help=(
            "Fine-tuned is the default product mode. "
            "Grounded RAG emphasizes retrieved context. "
            "Experimental exposes V1/V2/V3."
        ),
    )

    with st.expander("Advanced options"):
        temperature = st.select_slider(
            "Temperature", options=[0.2, 0.5, 0.8], value=0.2
        )
        paraphrase_style = st.selectbox(
            "Prompt wording style",
            ["Original", "Study guide style", "Exam prep style", "Concept focus style"],
            index=0,
        )
        comparison_mode = st.checkbox("Compare V1 / V2 / V3", value=False)
        experimental_variant = st.selectbox(
            "Experimental variant", ["V1", "V2", "V3"], index=1
        )

    generate_clicked = st.button("Generate Quiz", type="primary", use_container_width=True)
    reset_clicked = st.button("Reset", use_container_width=True)

    if reset_clicked:
        reset_quiz_state()
        st.rerun()

    st.divider()
    st.subheader("Sources")
    if uploaded_files:
        for f in uploaded_files:
            st.write(f"- {f.name}")
    if source_url:
        st.write(f"- {source_url}")
    if not uploaded_files and not source_url:
        st.caption("No PDFs uploaded and no URL entered yet.")

# ---------------------------------------------------------------------------
# Generate button logic (includes security checks)
# ---------------------------------------------------------------------------

if generate_clicked:
    if not uploaded_files and not source_url:
        st.warning("Upload at least one PDF or paste a webpage URL before generating a quiz.")
    elif not topic.strip():
        st.warning("Enter a topic before generating a quiz.")
    else:
        # Step 1 — sanitize topic field
        topic_check = sanitize_topic(topic)
        if not topic_check["safe"]:
            st.error(f"Topic blocked: {topic_check['warning']}")
        else:
            clean_topic = topic_check["sanitized"]
            if topic_check["warning"]:
                st.warning(f"Topic advisory: {topic_check['warning']}")

            # Step 2 — scan uploaded documents / URL
            with st.spinner("Scanning documents..."):
                doc_scan = scan_documents_for_security(uploaded_files, source_url)
            st.session_state.doc_scan_result = doc_scan

            inj = doc_scan.get("injection_scan", {})
            doc_cls = doc_scan.get("document_classification", {})

            # Block on high-risk document injection
            if inj.get("risk_level") == "high":
                st.error(
                    f"Document blocked: {inj.get('message', 'High-risk content detected.')} "
                    "Please upload a genuine lecture slides PDF."
                )
            else:
                # Show non-blocking warnings
                if inj.get("detected"):
                    st.warning(f"Document advisory: {inj.get('message', '')}")
                if doc_cls.get("warning"):
                    st.warning(f"Document type: {doc_cls.get('warning', '')}")

                # Step 3 — generate quiz
                try:
                    with st.spinner("Generating your quiz..."):
                        if comparison_mode and app_mode == "Experimental":
                            start_comparison_run(
                                uploaded_files=uploaded_files,
                                topic=clean_topic,
                                difficulty=difficulty,
                                question_type=question_type,
                                temperature=temperature,
                                paraphrase_style=paraphrase_style,
                                num_questions=num_questions,
                                source_url=source_url,
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
                                topic=clean_topic,
                                difficulty=difficulty,
                                question_type=question_type,
                                mode=mode_map[app_mode],
                                variant=variant,
                                temperature=temperature,
                                paraphrase_style=paraphrase_style,
                                num_questions=num_questions,
                                source_url=source_url,
                            )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Quiz generation failed: {exc}")

# ---------------------------------------------------------------------------
# Main workspace
# ---------------------------------------------------------------------------

st.title("Quiz Workspace")
st.caption("Polished app flow by default, deeper prompt-engineering controls when needed.")

# Topic validation banner (shown after generation)
if st.session_state.topic_validation:
    tv = st.session_state.topic_validation
    if tv.get("warning"):
        st.warning(tv["warning"])

if st.session_state.comparison_results:
    st.subheader("Variant comparison")
    rows = []
    for item in st.session_state.comparison_results:
        ev = item["evaluation"]
        rows.append({
            "Variant": item["variant"],
            "Questions": item.get("num_questions", DEFAULT_NUM_QUESTIONS),
            "Format": ev["format_score"],
            "Groundedness": ev["groundedness_score"],
            "Coverage": ev["coverage_score"],
            "Consistency": ev["consistency_score"],
        })
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

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Mode", result["mode"].replace("_", " ").title())
    c2.metric("Variant", result["variant"])
    c3.metric("Questions", total_questions)
    c4.metric("Difficulty", result["difficulty"])
    c5.metric("Quiz View", st.session_state.view_mode_used)

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
                pct = int((score / total_questions) * 100) if total_questions else 0
                st.success(f"Score: {score}/{total_questions}")
                st.success(f"{pct}% correct")
                if not st.session_state.score_notified:
                    st.toast(f"Quiz submitted. Score: {score}/{total_questions}")
                    st.session_state.score_notified = True
        elif st.session_state.view_mode_used != "Study mode":
            with st.container(border=True):
                st.info("Submit to see your score.")

    tab_labels = ["Quiz", "Sources", "Quality", "Security"]
    if result["variant"] == "V3":
        tab_labels.append("Concepts")
    tabs = st.tabs(tab_labels)
    quiz_tab = tabs[0]
    sources_tab = tabs[1]
    quality_tab = tabs[2]
    security_tab = tabs[3]
    concepts_tab = tabs[4] if result["variant"] == "V3" else None

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
                            msg = question["explanation"]
                            if question["type"] != "mcq":
                                msg += f"\n\nScoring method: {match_info['method']}"
                                if match_info["method"] in {"semantic_high", "llm_judge"}:
                                    msg += f"\nSimilarity score: {match_info['score']:.2f}"
                                if match_info["reason"]:
                                    msg += f"\n{match_info['reason']}"
                            st.success(msg, icon="✅")
                        else:
                            msg = f"Correct answer: {question['correct_answer']}\n\n{question['explanation']}"
                            if question["type"] != "mcq":
                                msg += f"\n\nScoring method: {match_info['method']}"
                                msg += f"\nSimilarity score: {match_info['score']:.2f}"
                                if match_info["reason"]:
                                    msg += f"\n{match_info['reason']}"
                            st.error(msg, icon="❌")
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

    with security_tab:
        st.markdown("### Content Check")
        render_security_status(
            st.session_state.get("doc_scan_result"),
            st.session_state.get("topic_validation"),
        )

    if concepts_tab is not None:
        with concepts_tab:
            render_concepts(result["concepts"])

else:
    st.info("Upload your slides, choose a topic, and generate a quiz.")
