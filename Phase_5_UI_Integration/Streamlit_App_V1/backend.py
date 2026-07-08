import json
import os
import re
import tempfile
import warnings
from typing import Any, Dict, List, Optional

warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

MODEL_NAME = "gpt-4o-mini"
# Update this in your .env or directly here if your fine-tuned model ID changes.
FINE_TUNED_MODEL_NAME = os.getenv(
    "OPENAI_FINE_TUNED_MODEL",
    "ft:gpt-4.1-mini-2025-04-14:personal::DJoPwRqY",
)

DEFAULT_TEMPERATURE = 0.2
EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_K = 3


def load_uploaded_pdfs(uploaded_files) -> List[Document]:
    docs: List[Document] = []

    for uploaded_file in uploaded_files:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name

            reader = PdfReader(temp_path)
            pages = []
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(text)

            text = "\n".join(pages)
            if text.strip():
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": uploaded_file.name},
                    )
                )

            os.remove(temp_path)
        except Exception as exc:
            print(f"Could not read {uploaded_file.name}: {exc}")

    return docs


def build_retriever_from_docs(docs: List[Document], k: int = DEFAULT_K):
    if not docs:
        raise ValueError("No valid PDF documents were uploaded.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": k})


QUIZ_SCHEMA = """
Return valid JSON only. Do not wrap the JSON in markdown fences.
Use this exact JSON schema:
{{
  "title": "string",
  "topic": "string",
  "questions": [
    {{
      "id": 1,
      "type": "mcq or short_answer or application",
      "question": "string",
      "options": ["string", "string", "string", "string"],
      "correct_answer": "string",
      "acceptable_answers": ["string"],
      "explanation": "string",
      "source_anchor": "short phrase from the lecture notes"
    }}
  ]
}}

Rules by question type:
- For "mcq": provide exactly 4 options, make "correct_answer" one of the options, and set "acceptable_answers" to [].
- For "short_answer": set "options" to [], make "correct_answer" the primary answer, and include 2 to 4 concise acceptable answers.
- For "application": set "options" to [], make the answer directly grounded in the notes, and include 1 to 3 acceptable grounded answers.
- Every explanation must briefly justify the answer using only the provided lecture context.
- Every source_anchor must be copied as closely as possible from the provided context. Do not paraphrase it.
"""

SYSTEM_V1 = f"""
You are a tutor. Make a quiz based on the notes.
Use only the notes provided.
Return valid JSON only.
Output 6 questions total.
{QUIZ_SCHEMA}
"""

SYSTEM_V2 = """
You are an AI study assistant acting like a graduate teaching assistant.

Your task is to generate a grounded quiz only from the uploaded lecture material.

Instructions:
- Use only the uploaded lecture material as the source of truth.
- Do not invent facts, examples, definitions, or terminology not present in the lecture notes.
- Every question must be directly traceable to the provided notes.
- Keep the quiz clear, academic, and well-structured.
- If the notes do not contain enough information, say so honestly in the explanation instead of hallucinating.
- Difficulty definitions:
  - Easy = direct recall of definitions, components, or concepts.
  - Medium = understanding or applying an idea already explained in the notes.
  - Hard = analysis, comparison, or synthesis strictly grounded in the notes.
- {format_instructions}
- Keep wording concise and student-friendly.


""" + QUIZ_SCHEMA

SYSTEM_V3_EXTRACT = """
You are an academic assistant.

Using only the lecture context, extract 5 to 7 key concepts most important for generating a graduate-level quiz on the requested topic.

Return valid JSON only in this exact format:
{{
  "topic": "string",
  "concepts": [
    {{
      "name": "string",
      "why_it_matters": "string",
      "source_anchor": "short phrase from the lecture notes"
    }}
  ]
}}

Rules:
- Use only the provided notes.
- Do not add outside knowledge.
- Keep each why_it_matters to 1 sentence.
- Keep source_anchor short and clearly traceable to the notes.
"""

SYSTEM_V3_GENERATE = """
You are an AI study assistant acting like a graduate teaching assistant.

Generate a grounded quiz using the provided lecture context and the extracted concepts.

Instructions:
- Use only the lecture material and concept list.
- Every question must be directly traceable to the notes and anchored to one of the extracted concepts.
- Do not invent facts, examples, definitions, or terminology not present in the notes.
- {format_instructions}
- The questions should collectively cover the most important concepts rather than repeating the same idea.
- Keep explanations short, accurate, and grounded.


""" + QUIZ_SCHEMA

FINE_TUNED_SYSTEM = """
You are the fine-tuned quiz generation model for the AI Smart Learning Companion.

Generate a clear, graduate-level quiz from the supplied study material.

Instructions:
- Follow the requested topic, difficulty, and format preference.
- Keep the output structured, concise, and student-friendly.
- {format_instructions}
- Return valid JSON only.
- When context is provided, stay grounded in it.
- When context is not provided, generate the best quiz you can from the supplied notes and topic.

""" + QUIZ_SCHEMA


PROMPT_V1 = PromptTemplate(
    input_variables=["context", "topic", "difficulty", "question_type"],
    template=SYSTEM_V1 + """

Lecture Notes:
{context}

Generate a {difficulty} level quiz in {question_type} format about:
{topic}
""",
)

PROMPT_V2 = PromptTemplate(
    input_variables=["context", "topic", "difficulty", "question_type", "format_instructions"],
    template=SYSTEM_V2 + """

Lecture Notes:
{context}

Requested topic:
{topic}
Requested emphasis:
Difficulty: {difficulty}
Format preference from UI: {question_type}
""",
)

PROMPT_V3_GENERATE = PromptTemplate(
    input_variables=["context", "topic", "concepts", "format_instructions"],
    template=SYSTEM_V3_GENERATE + """

Lecture Notes:
{context}

Requested topic:
{topic}

Extracted Concepts:
{concepts}
""",
)

PROMPT_FINE_TUNED = PromptTemplate(
    input_variables=["notes", "topic", "difficulty", "question_type", "format_instructions"],
    template=FINE_TUNED_SYSTEM + """

Study Material:
{notes}

Requested topic:
{topic}

Difficulty:
{difficulty}

Format preference from UI:
{question_type}
""",
)

PARAPHRASE_MAP = {
    "Original": "{topic}",
    "Study guide style": "Create a quiz that helps a graduate student study the topic: {topic}",
    "Exam prep style": "Generate an exam-prep quiz focused on this topic from the uploaded notes: {topic}",
    "Concept focus style": "Make a quiz that tests understanding of the main concepts related to: {topic}",
}


def _extract_json(content: str) -> Dict[str, Any]:
    cleaned = (content or "").strip()
    if not cleaned:
        raise ValueError("The model returned an empty response.")

    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines:
            lines = lines[1:]
        cleaned = "\n".join(lines)
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError(
        "The model returned text instead of valid JSON. Use Fine-tuned or V2 first, keep temperature at 0.2, "
        "and only enable comparison after the base flow works."
    )


def _normalize_question_type(question_type: str) -> str:
    mapping = {"MCQ": "MCQ", "Short answer": "Short answer", "Mixed": "Mixed"}
    return mapping.get(question_type, question_type)

def _get_format_instructions(question_type: str) -> str:
    qtype = (question_type or "").strip().lower()

    if qtype == "mcq":
        return "Output exactly 6 questions, and all 6 must be of type mcq."
    elif qtype == "short answer":
        return "Output exactly 6 questions, and all 6 must be of type short_answer."
    elif qtype == "mixed":
        return "Output exactly 6 questions: 3 mcq, 2 short_answer, and 1 application."
    else:
        return "Output exactly 6 questions: 3 mcq, 2 short_answer, and 1 application."


def _get_topic_prompt(topic: str, paraphrase_style: str) -> str:
    template = PARAPHRASE_MAP.get(paraphrase_style, PARAPHRASE_MAP["Original"])
    return template.format(topic=topic)


def _docs_to_text(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def _build_context_and_sources(uploaded_files, topic: str, retrieval_k: int = DEFAULT_K):
    docs = load_uploaded_pdfs(uploaded_files)
    retriever = build_retriever_from_docs(docs, k=retrieval_k)
    retrieved_docs = retriever.invoke(topic)

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    sources = []
    for doc in retrieved_docs:
        preview = doc.page_content[:280]
        if len(doc.page_content) > 280:
            preview += "..."
        sources.append(
        {
            "source": doc.metadata.get("source", "Unknown Source"),
            "preview": preview,
            "full_text": doc.page_content,
        }
    )

    unique_sources = []
    seen = set()
    for item in sources:
        key = (item["source"], item["preview"])
        if key not in seen:
            unique_sources.append(item)
            seen.add(key)

    return context, unique_sources


def _build_full_notes(uploaded_files):
    docs = load_uploaded_pdfs(uploaded_files)
    if not docs:
        raise ValueError("No valid PDF documents were uploaded.")

    full_notes = _docs_to_text(docs)
    sources = []
    for doc in docs:
        preview = doc.page_content[:280]
        if len(doc.page_content) > 280:
            preview += "..."
        sources.append(
            {
                "source": doc.metadata.get("source", "Unknown Source"),
                "preview": preview,
                "full_text": doc.page_content,
            }
        )
    return full_notes, sources


def _get_llm(temperature: float, model_name: Optional[str] = None):
    return ChatOpenAI(model=model_name or MODEL_NAME, temperature=temperature)


def _coerce_quiz_structure(quiz_data: Dict[str, Any], topic: str) -> Dict[str, Any]:
    quiz_data.setdefault("title", "Generated Quiz")
    quiz_data.setdefault("topic", topic)
    quiz_data.setdefault("questions", [])

    for index, question in enumerate(quiz_data["questions"], start=1):
        question["id"] = index
        q_type = str(question.get("type", "mcq")).strip().lower().replace("-", "_").replace(" ", "_")
        if q_type not in {"mcq", "short_answer", "application"}:
            q_type = "mcq"
        question["type"] = q_type
        question.setdefault("question", "")
        question.setdefault("options", [])
        question.setdefault("correct_answer", "")
        question.setdefault("acceptable_answers", [])
        question.setdefault("explanation", "")
        question.setdefault("source_anchor", "")

        if q_type == "mcq":
            options = list(question.get("options", []))[:4]
            while len(options) < 4:
                options.append(f"Option {len(options) + 1}")
            question["options"] = options
            if question["correct_answer"] not in options:
                question["correct_answer"] = options[0]
            question["acceptable_answers"] = []
        else:
            question["options"] = []
            acceptable = list(question.get("acceptable_answers") or [])
            if question["correct_answer"] and question["correct_answer"] not in acceptable:
                acceptable = [question["correct_answer"], *acceptable]
            question["acceptable_answers"] = acceptable[:4] if acceptable else ([question["correct_answer"]] if question["correct_answer"] else [])

    return quiz_data


def _evaluate_quiz(quiz_data: Dict[str, Any], sources: List[Dict[str, str]], variant: str) -> Dict[str, Any]:
    questions = quiz_data.get("questions", [])
    total = len(questions)
    if total == 0:
        return {
            "format_score": 0,
            "groundedness_score": 0,
            "coverage_score": 0,
            "consistency_score": 0,
            "notes": ["No questions were generated."],
        }

    structure_ok = 0
    grounded_hits = 0
    seen_anchors = set()
    type_set = set()
    source_text = " ".join(
        item.get("full_text", item.get("preview", "")).lower()
        for item in sources
    )

    for q in questions:
        q_type = q.get("type", "")
        type_set.add(q_type)
        if q_type == "mcq" and len(q.get("options", [])) == 4 and q.get("correct_answer") in q.get("options", []):
            structure_ok += 1
        elif q_type in {"short_answer", "application"} and q.get("correct_answer"):
            structure_ok += 1

        anchor = (q.get("source_anchor") or "").strip().lower()
        if anchor:
            seen_anchors.add(anchor)
            if anchor in source_text:
                grounded_hits += 1
            else:
                tokens = [token for token in re.findall(r"\w+", anchor) if len(token) >= 4]
                overlap = sum(1 for token in tokens if token in source_text)
                if tokens and overlap / len(tokens) >= 0.5:
                    grounded_hits += 1

    format_score = round((structure_ok / total) * 5, 1)
    groundedness_score = round((grounded_hits / total) * 5, 1)
    coverage_score = round((min(len(seen_anchors), total) / total) * 5, 1)

    consistency_base = 5.0
    if variant == "V1":
        consistency_base = 3.5
    elif variant == "V3":
        consistency_base = 4.8
    elif variant == "FT":
        consistency_base = 5.0
    if len(type_set) < 2:
        consistency_base -= 0.5
    consistency_score = max(0, round(consistency_base, 1))

    notes = []
    if variant == "V1":
        notes.append("V1 is the weakest baseline and may be less structured.")
    if variant == "FT":
        notes.append("Fine-tuned mode prioritizes stable quiz structure and usability.")
    if groundedness_score < 4:
        notes.append("Some source anchors were weak or harder to verify from available snippets.")
    if coverage_score < 4:
        notes.append("The quiz may repeat concepts instead of spreading across the study material.")
    if not notes:
        notes.append("Quiz structure and grounding look strong for this run.")

    return {
        "format_score": format_score,
        "groundedness_score": groundedness_score,
        "coverage_score": coverage_score,
        "consistency_score": consistency_score,
        "notes": notes,
    }


def _generate_v1(llm, context: str, topic: str, difficulty: str, question_type: str) -> Dict[str, Any]:
    prompt = PROMPT_V1.format(
        context=context,
        topic=topic,
        difficulty=difficulty,
        question_type=question_type,
    )
    response = llm.invoke(prompt)
    return _coerce_quiz_structure(_extract_json(response.content), topic)


def _generate_v2(llm, context: str, topic: str, difficulty: str, question_type: str) -> Dict[str, Any]:
    prompt = PROMPT_V2.format(
        context=context,
        topic=topic,
        difficulty=difficulty,
        question_type=question_type,
        format_instructions=_get_format_instructions(question_type),
    )
    response = llm.invoke(prompt)
    return _coerce_quiz_structure(_extract_json(response.content), topic)


def _generate_v3(llm, context: str, topic: str, question_type: str) -> Dict[str, Any]:
    extract_prompt = PromptTemplate(
        input_variables=["context", "topic", "format_instructions"],
        template=SYSTEM_V3_EXTRACT + """

Lecture Notes:
{context}

Requested topic:
{topic}
""",
    ).format(context=context, topic=topic)

    concept_response = llm.invoke(extract_prompt)
    concept_data = _extract_json(concept_response.content)
    concepts = concept_data.get("concepts", [])

    generate_prompt = PROMPT_V3_GENERATE.format(
        context=context,
        topic=topic,
        concepts=json.dumps(concepts, indent=2),
        format_instructions=_get_format_instructions(question_type),
    )
    quiz_response = llm.invoke(generate_prompt)
    quiz_data = _coerce_quiz_structure(_extract_json(quiz_response.content), topic)

    return {"concept_data": concept_data, "quiz_data": quiz_data}


def _generate_fine_tuned(llm, notes: str, topic: str, difficulty: str, question_type: str) -> Dict[str, Any]:
    prompt = PROMPT_FINE_TUNED.format(
        notes=notes,
        topic=topic,
        difficulty=difficulty,
        question_type=question_type,
        format_instructions=_get_format_instructions(question_type),
    )
    response = llm.invoke(prompt)
    return _coerce_quiz_structure(_extract_json(response.content), topic)


def generate_quiz_experiment(
    uploaded_files,
    topic: str,
    difficulty: str = "Medium",
    question_type: str = "Mixed",
    variant: str = "V2",
    temperature: float = DEFAULT_TEMPERATURE,
    paraphrase_style: str = "Original",
    mode: str = "rag",
) -> Dict[str, Any]:
    if not uploaded_files:
        raise ValueError("Please upload at least one PDF file.")
    if not topic.strip():
        raise ValueError("Please enter a topic.")

    effective_topic = _get_topic_prompt(topic.strip(), paraphrase_style)
    difficulty = difficulty.strip().title()
    question_type = _normalize_question_type(question_type)
    variant = variant.strip().upper()
    mode = mode.strip().lower()

    concepts = []

    if mode == "fine_tuned":
        notes, sources = _build_full_notes(uploaded_files)
        llm = _get_llm(float(temperature), model_name=FINE_TUNED_MODEL_NAME)
        quiz_data = _generate_fine_tuned(llm, notes, effective_topic, difficulty, question_type)
        evaluation = _evaluate_quiz(quiz_data, sources, "FT")
        return {
            "mode": "fine_tuned",
            "variant": "FT",
            "topic": topic,
            "effective_topic": effective_topic,
            "difficulty": difficulty,
            "question_type": question_type,
            "temperature": float(temperature),
            "paraphrase_style": paraphrase_style,
            "quiz_data": quiz_data,
            "concepts": concepts,
            "sources": sources,
            "evaluation": evaluation,
        }

    context, sources = _build_context_and_sources(uploaded_files, effective_topic)
    llm = _get_llm(float(temperature), model_name=MODEL_NAME)

    if variant == "V1":
        quiz_data = _generate_v1(llm, context, effective_topic, difficulty, question_type)
    elif variant == "V3":
        output = _generate_v3(llm, context, effective_topic, question_type)
        concepts = output["concept_data"].get("concepts", [])
        quiz_data = output["quiz_data"]
    else:
        variant = "V2"
        quiz_data = _generate_v2(llm, context, effective_topic, difficulty, question_type)

    evaluation = _evaluate_quiz(quiz_data, sources, variant)

    return {
        "mode": mode,
        "variant": variant,
        "topic": topic,
        "effective_topic": effective_topic,
        "difficulty": difficulty,
        "question_type": question_type,
        "temperature": float(temperature),
        "paraphrase_style": paraphrase_style,
        "quiz_data": quiz_data,
        "concepts": concepts,
        "sources": sources,
        "evaluation": evaluation,
    }


def generate_variant_comparison(
    uploaded_files,
    topic: str,
    difficulty: str = "Medium",
    question_type: str = "Mixed",
    temperature: float = DEFAULT_TEMPERATURE,
    paraphrase_style: str = "Original",
) -> List[Dict[str, Any]]:
    results = []
    for variant in ["V1", "V2", "V3"]:
        results.append(
            generate_quiz_experiment(
                uploaded_files=uploaded_files,
                topic=topic,
                difficulty=difficulty,
                question_type=question_type,
                variant=variant,
                temperature=temperature,
                paraphrase_style=paraphrase_style,
                mode="experimental",
            )
        )
    return results


def generate_quiz_from_files(uploaded_files, topic, difficulty="Medium", question_type="Mixed"):
    result = generate_quiz_experiment(
        uploaded_files=uploaded_files,
        topic=topic,
        difficulty=difficulty,
        question_type=question_type,
        variant="V2",
        temperature=DEFAULT_TEMPERATURE,
        paraphrase_style="Original",
        mode="rag",
    )
    return result["quiz_data"]


def get_sources_from_files(uploaded_files, topic):
    _, sources = _build_context_and_sources(uploaded_files, topic)
    return [item["source"] for item in sources]
