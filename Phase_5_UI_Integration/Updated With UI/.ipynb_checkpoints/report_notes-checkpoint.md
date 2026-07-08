# Assignment 5 — Report Notes for Teammate

## Project Overview
**App:** AI Smart Learning Companion — Quiz Generation Module
**Group:** Group 10
**Goal:** Refactor the A4 quiz generation prompt pipeline using LangChain and Azure Prompt Flow,
compare 3 prompt variants through iterative refinement, and evaluate with automated metrics.

---

## Tools & Environment

| Item | Detail |
|------|--------|
| LangChain version | 0.2.16 |
| LangChain Core | 0.2.38 |
| LLM (LangChain) | gpt-4o-mini |
| LLM (Azure Prompt Flow) | gpt-4 |
| Temperature | 0.2 (both tools) |
| Embedding model | text-embedding-3-small (OpenAI) |
| Vector store | Chroma (local) |
| Chunk size | 1200 chars, overlap 100 |
| Text splitter | RecursiveCharacterTextSplitter |
| RAG retrieval k | top-3 chunks per query |
| PDF loader | pypdf |

**Note on model difference:** Azure Prompt Flow's OpenAI connection does not expose
gpt-4o-mini in the model dropdown; gpt-4 was used there instead. Both share identical
prompts and temperature=0.2.

---

## Data Source
10 course slide PDFs from `Course Slides/` folder (Modules 1–10), loaded automatically.
All PDFs are chunked and indexed in a local Chroma vector store.
RAG retrieves the top-3 most relevant chunks for each topic query.

---

## Three Prompt Variants

### V1 — Weak Zero-Shot Baseline
**Purpose:** Intentionally minimal starting point to establish a low baseline.

**Full prompt:**
```
You are a tutor. Make a quiz based on the notes.
```

**Design decision:** No format requirements, no difficulty definitions, no grounding constraint.
This causes the model to produce free-form output without the required question structure.

---

### V2 — Hardened
**Purpose:** Refined version addressing all bottlenecks identified in V1.

**Changes from V1:**
1. Added explicit output format: 3 MCQ (Easy/Medium/Hard) + 2 Short Answer + 1 Application
2. Added difficulty definitions (Easy=recall, Medium=application, Hard=analysis/synthesis)
3. Replaced vague persona with direct role: "graduate teaching assistant"
4. Added grounding constraint: "every question must be directly traceable to the lecture notes"

**Chain type:** `RetrievalQA.from_chain_type` (single-step RAG chain)

---

### V3 — Decomposed Chain (new in A5)
**Purpose:** Demonstrate flow engineering through a two-step decomposed pipeline.

**Step 1 prompt (extract_concepts):**
- Role: expert academic content analyst
- Task: extract 5-7 key concepts from the notes (name, definition, why it matters)
- Constraint: base everything strictly on the provided notes

**Step 2 prompt (generate_quiz):**
- Role: graduate teaching assistant
- Task: generate quiz anchored to the extracted concepts
- Same format requirements as V2

**Implementation:** Two sequential `llm.invoke()` calls (not a single chain).
The intermediate concept list is inspectable between steps.

---

## Pipeline Architecture

### LangChain (Cells 1–15)
```
PDF files
    → RecursiveCharacterTextSplitter (1200 chars)
    → Chroma vector store (text-embedding-3-small)
    → Retriever (top-3 chunks)

For each topic:
    V1: topic → RetrievalQA (V1 prompt) → quiz
    V2: topic → RetrievalQA (V2 prompt) → quiz
    V3: topic → retriever → LLM call 1 (extract concepts)
                          → LLM call 2 (generate quiz)
```

### Azure Prompt Flow
```
Inputs: topic (string) + context (string, pasted from notebook Cell 17)
    → LLM Node 1: extract_concepts  (SYSTEM_V3_EXTRACT prompt)
    → LLM Node 2: generate_quiz     (SYSTEM_V3_QUIZ prompt)
    → Output: quiz
```

The context for Azure Prompt Flow is retrieved using the LangChain retriever
(notebook Cell 17) and manually pasted into the Azure flow input.
This mirrors the RAG step that LangChain performs automatically.

---

## Evaluation Methodology

**Evaluator model:** gpt-4o-mini, temperature=0.0 (deterministic)

**4 metrics, scored 1–5 (strict rubric):**

| Metric | 5/5 requires | 3/5 | 1/5 |
|--------|-------------|-----|-----|
| Format Compliance | Exactly 3 MCQ (Easy/Medium/Hard) + 2 SA + 1 App, all with answers | Some types missing or unlabeled | Free-form, no structure |
| Groundedness | Every Q&A traceable word-for-word to notes | Most grounded, 1-2 use outside knowledge | Relies heavily on general knowledge |
| Clarity | Unambiguous questions, distinct choices, precise explanations | Some vague questions or generic explanations | Confusing questions |
| Hallucination Risk | Zero statements not in notes | A few unverifiable claims | Multiple invented facts |

**Test topics (3):**
1. Prompting techniques including few-shot and chain-of-thought
2. Retrieval Augmented Generation and vector embeddings
3. ReAct agents and tool use in LangChain

---

## Experimental Results

| Variant | Format Compliance | Groundedness | Clarity | Hallucination Risk | Avg |
|---------|:-----------------:|:------------:|:-------:|:-----------------:|:---:|
| V1 Baseline | 3 | 5 | 4 | 5 | 4.25 |
| V2 Hardened | 5 | 5 | 5 | 5 | 5.0 |
| V3 Decomposed | 5 | 5 | 5 | 5 | 5.0 |

Scores are consistent across all 3 topics for V1 and V2.
V3 showed minor variance across runs (one run scored 4s on topic 1; another run scored 5s).

---

## Key Findings for Report

1. **V1 → V2:** Adding explicit format constraints + difficulty definitions produced
   the largest measurable improvement (Format Compliance: 3→5, Clarity: 4→5).
   This is the most impactful single change.

2. **V2 → V3:** Scores are equivalent on automated metrics. V3's value is
   **auditability** — the concept extraction step is inspectable before quiz generation,
   enabling human review and correction mid-pipeline. This is a core flow engineering principle.

3. **Variance observation:** V3 showed score fluctuation across runs while V1 and V2 were stable.
   This is because decomposed chains amplify upstream variance — a small difference in how
   Step 1 phrases a concept propagates into Step 2. In production, this is addressed with
   Self-Consistency (run multiple times, aggregate results) and Self-Criticism (add a review
   node between Step 1 and Step 2).

4. **Groundedness / Hallucination Risk were 5/5 for all variants** because RAG retrieval
   already provides highly relevant context, limiting the model's ability to hallucinate
   regardless of prompt quality. The prompt constraint on grounding therefore shows its
   value primarily in Clarity and Format, not in these two metrics.

---

## Azure Prompt Flow Screenshots (in `Azure Prompt Flow/` folder)

| File | Shows |
|------|-------|
| `azure_pf_flow_structure.jpg` | DAG: inputs → extract_concepts → generate_quiz → outputs, both nodes Completed |
| `azure_pf_node1_concepts.jpg` | Node 1 configuration: prompt, model (gpt-4), temperature (0.2), input connections |
| `azure_pf_node1_concepts_output.jpg` | Node 1 output: 7 extracted key concepts |
| `azure_pf_node2_quiz.jpg` | Node 2 configuration: prompt, model (gpt-4), temperature (0.2), input connections |
| `azure_pf_node2_quiz_output.jpg` | Node 2 output: full generated quiz (MCQ + Short Answer + Application) |

