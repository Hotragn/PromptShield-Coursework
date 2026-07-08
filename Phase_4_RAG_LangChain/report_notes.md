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
10 course slide PDFs from `course_slides/` folder (Modules 1–10), loaded automatically.
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
| V2 Hardened | ~5 | 5 | ~5 | 5 | ~5.0 |
| V3 Decomposed | ~5 | 5 | ~5 | 5 | ~5.0 |

V1 scores are perfectly stable across all runs and all 3 topics.
V2 and V3 typically score 5/5 but showed occasional variance (see Key Findings #3).

---

## Key Findings for Report

1. **V1 → V2:** Adding explicit format constraints + difficulty definitions produced
   the largest measurable improvement (Format Compliance: 3→5, Clarity: 4→5).
   This is the most impactful single change.

2. **V2 → V3:** Scores are equivalent on automated metrics. V3's value is
   **auditability** — the concept extraction step is inspectable before quiz generation,
   enabling human review and correction mid-pipeline. This is a core flow engineering principle.

3. **Variance observation:** Both V2 and V3 showed score fluctuations across runs, while V1
   remained perfectly stable every time. Paradoxically, V1's minimal prompt produces the most
   consistent (but lowest quality) output — there are no complex instructions to interpret
   differently. V2 and V3's richer prompts give the model more to work with, which improves
   average quality but introduces run-to-run variability. In production, Self-Consistency
   and Self-Criticism are used to address this.

4. **Groundedness / Hallucination Risk were 5/5 for all variants** because RAG retrieval
   already provides highly relevant context, limiting the model's ability to hallucinate
   regardless of prompt quality.

---

## Azure Prompt Flow — Full Report Section Notes

### What is Azure Prompt Flow
Azure Prompt Flow is a development tool inside Azure Machine Learning Studio that lets you
build, test, and deploy LLM-powered pipelines visually. Instead of writing raw API calls,
you connect nodes (Python logic, prompt templates, LLM calls) into a structured, testable
flow with a visual DAG (Directed Acyclic Graph) interface.

### Why we used it alongside LangChain
The two tools serve complementary roles in this project:

| | LangChain (notebook) | Azure Prompt Flow |
|--|--|--|
| Course slides (RAG) | Automatic — Chroma retriever | Manual — context pasted from Cell 17 |
| Flow structure | Code (Python functions) | Visual DAG (node-based interface) |
| Evaluation | Automated scoring across 3 topics | Manual inspection of each node's output |
| Strength | Full pipeline automation + metrics | Visualization, auditability, deployability |

### Setup
- Azure Machine Learning workspace created in Azure Portal (West US 2 region)
- Role assignments configured: AzureML Data Scientist on workspace; Storage Blob Data
  Contributor and Storage Table Data Contributor on the auto-created storage account
- OpenAI connection created inside Azure ML Studio (Prompt Flow > Connections),
  named "Michelle", connected with OpenAI API key

### Flow Design — Why V3
We chose to implement V3 (Decomposed Chain) in Azure Prompt Flow rather than V1 or V2
for two reasons:
1. V3's two-step structure maps directly onto Azure Prompt Flow's node-based model —
   each LLM call becomes one node, making the decomposition visually explicit
2. It demonstrates the core value proposition of Prompt Flow: breaking a complex task
   into auditable, individually testable sub-steps

### Flow Structure (see azure_pf_flow_structure.jpg)
```
Inputs
  ├── topic   (string) — the subject to generate a quiz about
  └── context (string) — relevant lecture notes retrieved from the RAG pipeline

      ↓
LLM Node 1: extract_concepts
  - Model: gpt-4, temperature: 0.2
  - Prompt: SYSTEM_V3_EXTRACT (academic content analyst role)
  - Inputs: topic → ${inputs.topic}, context → ${inputs.context}
  - Output: structured list of 5-7 key concepts

      ↓
LLM Node 2: generate_quiz
  - Model: gpt-4, temperature: 0.2
  - Prompt: SYSTEM_V3_QUIZ (teaching assistant role)
  - Input: concepts → ${extract_concepts.output}
  - Output: full quiz (3 MCQ + 2 Short Answer + 1 Application)

      ↓
Output: quiz
```

### How context is provided
The `context` input contains lecture notes retrieved from the course slides.
We use notebook Cell 17 to run the LangChain RAG retriever and print the top-3 chunks
for any topic, then paste that text into the Azure Prompt Flow `context` input field.
This makes the RAG step explicit and visible — in LangChain it happens automatically
inside the chain, while in Azure Prompt Flow it is a manual, inspectable input.

### What the screenshots show

**azure_pf_flow_structure.jpg**
The full DAG showing inputs → extract_concepts → generate_quiz → outputs.
Both nodes display a green "Completed" checkmark, confirming the flow ran successfully end-to-end.
This is the clearest visual proof that the decomposed flow works as designed.

**azure_pf_node1_concepts.jpg**
The configuration panel of the extract_concepts node, showing:
- The full SYSTEM_V3_EXTRACT prompt in the editor
- Model: gpt-4, temperature: 0.2
- Input connections: context → ${inputs.context}, topic → ${inputs.topic}
- "Validation and parsing input completed successfully"

**azure_pf_node1_concepts_output.jpg**
The actual output of Node 1 after running — a structured list of 7 key concepts extracted
from the course slides on the topic "Prompting techniques including few-shot and chain-of-thought".
Each concept includes a name, definition, and explanation of why it matters.
This intermediate output is the key advantage of the decomposed design: it can be reviewed
and corrected before the quiz is generated.

**azure_pf_node2_quiz.jpg**
The configuration panel of the generate_quiz node, showing:
- The full SYSTEM_V3_QUIZ prompt in the editor
- Model: gpt-4, temperature: 0.2
- Input connection: concepts → ${extract_concepts.output}

**azure_pf_node2_quiz_output.jpg**
The final quiz generated by Node 2, containing MCQ questions (Easy/Medium/Hard),
Short Answer questions, and an Application question — all anchored to the concepts
extracted in Node 1. This confirms the two-step chain produces a properly structured,
grounded quiz.

### Key advantages demonstrated by Azure Prompt Flow

1. **Testability** — each node can be run and inspected in isolation; if the quiz is wrong,
   you can check Node 1's output to see if the concepts were extracted correctly first

2. **Observability** — every run is logged with duration (Node 1: 20.51s, Node 2: 18.44s),
   inputs, and outputs; this is essential for debugging production pipelines

3. **Visual clarity** — the DAG makes the flow structure immediately understandable to
   anyone reviewing the pipeline, without reading code

4. **Deployability** — once validated, the flow can be deployed as a REST API endpoint
   from Azure ML Studio with one click

---

## Screenshots Folder
All 5 screenshots are in `Azure Prompt Flow_Screenshots/` folder:

| File | Content |
|------|---------|
| `azure_pf_flow_structure.jpg` | Full DAG, both nodes Completed |
| `azure_pf_node1_concepts.jpg` | Node 1 configuration panel |
| `azure_pf_node1_concepts_output.jpg` | Node 1 output: 7 extracted concepts |
| `azure_pf_node2_quiz.jpg` | Node 2 configuration panel |
| `azure_pf_node2_quiz_output.jpg` | Node 2 output: full generated quiz |
