# QuizLab: AI Smart Learning Companion

Welcome to the **PromptShield Coursework** repository! This repository hosts the evolution and final implementation of **QuizLab**, an AI-powered Smart Learning Companion developed iteratively through a rigorous prompt engineering lifecycle.

## Overview
QuizLab allows users to upload lecture slides (PDFs) or webpage URLs and automatically generates customized, grounded study quizzes (MCQ, Short Answer, Application) using advanced Retrieval-Augmented Generation (RAG) and specialized prompt engineering techniques.

### Key Features
* **Custom Quiz Generation**: Dynamically generate quizzes based on specific topics and difficulty levels.
* **Grounded RAG Engine**: Ensures AI answers and generated questions strictly adhere to the uploaded course material context, preventing hallucination.
* **Semantic Answer Scoring**: Hybrid grading pipeline (Exact Match + Semantic Embeddings + LLM-as-a-Judge) for short answers.
* **Security Hardening**: Document scanning to detect prompt injection attempts, topic sanitization, and strict system instructions to keep the AI on track.
* **Agile/Phase Evolution**: The codebase and prompts were iteratively tested, fine-tuned, and refined across 6 major development phases.

## Tech Stack
* **Frontend UI**: [Streamlit](https://streamlit.io/)
* **AI & Orchestration**: [LangChain](https://www.langchain.com/)
* **LLM Provider**: OpenAI (`gpt-4o-mini`, Text Embeddings)
* **Vector Store**: [ChromaDB](https://www.trychroma.com/)
* **PDF Parsing**: `pypdf`
* **Data Handling**: `pandas`

## Prompt Tags & Techniques Used
Throughout the evolution of this project, we applied various advanced prompt engineering paradigms:
* `#ZeroShot` `#FewShot`
* `#ChainOfThought` (CoT)
* `#Decomposition` (Breaking generation into Concept Extraction -> Quiz Generation)
* `#PromptSensitivity` (Analyzing model performance across semantic variants)
* `#FineTuning` (Creating specialized models for consistent JSON output)
* `#RAG` (Retrieval-Augmented Generation via LangChain and Chroma)
* `#PromptHackingDefenses` (Input sanitization, injection detection, structural hardening)

## Agile Phase Deployment
This repository has been structured into iterative **Phases** simulating a professional Agile workflow. You can view the code and documentation artifacts in their respective folders:

* **Phase 1**: Project Proposal & System Design (`/Phase_1_Project_Proposal`)
* **Phase 2**: Prompt Engineering & Base Model Evaluation (`/Phase_2_Prompt_Engineering`)
* **Phase 3**: Model Fine-Tuning (`/Phase_3_Model_Fine_Tuning`)
* **Phase 4**: RAG Implementation & LangChain Pipeline (`/Phase_4_RAG_LangChain`)
* **Phase 5**: Streamlit UI Integration (`/Phase_5_UI_Integration`)
* **Phase 6**: Security Hardening & Final Application (`/Phase_6_Security_Final_App`)

## Setup & Local Deployment

### Prerequisites
* Python 3.9+
* OpenAI API Key

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/promptshield-coursework.git
   cd promptshield-coursework
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   Create a `.env` file in the root directory and add your OpenAI API Key:
   ```env
   OPENAI_API_KEY=sk-your-key-here
   ```

### Running the App
The final, hardened application resides in `Phase_6_Security_Final_App`.
```bash
cd Phase_6_Security_Final_App
streamlit run app.py
```

## Research Papers
All referenced literature and background research papers are included in the `/Research_Papers` directory.
