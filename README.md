<div align="center">
  
# 🧠 QuizLab: AI Smart Learning Companion

**An advanced, secure, and grounded AI system for automated quiz generation and semantic grading.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/🦜🔗-LangChain-gray.svg)](https://langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 📖 Overview

Welcome to the **PromptShield Coursework** repository! This project hosts **QuizLab**, an intelligent application built to revolutionize study habits. QuizLab allows users to upload course materials (like PDF slides) or provide URLs, from which it dynamically generates high-quality, customized study quizzes (Multiple Choice, Short Answer, and Application questions). 

Unlike basic wrappers around Large Language Models (LLMs), QuizLab employs a rigorous pipeline integrating **Retrieval-Augmented Generation (RAG)**, **Semantic Grading**, **Fine-Tuned Models**, and **Robust Prompt Security** to prevent hallucinations and provide accurate, syllabus-aligned feedback.

---

## 🏗️ System Architecture

QuizLab is built on a modular, multi-layered architecture designed for scalability and accuracy:

1. **Frontend Interface (Streamlit):** A clean, interactive UI where students upload documents, configure quiz difficulty/topics, and interact with the AI.
2. **Document Ingestion & Chunking (LangChain & PyPDF):** Uploaded materials are parsed, semantically chunked, and prepared for vectorization.
3. **Retrieval Engine (ChromaDB & OpenAI Embeddings):** Chunks are converted into dense vector embeddings and stored in ChromaDB, enabling semantic search and retrieval to ground the LLM's responses.
4. **Generation & Grading Core (OpenAI `gpt-4o-mini` & Fine-tuned Models):** Generates questions strictly based on retrieved context and evaluates user answers using a hybrid grading pipeline (Exact Match + Semantic Similarity + LLM-as-a-Judge).
5. **Security Layer (PromptShield):** Intercepts user inputs to detect prompt injection attempts and filters out off-topic requests before they ever reach the generation engine.

---

## 🔬 Prompt Engineering Methodology

A core focus of this project was the rigorous evaluation of different prompt engineering techniques. We compared various methods to identify the most robust approach for educational content generation.

### Methods Researched & Compared

1. **Zero-Shot Prompting**
   * **Use Case:** Used initially for basic question generation without providing examples.
   * **Limitations Found:** Highly inconsistent output formats. The model frequently hallucinated information not present in the source material and struggled to consistently return valid JSON.

2. **Few-Shot Prompting**
   * **Use Case:** Providing 3-5 examples of expected input/output pairs to guide the model's formatting.
   * **Limitations Found:** Improved formatting significantly over Zero-Shot, but the model still occasionally drifted in difficulty level or included outside knowledge when the context was ambiguous.

3. **Chain-of-Thought (CoT) & Decomposition**
   * **Use Case:** Breaking the task into explicit steps. We instructed the model to first *extract core concepts* from the text, and *then* generate questions based only on those concepts.
   * **Limitations Found:** Increased latency and token usage. However, accuracy improved drastically.

4. **Model Fine-Tuning**
   * **Use Case:** Training a specialized, smaller model on a curated dataset of high-quality quiz generation pairs to internalize the formatting and style requirements.
   * **Limitations Found:** Expensive to set up and less adaptable to suddenly changing prompt structures.

5. **Retrieval-Augmented Generation (RAG)**
   * **Use Case:** Providing the model with dynamic, highly relevant snippets of the textbook/slides as a strict context window.
   * **Limitations Found:** Dependent on the quality of the embedding and chunking strategy. Poor chunking leads to disjointed context.

### The Chosen Approach: The Best Methods & Why

After extensive comparison, we selected a **Hybrid RAG + Decomposed Few-Shot Prompting** approach as the best solution for QuizLab. 

**Why this combination?**
* **Specificity & Grounding:** By using **RAG**, we eliminate hallucinations. The LLM is strictly instructed: *"Answer only using the provided context."* If the context doesn't contain the answer, the model gracefully declines.
* **Structural Consistency:** By utilizing **Decomposition** alongside **Few-Shot examples**, we force the model to show its work (identifying topics first) before outputting the final JSON schema. This nearly guarantees 100% parseable JSON outputs and high-quality distractors in multiple-choice questions.
* **Security & Alignment:** We wrap our prompts in explicit system instructions that define the AI's persona ("You are a strict, helpful academic professor") and establish hard boundaries against jailbreaks or topic-drifting, which zero-shot prompting completely failed to defend against.

---

## 🚀 Agile Development Phases

This project was developed iteratively, simulating an industry-standard Agile lifecycle. The repository is structured into these historical phases:

* 📁 **`Phase_1_Project_Proposal`**: Initial system design and architecture ideation.
* 📁 **`Phase_2_Prompt_Engineering`**: Baseline model evaluation, prompt sensitivity testing, and establishing core metrics.
* 📁 **`Phase_3_Model_Fine_Tuning`**: Creation of custom datasets and fine-tuning experiments for consistent JSON output.
* 📁 **`Phase_4_RAG_LangChain`**: Implementation of ChromaDB, document chunking, and the LangChain retrieval pipeline.
* 📁 **`Phase_5_UI_Integration`**: Merging the backend logic with the Streamlit interactive frontend.
* 📁 **`Phase_6_Security_Final_App`**: Hardening the application against prompt hacking, injection, and off-topic queries. (Contains the final production application).

---

## 💻 Installation & Usage

### Prerequisites
* Python 3.10+
* An active [OpenAI API Key](https://platform.openai.com/)

### Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Hotragn/PromptShield-Coursework.git
   cd PromptShield-Coursework
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment:**
   Create a `.env` file in the root directory (or inside the Phase 6 folder) and add your API key:
   ```env
   OPENAI_API_KEY=sk-your-openai-api-key
   ```

### Running the Application
The most advanced, hardened version of the app is located in Phase 6.
```bash
cd Phase_6_Security_Final_App
streamlit run app.py
```

---

## 📚 Research Papers & Literature
Our methodologies are heavily backed by academic research in prompt engineering, LLM mechanics, and AI security. All referenced literature can be found in the `/Research_Papers` directory. 

Key themes include:
* *The mechanics of LLM language understanding*
* *Techniques for mitigating prompt injection in production systems*
* *Comparative studies on CoT vs Standard Prompting for complex reasoning tasks*

---

*Developed as part of the PromptShield Coursework initiative.*
