# compliance-analysis
✅ Technical Summary for UAE IA Compliance Analyzer
🧩 Key Libraries Used
Category	Library	Purpose
Web UI	Gradio	Provides interactive frontend for uploading documents and viewing results
PDF/DOCX Parsing	pdfplumber, python-docx	Extracts readable text from PDF and DOCX files
NLP & AI	Transformers, peft, sentence-transformers, torch	Loads fine-tuned LLaMA model and sentence embeddings for semantic retrieval
LLM Inference	transformers.pipeline	Handles LLM response generation (LLaMA 3.2)
Vector Similarity	scikit-learn (cosine_similarity)	Calculates relevance of document sections using embedding vectors
Data Models	pydantic	Structured schema for compliance results
Utility	re, yaml, json, os, datetime, threading, io, pathlib	File I/O, cleanup, validation, logging, and concurrency

⚙️ How the Compliance Analyzer Works
User Interface

Built with Gradio using a multi-tab layout: Dashboard, Matrix View, Detailed Report

User uploads PDF or DOCX files via drag-and-drop

Document Preprocessing

Extracts and cleans text using pdfplumber or python-docx

Splits document into semantic sections for analysis

Control Mapping

Loads UAE IA control definitions from a JSON file (controlt.json)

Each control contains id, description, requirement

Semantic Context Retrieval

For each control, retrieves the most relevant document sections

Uses SentenceTransformer embeddings and cosine similarity

LLM-Based Evaluation

Prompts a fine-tuned LLaMA 3.2 model using transformers.pipeline

LLM responds in structured JSON with:

control_id, status (Fully / Partially / Not Addressed), reason, evidence

If the model fails or is vague, regex/YAML/JSON fallback is used

Post-Processing & Validation

Evidence quality is validated (length, keyword match)

Status downgraded if evidence is missing, vague, or irrelevant

UI Rendering

Results are rendered in:

Summary dashboard (pie chart-like visual)

Matrix view (controls vs documents with status dots)

Detailed report (per-control analysis with reason/evidence)

💡 Explanation
This system automates UAE IA compliance checks using advanced AI.

You upload your internal security policies (PDF or DOCX).

The system breaks them into logical chunks and compares each chunk to UAE IA controls.

Our fine-tuned AI (LLaMA 3.2) reads the documents and determines whether each control is Fully, Partially, or Not Addressed.

It gives an explanation ("reason") and points to exact quotes from your document ("evidence").

You get visual dashboards, matrix views, and downloadable reports showing where your documents meet or miss compliance.

🧠 Model Summary
Model: Fine-tuned LLaMA 3.2 1B

Adapter: PEFT (LoRA) merged into base model

Embedding Model: all-MiniLM-L6-v2 from sentence-transformers

Inference: Controlled generation for JSON format extraction

LLM Evaluation Logic: Status is validated using evidence length + keyword overlap + vague pattern detection

