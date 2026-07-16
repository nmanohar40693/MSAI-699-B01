# Project Knowledge Continuity in AI-Assisted Software Engineering: Evaluating a Lifecycle-Guided Context Construction Strategy

This repository contains the preprocessing pipeline, comparative experimental framework, evaluation tasks, and model client implementations for this capstone project. The project evaluates how different repository context construction strategies impact a Large Language Model's (LLM) ability to maintain project knowledge continuity across sequential software engineering tasks.

---

## 1. Repository Objective

This repository contains the Week 3 baseline implementation for the MSAI-699 Capstone project. The objective of this milestone was to establish a reproducible baseline experimental framework for evaluating project knowledge continuity in AI-assisted software engineering.

---

## 2. Overall System Architecture

The project is structured into two decoupled, modular subsystems:

```
MSAI-699-B01/
├── .gitignore
├── README.md
│
├── capstone_preprocessing/      # Subsystem 1: Preprocessing & Data Extraction
│   ├── main.py                  # Pipeline driver orchestrator
│   ├── config.json              # Repo cloner settings (GitHub repository, identifiers)
│   ├── requirements.txt         # Preprocessing library requirements
│   └── src/                     # Modular processing modules
│       ├── cloner.py            # Local clone & git index inspector
│       ├── extractor.py         # Code, commit, issue, and PR crawler
│       ├── cleaner.py           # HTML stripping and metadata cleaning
│       ├── standardizer.py      # Schema enforcement & hashing
│       ├── aligner.py           # release boundaries and version snapshot aligner
│       └── relationship.py      # Traceability registry constructor
│
└── capstone_experiments/        # Subsystem 2: Context Building & Model Execution
    ├── run_experiment.py        # CLI experiment orchestrator & reporter
    ├── requirements.txt         # Inference and indexing requirements
    ├── config/
    │   └── default_config.json  # Model parameters & RAG settings
    ├── data/
    │   └── evaluation_tasks.json# Standardized SE tasks and targets
    └── src/
        ├── config.py            # Config file parser
        ├── model.py             # Official GenAI SDK wrapper (Mock / Live mode)
        ├── evaluation.py        # Task loaders
        ├── metrics.py           # Performance and token usage metrics collector
        ├── storage.py           # JSON run sheet results serializer
        ├── rag_indexer.py       # SentenceTransformers local search
        ├── graph_builder.py     # NetworkX Lifecycle-Guided Project Knowledge Graph constructor
        └── strategies/
            └── base_strategy.py # Concrete strategy builders (PromptOnly, RAG, Memory, Lifecycle)
```

---

## 3. Core Technical Components

### 3.1. Response Generation Model
The framework integrates Gemini 3.5 Flash as the **response generation model** using the official `google-genai` SDK.
* **Configuration**: The model is configured at `temperature: 0.0` to ensure deterministic, reproducible outputs for comparative analysis.
* **Authentication**: The API key is loaded dynamically from the `GEMINI_API_KEY` environment variable when live mode is active, preventing hardcoded secrets in the codebase.

### 3.2. Baseline Semantic Embedding Model
For semantic vector similarity retrieval, the standard all-MiniLM-L6-v2 is integrated as the **baseline semantic embedding model**. 
* **Vector Search**: Computes cosine similarities using an in-memory NumPy matrix index matching the search queries.
* **Caching**: Leverages local JSON SHA-256 content-hash caches to avoid redundant embedding calculations.

### 3.3. Lifecycle-Guided Project Knowledge Graph
The context construction strategy builds a **Lifecycle-Guided Project Knowledge Graph** using NetworkX for the active release version.
* **Nodes**: Represent `source_code`, `test_case`, `documentation`, `build_config`, `commit`, `issue`, and `pull_request` artifacts. Each node is classified into its corresponding software development lifecycle stage (Requirements, Implementation, Testing, Debugging, Code review, Documentation, Maintenance).
* **Edges**: Represent explicit traceability links (e.g., commit changes file, test suite tests class, PR resolves issue).
* **Context Traversal**: Resolves semantic entry points and runs a depth-restricted (depth=2) breadth-first search (BFS) to gather surrounding context from the graph.

---

## 4. Context Construction Strategies

The experimental framework compares four context strategy conditions:

1. **Prompt-only interactions**: The response generation model receives only the task query description. No repository context is supplied.
2. **Retrieval-Augmented Generation (RAG)**: Retrieves the Top-K (default: 5) most semantically similar code chunks from the snapshot.
3. **Memory-Augmented Prompting**: Appends summaries of prior task outputs chronologically during sequential executions to simulate developmental memory.
4. **Lifecycle-Guided Context Construction Strategy**: Traverses the Lifecycle-Guided Project Knowledge Graph starting from semantic file nodes to gather historically connected lifecycle artifacts.

---

## 5. Installation & Execution

### 5.1. Installation Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/naveenmanohardeveloper/MSAI-699-B01.git
   cd MSAI-699-B01
   ```
2. Set up a virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r capstone_preprocessing/requirements.txt
   pip install -r capstone_experiments/requirements.txt
   ```

### 5.2. Execution Commands
1. **Run the Preprocessing Pipeline**:
   ```bash
   cd capstone_preprocessing
   python3 main.py
   ```
   This clones the repository, standardizes metadata, and writes version-aligned snapshots to `output/prepared_dataset/`.

2. **Configure API Key (For Live Inference)**:
   ```bash
   export GEMINI_API_KEY="AIzaSyYourActualKeyHere"
   ```
   *(Note: If `GEMINI_API_KEY` is not set, the framework runs in offline mock mode using the default configs).*

3. **Run the Context Engineering Experiments**:
   ```bash
   cd ../capstone_experiments
   
   # 1. Prompt-Only Run
   python3 run_experiment.py --strategy prompt-only
   
   # 2. RAG Baseline Run
   python3 run_experiment.py --strategy rag
   
   # 3. Memory-Augmented Run
   python3 run_experiment.py --strategy memory
   
   # 4. Lifecycle-Guided Run
   python3 run_experiment.py --strategy lifecycle
   ```

---

## 6. Baseline Performance Summary

The consolidated table below displays the statistical results captured during live model runs on your preprocessed Spring PetClinic dataset (3 tasks across 3 version tags):

| Metric | Prompt-only interactions | Retrieval-Augmented Generation (RAG) | Memory-Augmented Prompting | Lifecycle-Guided Context Construction Strategy |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Latency** | 15.49s | 18.72s | 12.78s | **12.16s** |
| **Total Input Tokens** | 139 | 3,226 | 405 | **602** |
| **Total Output Tokens** | 387 | 123 | 318 | **181** |
| **Mean Context Size** | 0.0 chars | 3,331.0 chars | 446.7 chars | **497.0 chars** |

*   **Prompt Overhead**: The Lifecycle-Guided Context Construction Strategy reduces input token usage compared to standard RAG while retaining structural project context.
*   **Latency**: The Lifecycle-Guided Context Construction Strategy recorded the lowest mean latency among the context strategies tested.

---

## 7. Current Limitations & Week 4 Roadmap

### Current Limitations
1. **Version Boundary Density**: When git commits are made within a very short duration (e.g., sub-minute intervals), chronological release windows become narrow, resulting in graph data sparsity.
2. **Missing Pre-Version Targets**: If a target test file is not introduced until a later release version, older snapshots will yield empty graph traversals.

### Week 4 Roadmap
* **Experimentation**: Run extensive comparative runs across more release tags to evaluate strategy stability.
* **Model Optimization**: Fine-tune chunking thresholds, top-k parameters, and traversal depths.
* **Evaluation of Additional Performance Metrics**: Introduce programmatic evaluations for response relevance, consistency, and syntax accuracy.
* **Explainability**: Map path traversals to explain how the context was gathered from raw artifacts.
