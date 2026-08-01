# Project Knowledge Continuity in AI-Assisted Software Engineering: Evaluating a Lifecycle-Guided Context Construction Strategy

This repository contains the preprocessing pipeline, optimized experimental framework, hyperparameter search scripts, feature engineering datasets, and explainability modules for this capstone project. The project evaluates how different repository context construction strategies impact a Large Language Model's (LLM) ability to maintain project knowledge continuity across sequential software engineering tasks.

---

## 1. Repository Objective

This repository contains the Week 4 model optimization and explainability implementation for the MSAI-699 Capstone project. The objective of this milestone was to evaluate semantic embedding representations, optimize graph traversal weights using Optuna hyperparameter tuning, construct a multidimensional feature matrix, and interpret model decisions using TreeSHAP explainability.

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
│       ├── aligner.py           # Release boundaries and version snapshot aligner
│       └── relationship.py      # Traceability registry constructor
│
└── capstone_experiments/        # Subsystem 2: Context Building & Model Execution
    ├── run_experiment.py        # CLI downstream task evaluator & reporter
    ├── run_pilot.py             # Embedding model evaluator runner
    ├── tune_hyperparameters.py  # Optuna hyperparameter search orchestrator
    ├── run_explainability.py    # Feature engineering & TreeSHAP plotter
    ├── render_tables.py         # Table image generator (APA 7 formatted)
    ├── requirements.txt         # Inference and indexing requirements
    ├── config/
    │   ├── default_config.json  # Baseline parameters & RAG settings
    │   └── optimized_config.json# Optuna-tuned traversal and size weights
    ├── data/
    │   └── evaluation_tasks.json# Standardized SE tasks and targets
    └── src/
        ├── config.py            # Config file parser
        ├── model.py             # Official GenAI SDK wrapper (with 50x API retry loops)
        ├── evaluation.py        # Task loaders
        ├── metrics.py           # Performance and token usage metrics collector
        ├── storage.py           # JSON run sheet results serializer
        ├── rag_indexer.py       # SentenceTransformers local search & caching
        ├── graph_builder.py     # NetworkX Project Knowledge Graph constructor
        └── strategies/
            └── base_strategy.py # Concrete strategy builders (PromptOnly, RAG, Memory, Lifecycle)
```

---

## 3. Core Technical Components

### 3.1. Response Generation Model
The framework integrates Google Gemini 3.5 Flash as the **response generation model** using the official `google-genai` SDK.
* **Configuration**: The model is configured at `temperature: 0.0` and `max_output_tokens: 1024` to ensure deterministic, reproducible outputs for comparative analysis.
* **Robustness**: The API client wrapper implements a 50x exponential backoff retry loop to handle client-side `499` cancellations and `429` rate limits seamlessly.

### 3.2. Optimized Semantic Embedding Model
Following a pilot evaluation of multiple encoders (including `all-MiniLM-L6-v2` and `microsoft/codebert-base`), `sentence-transformers/all-mpnet-base-v2` was selected as the **optimized semantic embedding model**.
* **Performance**: Achieved the highest retrieval quality (F1-Score of **0.324** and MRR of **0.750**).
* **Caching**: Leverages local JSON SHA-256 content-hash caches partitioned by model name to avoid redundant embedding calculations.

### 3.3. Lifecycle-Guided Project Knowledge Graph
The context construction strategy builds a **Lifecycle-Guided Project Knowledge Graph** using NetworkX for the active release version.
* **Nodes**: Represent `source_code`, `test_case`, `documentation`, `build_config`, `commit`, `issue`, and `pull_request` artifacts. Each node is classified into its corresponding software development lifecycle stage (Requirements, Implementation, Testing, Debugging, Code review, Documentation, Maintenance).
* **Edges**: Represent explicit traceability links (e.g., commit changes file, test suite tests class, PR resolves issue).
* **Context Traversal**: Resolves semantic entry points and runs a depth-restricted (depth=3) weighted breadth-first search (BFS) using optimized weights: `weight_tests_class = 2.3019`, `weight_resolves_issue = 2.4364`, `weight_modified_file = 1.8027`.

---

## 4. Context Construction Strategies

The experimental framework compares four context strategy conditions:

1. **Prompt-only interactions**: The response generation model receives only the task query description. No repository context is supplied.
2. **Retrieval-Augmented Generation (RAG)**: Retrieves the Top-K (default: 5) most semantically similar code chunks from the snapshot.
3. **Memory-Augmented Prompting**: Appends summaries of prior task outputs chronologically during sequential executions to simulate developmental memory.
4. **Lifecycle-Guided Context Construction Strategy**: Traverses the Lifecycle-Guided Project Knowledge Graph starting from semantic file nodes to gather historically connected lifecycle artifacts using optimized weights and a hard budget cap of 2,010 characters.

---

## 5. Installation & Execution

### 5.1. Installation Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/nmanohar40693/MSAI-699-B01.git
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

2. **Configure API Key (For Live Downstream Inference)**:
   ```bash
   export GEMINI_API_KEY="YOUR_API_KEY_HERE"
   ```

3. **Run the Model Optimization and Explainability Scripts**:
   ```bash
   cd ../capstone_experiments
   
   # 1. Run the Embedding Model Pilot Evaluation
   python3 run_pilot.py
   
   # 2. Run the Optuna Hyperparameter Optimization (80 trials)
   python3 tune_hyperparameters.py
   
   # 3. Compile Feature Matrix and Generate TreeSHAP Visualizations
   python3 run_explainability.py
   ```

4. **Run Downstream Strategy Experiments**:
   ```bash
   # Run RAG Baseline
   python3 run_experiment.py --strategy rag --config config/default_config.json
   
   # Run Optimized Lifecycle-Guided Strategy
   python3 run_experiment.py --strategy lifecycle --config config/default_config.json
   ```

---

## 6. Experimental Results & Performance Summary

The consolidated table below displays the statistical results captured during live model runs comparing the baseline (Default Lifecycle) and optimized configurations (10 tasks across Spring PetClinic repository):

| Performance Metric | Baseline (Default) | Optimized | Absolute Change | Percentage Change |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Context Length (characters)** | 660.50 | 757.40 | +96.90 | +14.67% |
| **Total Input Tokens** | 2,372 | 2,730 | +358 | +15.09% |
| **Total Output Tokens** | 1,508 | 780 | -728 | -48.28% |
| **Mean Downstream Latency (s)** | 4.962 | 27.013 | +22.051 | +444.40% |
| **Code Solution Rate** | 40.0% | 40.0% | 0.00 | 0.00% |

*   **Context Efficiency**: The proposed Lifecycle-Guided Strategy achieves a **62.1% reduction in context size** and a **60.6% reduction in input token consumption** compared to standard RAG.
*   **Optuna Tuning**: Localized the optimal context budget (2,010 characters) and edge traversal weights, achieving a best composite objective score of **0.8765** (Trial 98).
*   **Explainability**: The Random Forest surrogate model achieved **99.31% cross-validation accuracy** and an F1-score of **0.9524**. TreeSHAP analysis verified that topological features (`graph_distance` and `traceability_degree` centrality) are the most globally significant features in context selection.

---

## 7. Week 6 Testing and Diagnostics

This section covers the diagnostic suite and testing framework implemented in Week 6 to evaluate context construction strategies, perform controlled paired A/B comparisons, and run robust fault injection tests.

### 7.1. Setup & Requirements
*   **Python Version**: Python 3.8+ (tested on Python 3.13)
*   **Dependencies**: Ensure package dependencies listed in `capstone_experiments/requirements.txt` are installed.
*   **Dataset Requirements**: The complete 10-task evaluation dataset is located at `/Users/krithigamahadevan/capstone_experiments/data/evaluation_tasks.json`.
*   **Configuration Requirements**: The diagnostic runner dynamically reads the configurations from `capstone_experiments/config/optimized_config.json` via the process-level `CAPSTONE_CONFIG_PATH` environment variable.

### 7.2. Execution Commands
Navigate to the experiments directory:
```bash
cd capstone_experiments
```

Run the diagnostic suite in different modes:
*   **Run All Diagnostics**:
    ```bash
    python3 run_diagnostics.py --mode all
    ```
*   **Run Leave-One-Task-Out Generalization Testing**:
    ```bash
    python3 run_diagnostics.py --mode generalization
    ```
*   **Run Controlled Paired Comparison**:
    ```bash
    python3 run_diagnostics.py --mode paired
    ```
*   **Run Fault Injection Suite**:
    ```bash
    python3 run_diagnostics.py --mode fault
    ```

### 7.3. Expected Outputs
Diagnostic runs generate structured JSON reports inside `capstone_experiments/results/diagnostics/` named as `diagnostics_run_%Y%m%d_%H%M%S.json` containing:
*   Run metadata and dynamic configuration snapshots.
*   Fold-by-fold metrics (Precision@5, Recall@5, F1, MRR, latency, RSS peak memory) for generalization testing.
*   Task-by-task performance deltas (absolute and percentage differences) for the controlled paired comparison.
*   Observed behaviors, status records, and exception details for all fault test cases (`FLT-001` through `FLT-005`).

### 7.4. Troubleshooting Guidance
*   **Model Loading Cold-Starts**: The first task execution will load the SentenceTransformer embedding model `all-mpnet-base-v2` from Hugging Face into memory. A warm-up retrieval call is executed before collecting measurements to ensure that model download and graph builder cold-start initialization times are excluded from task-level latency calculations.
*   **Missing Dataset Path**: If the configured dataset path does not exist, the runner automatically attempts to fallback to a valid local workspace checkout path.
*   **Missing target_files Key**: In malformed task input scenarios where the ground-truth target file list is missing, retrieval continues normally, but the diagnostic suite records that retrieval-quality metrics cannot be calculated.

### 7.5. Reproducibility Limitations
*   **Gemini Generation Variance**: A temperature of `0.0` does not guarantee bit-wise identical Gemini text generation outputs due to remote backend execution path variations.
*   **Seed Management**: No explicit random seeds are initialized by `run_diagnostics.py` (since local RAG indexing operates via deterministic vector calculations).
*   **Embedding Variations**: Cosine similarity calculations and embedding values may vary slightly depending on processor architecture (CPU/GPU) or PyTorch/SentenceTransformer package versions.
*   **Memory RSS Measurements**: `ru_maxrss` measures process-level cumulative peak memory rather than strategy-isolated footprint.
*   **Dataset Location**: The complete 10-task dataset is external to the active repository clone unless manually copied in.
*   **Configuration State**: `CAPSTONE_CONFIG_PATH` relies on process-level environment configuration state.
*   **Independence**: The 10-task evaluation set is not an independent held-out set if some tasks participated in the initial Week 4 optimization loop.

