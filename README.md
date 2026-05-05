# Bird-Pandas: SQL vs. Pandas Benchmark

Bird-Pandas bridges the gap between declarative SQL and procedural Python in data analytics by providing a consistent evaluation baseline. This project delivers a rigorously aligned dataset with verified Pandas solutions derived from the BIRD benchmark, integrated with the Logic Completion Framework (LCF) to resolve natural language ambiguity. The codebase includes a complete pipeline for code generation and semantic execution validation.

## 📌 Introduction

While Text-to-SQL remains the dominant approach for database interaction, real-world analytics increasingly require the flexibility of general-purpose programming languages such as Python or Pandas. **Bird-Pandas** is a benchmark designed for cross-paradigm evaluation between SQL and Pandas in data analysis tasks.

We systematically refined the original BIRD benchmark to reduce annotation noise and align execution semantics, establishing a consistent baseline. Our work investigates the paradigmatic divergence between SQL's declarative structure and Python's explicit procedural logic. To address the sensitivity of Pandas generation to underspecified user intent, we introduce the **Logic Completion Framework (LCF)**, which resolves ambiguity by incorporating latent domain knowledge.

This repository provides:
1.  **Aligned SQL-Python Dataset**: A refined version of the BIRD development set with verified Python solutions (using Pandas).
2.  **Generation & Verification Pipeline**: Tools to generate Python/SQL code and semantically validate execution results.
3.  **Logic Completion Framework**: Implementation of LCF to bridge the reasoning gap caused by missing domain context.

## 📂 Dataset Description

The core contribution of this repository is the aligned SQL-Python dataset. The data files are located in the `Bird-Pandas/` directory:

### 1. `Bird-Pandas/Origin_dev_Bird.json`
*   **Description:** This is the original development set from the BIRD benchmark.
*   **Content:** It serves as the baseline input, containing natural language questions, gold SQL queries, evidence (external knowledge), and difficulty levels.
*   **Fields:** `question_id`, `db_id`, `question`, `evidence`, `SQL`, `difficulty`.

### 2. `Bird-Pandas/Verified_Bird_Python.json`
*   **Description:** This is our **enhanced dataset** containing the Pandas code annotations.
*   **Content:** It includes all the information from the original BIRD dev set, augmented with verified Pandas code solutions that yield the same execution results as the Gold SQL.
*   **Methodology:** The Pandas code has been generated and rigorously verified against the database to ensure execution accuracy and logic alignment with the original SQL.

## 🚀 Project Structure

```text
.
├── Bird-Pandas/
│   ├── dev_databases/              # Original BIRD SQL databases (SQLite)
│   ├── excel_database/             # Converted CSV datasets for Python/Pandas analysis
│   ├── LCP_Enhanced_Bird/          # Enhanced dataset with explicit constraints (addressing info deficits)
│   ├── LLM-based Evaluation/       # LLM-based Semantic Validator Scripts
│   ├── Logic Completion Framework/ # LCF prompt templates and Qwen3-Max generated logic
│   ├── Text2Pandas/                # Test scripts and prompts for Text-to-Pandas
│   ├── Text2SQL/                   # Test scripts and prompts for Text-to-SQL
│   ├── Origin_dev_Bird.json        # Original BIRD development set
│   ├── Verified_Bird_Pandas.json   # Verified dataset containing SQL and Pandas ground truths
│   ├── Convert_SQLite_to_CSV.py    # Convert the SQLite database to a CSV file
│   └── README.md                       # Project documentation
└── 
```

## 🔧 Configuration & Setup

Before running the evaluation scripts, you must configure your API keys and verify file paths.

### 1. Data Preparation

To ensure consistent experimental conditions, you first need to download the original BIRD development dataset and convert it into the CSV format used for Python analysis.

1.  **Download BIRD Dev Set**:
    Download the development set (`dev.zip`) from the [BIRD Benchmark website](https://bird-bench.github.io/) and unzip it into the `Bird-Pandas/dev_databases/` directory.

2.  **Convert SQLite to CSV**:
    Run the conversion script to generate the rigorous CSV dataset (preserving float precision and type hints):
    ```bash
    python Bird-Pandas/Convert_SQLite_to_CSV.py
    ```
    This will populate the `Bird-Pandas/excel_database/` directory with CSV files required for the Text-to-Python pipeline.

### 2. API Key Configuration
    Navigate to the following files and replace `"YOUR_API_KEY"` with your actual `dashscope` or compatible API key:
    *   `Bird-Pandas/LLM-based Evaluation/evaluation.py`
    *   `Bird-Pandas/Logic Completion Framework/LCP_CODE_Test.py`
    *   `Bird-Pandas/Logic Completion Framework/LCP_SQL_Test.py`
    *   `Bird-Pandas/Text2Pandas/Python-test.py`
    *   `Bird-Pandas/Text2SQL/SQL-test.py`

2.  **File Path Verification**:
    The scripts use relative paths assuming the default directory structure. If you move files, ensure you update the `eval_path`, `db_root_path`, and other path variables in the `if __name__ == '__main__':` section of the respective scripts.

## ⚙️ Generation & Verification

We provide a comprehensive pipeline for generating and verifying Python solutions, designed to facilitate reproducible research. **Note that all experimental parameters, including file paths and model configurations, are hardcoded within the scripts.**

### 1. Code Generation
*   **Script:** `Bird-Pandas/Text2Pandas/Python-test.py` or `Bird-Pandas/Text2SQL/SQL-test.py`
*   **Function:** Generates Python code or to SQL queries in the dataset.
*   **Details:** The script uses a retrieval-augmented generation approach (if knowledge is enabled) to produce Python logic. The output directory and model parameters are defined in the `__main__` block.

### 2. Evaluation
*   **Script:** `Bird-Pandas/LLM-based Evaluation/evaluation.py`
*   **Function:** Semantically validates the generated code by comparing its execution results against the ground truth.
*   **Details:** This module executes the generated Python code and compares the resulting data structures with the verified ground truth from `Bird-Pandas/Verified_Bird_Python.json`. An LLM-based validator is employed to determine equivalence, robustly handling format variations (e.g., list vs. tuple, float precision). Users should ensure the `PREDICTED_CODE_PATH` in this script matches the output location from the generation step.

## 📄 Citation

Please cite our paper if you use this code or dataset in your work (citation will be updated upon publication):

```bibtex
@inproceedings{BirdPandas2026,
  title={BIRD-Pandas: Diagnosing the Cross-Paradigm Divergence Between Text-to-SQL and Text-to-Pandas},
  author={Anonymous Authors},
  booktitle={Under Review at NeurIPS},
  year={2026}
}
```

## 📝 License

[Insert License Name, e.g., MIT, CC-BY-4.0]

