# ai-project-template

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3110/)

This repository serves as an AI project template, offering a pre-configured GitHub repository structure with environment dependencies and sample code to help students quickly get started.

## 🚀 Features Included

- **Config Directory**: Centralized configuration files for the project.
- **Data Directory**: Structured data management (raw, processed, and database data).
- **Notebooks Directory**: Jupyter notebooks for data analysis and experiments.
- **Results Directory**: Store experiment results, logs, and models.
- **Source Code Directory**: Modularized source code for models, pipelines, utilities, schemas, and more.
- **Tests Directory**: Unit and integration tests using `pytest`.
- **[pre-commit](https://pre-commit.com/)**: Configuration for linting, type checking, and secret detection.
- **[uv](https://docs.astral.sh/uv/)**: An extremely fast Python package and project manager
- **[pyproject.toml](https://pip.pypa.io/en/stable/reference/build-system/pyproject-toml/)**: Config for project, ruff linter, formatter, type checking, and testing.

> [!NOTE]
> Not every element from this template is required for every project. Please adjust the repository structure and configuration files to fit your project's needs.

---

## 📁 Detailed Directory Breakdown

- **`/.github/`**: GitHub-specific configuration and workflows.
  - `dependabot.yaml`: Configuration for Dependabot to automatically update dependencies.

- **`/config/`**: Store all configuration files needed for your project.
  - Example: `config.yaml` can be used for setting up hyperparameters, API keys, or other environment variables.

- **`/data/`**: Structured data folders for different stages of the project.
  - `/raw/`: Raw data straight from the source.
  - `/processed/`: Cleaned and preprocessed data ready for use in models.
  - `/database/`: Database files if using local storage solutions.


- **`/notebooks/`**: Jupyter notebooks for performing Exploratory Data Analysis (EDA), experimenting with models, and reporting.
  - Example: `00_example.ipynb` shows how a notebook looks in the project.

- **`/results/`**: Store experiment results, logs, and models.

- **`/src/`**: Main source code for your project.
  - `/constants/`: Store project-wide constants (e.g., file paths, API endpoints).
  - `/models/`: Machine learning models (e.g., neural networks, decision trees) scripts, classes, and functions.
  - `/schemas/`: Data schemas and validation logic (e.g., Pydantic models).
  - `/pipelines/`: Data and model pipelines.
  - `/utils/`: Utility functions and helpers (e.g., data loaders, preprocessing functions).
  - `main.py`: Main execution script for the project.
- **`/tests/`**: Unit and integration tests.

---

## ⚙️ Configuration Files

- **`.env`**: Store environment variables like API keys, database credentials, and sensitive information.

- **`.gitignore`**: Standard [`.gitignore`](https://git-scm.com/docs/gitignore) file to exclude unnecessary files (e.g., environment files, data files).

- **`.pre-commit-config.yaml`**: Configuration for [pre-commit](https://pre-commit.com/) hooks for linting, formatting, and type checking.


- **`pyproject.toml`**: Configuration of the project, plus the configuration for formatting and linting. This file is used by [uv](https://docs.astral.sh/uv/) to manage the project and [ruff](https://docs.astral.sh/ruff/) for linting and formatting.


---

## 🛠️ How to Use

### 1. **Clone the Repository**
```bash
git clone https://github.com/username/ai-project-template.git
cd ai-project-template
```

### 2a. **Install Default Dependencies** (includes ddacs)
```bash
uv sync --all-extras --dev
```

### 2b. **Install your Project Specific Dependencies**
```bash
uv add <package_name>
```

---

## 🗂️ Repository Structure
```bash

├── .github                   # GitHub Actions workflows and configuration
│   ├── dependabot.yaml       # Dependabot configuration for automated dependency updates
│   └── workflows             # CI/CD workflow definitions
├── .venv                     # Virtual environment (created after installation)
├── config                    # Configuration files for your project
│   └── config.yaml           # Example configuration file
├── data                      # Folder to store raw and processed data
│   ├── database              # Databases or local data storage
│   ├── processed             # Preprocessed/cleaned data
│   └── raw                   # Raw data inputs
├── notebooks                 # Jupyter notebooks for exploratory data analysis, experiments
│   └── 00_example.ipynb      # Example notebook
├── results                   # Folder to store the results of experiments and models
├── src                       # Source code of your project
│   ├── constants             # Constants used in the project
│   │   └── __init__.py       # Package initialization file
│   ├── models                # Machine learning model scripts
│   │   └── __init__.py       # Package initialization file
│   ├── pipelines             # ML pipelines for preprocessing and modeling
│   │   └── __init__.py       # Package initialization file
│   ├── schemas               # Data schemas and validation logic
│   │   └── __init__.py       # Package initialization file
│   ├── utils                 # Utility functions
│   │   └── __init__.py       # Package initialization file
│   └── main.py               # Main execution script
├── tests                     # Unit and integration tests
│   └── test_example.py       # Example test file using pytest
├── .env                      # Environment variables file
├── .gitignore                # Standard .gitignore file
├── .pre-commit-config.yaml   # Configuration for pre-commit hooks
├── pyproject.toml            # Configuration for formatting, linting, type-checking, and testing
└── README.md                 # Documentation for the project (you're reading it!)

```



