
# Fraud Detection System

A production-oriented machine learning project for detecting potentially fraudulent financial transactions using supervised binary classification.

The project follows an end-to-end MLOps workflow, covering data validation, feature engineering, model development, experiment tracking, API development, containerization, automated testing, deployment, and monitoring.

## Project Objective

Develop a fraud detection system that classifies transactions as:

- `0` – Legitimate transaction
- `1` – Fraudulent transaction

The objective is to build a reproducible and maintainable machine learning application that can be tested, deployed, monitored, and improved over time.

## Dataset

The project uses a CSV dataset containing 51,000 transaction records and the following features:

| Feature | Description |
|---|---|
| Transaction_ID | Transaction identifier |
| User_ID | User identifier |
| Transaction_Amount | Transaction amount |
| Transaction_Type | Type of transaction |
| Time_of_Transaction | Transaction hour (0–23) |
| Device_Used | Device used |
| Location | Transaction location |
| Previous_Fraudulent_Transactions | Previous fraudulent transaction count |
| Account_Age | Account age in days |
| Number_of_Transactions_Last_24H | Transaction frequency |
| Payment_Method | Payment method |
| Fraudulent | Binary target variable |

Initial data inspection identified repeated transaction IDs and missing or invalid values. These issues will be investigated and addressed during data preparation.
> **Dataset access:** The raw dataset is intentionally excluded from GitHub.
> Place the dataset locally at `data/raw/fraud_detection.csv` before running
> the project tests and data pipeline.

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Pytest
- MLflow
- FastAPI
- Docker
- Git and GitHub
- GitHub Actions
- Render
- Kubernetes

Technologies will be introduced progressively as the project develops.

## Project Structure

```text
fraud-detection/
├── data/
│   └── raw/
│       └── fraud_detection.csv
├── src/
│   └── data/
│       ├── __init__.py
│       └── loader.py
├── tests/
│   ├── __init__.py
│   └── test_data_loader.py
├── notebooks/
├── requirements.txt
├── pytest.ini
├── .gitignore
└── README.md
```

## Current Development Status

### Phase 1: Data and Project Foundation

- [x] Project structure established
- [x] Python virtual environment configured
- [x] Dataset loading implemented
- [x] Dataset validation implemented
- [x] Automated tests implemented
- [x] Initial duplicate-record investigation completed
- [ ] GitHub repository connected

### Planned Development Phases

1. Data cleaning, feature engineering, and preprocessing
2. Model development and evaluation
3. MLflow experiment tracking and model registry
4. Prediction API development using FastAPI
5. Docker containerization
6. GitHub Actions and CI/CD
7. Cloud deployment using Render
8. Model and API monitoring
9. Kubernetes orchestration
10. Continuous improvement and retraining

## Running the Project Locally

### 1. Clone the repository

```bash
git clone <repository-url>
cd fraud-detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows:

```bat
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run automated tests

```bash
pytest -v
```

## Testing

Pytest is used throughout development to validate functionality and help prevent regressions.

Current Phase 1 test coverage includes dataset loading, required columns, target validation, transaction time, account age, and duplicate-record checks.

## Project Philosophy

The system is developed incrementally, with each feature implemented and tested before the next development phase begins.
