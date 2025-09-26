# Agentic AI Risk Agent for Cross-Border Remittances

This project develops an autonomous AI agent for real-time risk assessment in cross-border remittances, tailored for the UAE’s $50B remittance market (e.g., DIFC ecosystem).  

It integrates GenAI for dynamic decision-making and a MLOps pipeline for scalability and reproducibility. The agent queries mock FX APIs and uses Bayesian uncertainty modeling to enhance AML compliance.

## Features

- **GenAI Agent**: Autonomous agent queries mock FX APIs for dynamic risk decisions, improving on static AML rules.
- **Synthetic Dataset**: 100K transactions generated with Faker, reflecting UAE’s diverse remittance corridors (India, Pakistan, Nigeria, etc.).
- **MLOps Pipeline**: Data versioning with DVC, cloud storage in Supabase, experiment tracking with MLflow, and workflow orchestration with Prefect.H2O.ai for risk modeling, FastAPI UI for real-time risk scoring.

## Tech Stack
- **Languages**: Python 3.8+
- **ML/GenAI**: H2O.ai, Hugging Face (DistilBERT), PyMC3 (Bayesian uncertainty), Pytorch
- **MLOps**:
  - **DVC**: Versions synthetic remittance datasets.
  - **Supabase Storage**: Cloud storage for dataset.
  - **MLflow/W&B**: Tracks model experiments.
  - **Prefect**: Schedules agent runs.
  - **FastAPI**: API for real-time risk scoring (in progress).
- **Data**: Synthetic data generated with Faker.

## State
- AutoML resulted with 0.62 AUC
- Trained PyTorch model with SMOTE (AUC ~0.79), interactive training via tqdm, and MLflow experiment tracking with model signature.
- Model versioned with DVC and stored in Supabase
- Deployed FastAPI endpoint (`/predict`) with PyTorch (AUC ~0.79) and DistilBERT for agentic risk assessment, fetching model from Supabase via DVC.

## Setup Instructions
1. Clone the repo:
   ```bash
   git clone https://github.com/tomthomas/ai-agent-remittance-risk-banking.git
   cd ai-agent-remittance-risk-banking


2. Install dependencies:
    ```bash
    pip install -r requirements.txt

3. Initialize DVC and pull data from Supabase:
    ```bash
    dvc init
    dvc remote add -d supabase s3://remittances
    dvc remote modify supabase endpointurl https://wtkgphpsdjxwdlxwmyoh.supabase.co/storage/v1/s3
    dvc remote modify supabase access_key_id test_v1
    dvc remote modify supabase secret_access_key <your-s3-secret-access-key>
    dvc pull