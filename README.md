# Agentic AI Risk Agent for Cross-Border Remittances

This project develops an autonomous AI agent for real-time risk assessment in cross-border remittances, tailored for the UAE’s $50B remittance market (e.g., DIFC ecosystem).  

It integrates GenAI for dynamic decision-making and a MLOps pipeline for scalability and reproducibility. The agent queries mock FX APIs and uses Bayesian uncertainty modeling to enhance AML compliance.

## Features

GenAI Agent: Autonomous agent queries mock FX APIs to make dynamic risk decisions, aiming to improve on static AML rules.


## Tech Stack

**Languages**: Python

**ML/GenAI**: H2O.ai, Hugging Face (DistilBERT), PyMC3 (Bayesian uncertainty)

## MLOps Pipeline:

**Prefect**: Schedules agent runs for automated workflows.

**DVC**: Versions synthetic remittance datasets for reproducibility.

**MLflow/W&B**: Tracks model experiments and hyperparameters.

**H2O.ai**: Baselines risk models for robust performance.

**FastAPI** for Demo: API for real-time transaction risk scoring (in progress).

**Data**: Synthetic remittance data generated with Faker
