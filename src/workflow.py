from prefect import flow, task
import httpx
import pandas as pd
from datetime import datetime
import mlflow

@task
def fetch_new_transactions():
    # Simulated new transactions (replace with real data source, e.g., Supabase query)
    data = {
        "sender_country": "UAE",
        "receiver_country": "Nigeria",
        "amount": 25000.0,
        "currency": "NGN",
        "remittance_purpose": "Family Support",
        "payment_method": "Bank Transfer",
        "transaction_status": "Completed",
        "bank": "Emirates NBD",
        "agent": "Western Union",
        "transaction_type": "Online",
        "timestamp": "2025-09-26 14:30:00"
    }
    return data

@task
def predict_risk(transaction):
    # Call FastAPI endpoint
    with httpx.Client() as client:
        response = client.post("http://localhost:8000/predict", json=transaction)
        response.raise_for_status()
        return response.json()

@task
def log_results(result):
    # Log to MLflow
    with mlflow.start_run():
        mlflow.log_param("transaction_timestamp", datetime.now().isoformat())
        mlflow.log_metric("combined_probability", result["combined_probability"])
        mlflow.log_metric("pytorch_probability", result["pytorch_probability"])
        mlflow.log_metric("genai_risk_score", result["genai_risk_score"])
        mlflow.log_param("risk_flag", result["risk_flag"])

@flow(name="Daily Remittance Risk Assessment")
def daily_risk_assessment():
    transaction = fetch_new_transactions()
    result = predict_risk(transaction)
    log_results(result)
    print(f"Processed transaction: Risk Flag = {result['risk_flag']}, Probability = {result['combined_probability']:.4f}")

if __name__ == "__main__":
    daily_risk_assessment.serve(name="daily-risk-check", cron="0 0 * * *")  # Run daily at midnight