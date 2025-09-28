from prefect import flow, task
import httpx
import pandas as pd
from datetime import datetime
import mlflow
import os
from dotenv import load_dotenv
from supabase import create_client, Client
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
if not API_URL:
    raise ValueError("API_URL must be set in .env")

# Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Set MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

@task(retries=3, retry_delay_seconds=60)
def fetch_new_transactions():
    try:
        # Query Supabase remittances table for a recent transaction
        response = supabase.table('remittances').select('*').eq('transaction_status', 'Completed').is_('risk_flag', None).limit(1).execute()
        if not response.data:
            logger.warning("No unprocessed transactions found, using fallback synthetic transaction")
            # Fallback to synthetic transaction
            return {
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
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        transaction = response.data[0]
        logger.info(f"Fetched transaction: {transaction}")
        return transaction
    except Exception as e:
        logger.error(f"Error fetching transactions from Supabase: {str(e)}")
        raise

@task(retries=3, retry_delay_seconds=60)
def predict_risk(transaction):
    try:
        with httpx.Client() as client:
            response = client.post(f"{API_URL}/predict", json=transaction, timeout=30)
            response.raise_for_status()
            result = response.json()
        logger.info(f"Prediction result: {result}")
        # Update Supabase with prediction results
        supabase.table('remittances').update({
            "risk_flag": result["risk_flag"],
            "probability": result["combined_probability"]
        }).eq('id', transaction['id']).execute()
        return result
    except httpx.HTTPError as e:
        logger.error(f"Error calling FastAPI: {str(e)}")
        raise

@task
def log_results(result, transaction):
    try:
        with mlflow.start_run():
            mlflow.log_param("transaction_timestamp", datetime.now().isoformat())
            mlflow.log_param("sender_country", transaction["sender_country"])
            mlflow.log_param("receiver_country", transaction["receiver_country"])
            mlflow.log_param("amount", transaction["amount"])
            mlflow.log_metric("combined_probability", result["combined_probability"])
            mlflow.log_metric("pytorch_probability", result["pytorch_probability"])
            mlflow.log_metric("genai_risk_score", result["genai_risk_score"])
            mlflow.log_param("risk_flag", result["risk_flag"])
        logger.info("Logged results to MLflow")
    except Exception as e:
        logger.error(f"Error logging to MLflow: {str(e)}")
        raise

@flow(name="Daily Remittance Risk Assessment")
def daily_risk_assessment():
    try:
        transaction = fetch_new_transactions()
        result = predict_risk(transaction)
        log_results(result, transaction)
        logger.info(f"Processed transaction: Risk Flag = {result['risk_flag']}, Probability = {result['combined_probability']:.4f}")
    except Exception as e:
        logger.error(f"Flow failed: {str(e)}")
        raise

if __name__ == "__main__":
    daily_risk_assessment.serve(name="daily-risk-check", cron="0 0 * * *")  # Run daily at midnight