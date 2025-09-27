from fastapi import FastAPI, HTTPException
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pydantic import BaseModel
from transformers import DistilBertTokenizer, DistilBertModel
import dvc.api
import io
import re
import json
from openai import OpenAI
import os
from dotenv import load_dotenv
import httpx
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env file")

app = FastAPI()

# OpenRouter client
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

# Define model architecture
class RiskModel(torch.nn.Module):
    def __init__(self, input_size=11):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, 256)
        self.dropout1 = torch.nn.Dropout(0.3)
        self.fc2 = torch.nn.Linear(256, 128)
        self.dropout2 = torch.nn.Dropout(0.3)
        self.fc3 = torch.nn.Linear(128, 64)
        self.dropout3 = torch.nn.Dropout(0.3)
        self.fc4 = torch.nn.Linear(64, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.sigmoid(self.fc4(x))
        return x

# Load model from Supabase via DVC
model_path = 'models/pytorch_model.pth'
try:
    with dvc.api.open(model_path, mode='rb') as f:
        model_state = torch.load(io.BytesIO(f.read()), map_location=torch.device('cpu'))
    model = RiskModel()
    model.load_state_dict(model_state)
    model.eval()
except Exception as e:
    raise Exception(f"Failed to load model: {str(e)}")

# Load DistilBERT
try:
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
except Exception as e:
    raise Exception(f"Failed to load DistilBERT: {str(e)}")

# Input schemas
class Transaction(BaseModel):
    sender_country: str
    receiver_country: str
    amount: float
    currency: str
    remittance_purpose: str
    payment_method: str
    transaction_status: str
    bank: str
    agent: str
    transaction_type: str
    timestamp: str

class TextInput(BaseModel):
    text: str
    history: list = None  # Optional history for multi-turn

# Preprocess for PyTorch model
def preprocess_transaction(data: dict):
    try:
        df = pd.DataFrame([data])
        if df['amount'].iloc[0] < 0:
            raise ValueError("Amount cannot be negative")
        df['log_amount'] = np.log(df['amount'] + 1)
        df['hour'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.hour
        if df['hour'].isna().any():
            raise ValueError("Invalid timestamp format")
        categorical_cols = ['sender_country', 'receiver_country', 'currency', 'remittance_purpose', 
                           'payment_method', 'transaction_status', 'bank', 'agent', 'transaction_type']
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        X = torch.tensor(df[['log_amount', 'hour'] + categorical_cols].values, dtype=torch.float32)
        if torch.isnan(X).any():
            raise ValueError("NaN values detected in preprocessed data")
        return X
    except Exception as e:
        raise Exception(f"Preprocessing error: {str(e)}")

# Parse LLM output to transaction
def parse_llm_to_transaction(llm_output: str):
    try:
        # Extract JSON from text (e.g., ```json {...} ``` or plain JSON)
        json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            # Fallback for conversational responses
            if "didn't understand" in llm_output.lower():
                return {"response": "Sorry, I didn't understand your input. Please include a country and amount (e.g., 'UAE to Nigeria, $25000').", "valid": False}
            if "specify the amount" in llm_output.lower():
                return {"response": "Please specify the amount and purpose (e.g., '$25000, Family Support').", "valid": False}
            if "illegal" in llm_output.lower():
                return {"response": "I'm sorry, but I can't assist with or provide advice on illegal activities. If you have a legitimate remittance query, feel free to ask.", "valid": False}
            return {"response": "Sorry, I couldn't parse the response. Please try again with a clearer input.", "valid": False}
        
        if not result.get('valid', False):
            return result
        transaction = result.get('transaction', {})
        defaults = {
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
            "timestamp": "2025-09-27 14:15:00"
        }
        defaults.update(transaction)
        return {"response": None, "valid": True, "transaction": defaults}
    except Exception as e:
        logger.error(f"Error parsing LLM output: {str(e)}")
        return {"response": f"Error parsing LLM output: {str(e)}", "valid": False}

# GenAI risk assessment with DistilBERT
def agentic_risk_assessment(transaction: dict):
    try:
        context = f"Sender: {transaction['sender_country']}, Receiver: {transaction['receiver_country']}, " \
                  f"Amount: {transaction['amount']}, Agent: {transaction['agent']}, Purpose: {transaction['remittance_purpose']}"
        inputs = tokenizer(context, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = bert(**inputs)
        risk_score = outputs.last_hidden_state.mean(dim=1).numpy()
        if np.isnan(risk_score).any():
            raise ValueError("NaN values in DistilBERT output")
        return float(risk_score.mean())
    except Exception as e:
        logger.error(f"DistilBERT error: {str(e)}")
        raise Exception(f"DistilBERT error: {str(e)}")

# Retry wrapper for OpenRouter requests
def with_retry(func):
    def wrapper(*args, **kwargs):
        for attempt in range(3):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < 2:
                    time.sleep(5)  # Wait 5s before retry
                else:
                    raise
    return wrapper

@app.post("/predict")
async def predict(transaction: Transaction):
    try:
        X = preprocess_transaction(transaction.dict())
        with torch.no_grad():
            pred = model(X).numpy()
        pytorch_risk = {"risk_flag": bool(pred[0] > 0.5), "probability": float(pred[0])}
        genai_risk = agentic_risk_assessment(transaction.dict())
        combined_risk = 0.7 * pytorch_risk['probability'] + 0.3 * genai_risk
        if np.isnan(combined_risk):
            raise ValueError("NaN in combined risk score")
        return {
            "risk_flag": bool(combined_risk > 0.5),
            "pytorch_probability": pytorch_risk['probability'],
            "genai_risk_score": float(genai_risk),
            "combined_probability": float(combined_risk)
        }
    except Exception as e:
        logger.error(f"Predict error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Predict error: {str(e)}")

@app.post("/predict-text")
async def predict_text(input: TextInput):
    try:
        prompt = f"""
        You are a remittance risk assessment agent. Extract transaction details from: '{input.text}'.
        Always return valid JSON in this format:
        - Invalid input: {{'response': 'Sorry, I didn't understand your input. Please include a country and amount (e.g., "UAE to Nigeria, $25000").', 'valid': false}}
        - Partial input (e.g., missing amount): {{'response': 'Please specify the amount and purpose (e.g., "$25000, Family Support").', 'valid': false}}
        - Illegal purpose: {{'response': "I'm sorry, but I can't assist with or provide advice on illegal activities. If you have a legitimate remittance query, feel free to ask.", 'valid': false}}
        - Valid input: {{'valid': true, 'transaction': {{'sender_country': str, 'receiver_country': str, 'amount': float, 'currency': str, 'remittance_purpose': str, 'payment_method': str, 'transaction_status': str, 'bank': str, 'agent': str, 'transaction_type': str, 'timestamp': str}}}}
        Use history to fill missing details if possible: {input.history or []}
        """
        messages = [
            {"role": "system", "content": "You are a precise remittance risk assessment agent. Always return valid JSON as specified, even for invalid inputs."}
        ]
        valid_history = [msg for msg in (input.history or []) if isinstance(msg, dict) and "role" in msg and "content" in msg and isinstance(msg["content"], str)]
        messages.extend(valid_history)
        messages.append({"role": "user", "content": prompt})

        @with_retry
        def make_request():
            return openrouter_client.chat.completions.create(
                model="x-ai/grok-4-fast:free",
                messages=messages
            )

        response = make_request()
        llm_result = response.choices[0].message.content
        logger.info(f"LLM output: {llm_result}")
        result = parse_llm_to_transaction(llm_result)
        if not result["valid"]:
            return {"response": result["response"]}
        transaction = result["transaction"]
        X = preprocess_transaction(transaction)
        with torch.no_grad():
            pred = model(X).numpy()
        pytorch_risk = {"risk_flag": bool(pred[0] > 0.5), "probability": float(pred[0])}
        genai_risk = agentic_risk_assessment(transaction)
        combined_risk = 0.7 * pytorch_risk['probability'] + 0.3 * genai_risk
        if np.isnan(combined_risk):
            raise ValueError("NaN in combined risk score")
        return {
            "risk_flag": bool(combined_risk > 0.5),
            "probability": float(combined_risk)
        }
    except Exception as e:
        logger.error(f"Predict-text error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Predict-text error: {str(e)}")

@app.post("/chat")
async def chat(input: TextInput):
    try:
        # Build message history
        messages = [
            {"role": "system", "content": """You are a friendly remittance risk assessment agent. Respond naturally and conversationally, using history to track partial inputs. Always return valid JSON:
            - Invalid: {"response": "Sorry, I didn't understand your input. Please include a country and amount (e.g., 'UAE to Nigeria, $25000').", "valid": false}
            - Partial (e.g., missing amount): {"response": "Please specify the amount and purpose (e.g., '$25000, Family Support').", "valid": false}
            - Illegal purpose: {"response": "I'm sorry, but I can't assist with or provide advice on illegal activities. If you have a legitimate remittance query, feel free to ask.", "valid": false}
            - Valid: {"response": "Hey, for that transfer from {sender} to {receiver} of ${amount:.2f} for {purpose}, there's a {risk:.1f}% chance it could be fraudulent. Want me to check anything else?", "valid": true, "transaction": {"sender_country": str, "receiver_country": str, "amount": float, "currency": str, "remittance_purpose": str, "payment_method": str, "transaction_status": str, "bank": str, "agent": str, "transaction_type": str, "timestamp": str}}
            Use history to fill missing details if possible."""}
        ]
        valid_history = [msg for msg in (input.history or []) if isinstance(msg, dict) and "role" in msg and "content" in msg and isinstance(msg["content"], str)]
        messages.extend(valid_history)
        messages.append({"role": "user", "content": str(input.text)})

        @with_retry
        def make_request():
            return openrouter_client.chat.completions.create(
                model="x-ai/grok-4-fast:free",
                messages=messages
            )

        response = make_request()
        llm_result = response.choices[0].message.content
        logger.info(f"LLM output: {llm_result}")
        result = parse_llm_to_transaction(llm_result)
        if not result["valid"]:
            return {"response": result["response"]}
        transaction = result["transaction"]
        X = preprocess_transaction(transaction)
        with torch.no_grad():
            pred = model(X).numpy()
        pytorch_risk = {"risk_flag": bool(pred[0] > 0.5), "probability": float(pred[0])}
        genai_risk = agentic_risk_assessment(transaction)
        combined_risk = 0.7 * pytorch_risk['probability'] + 0.3 * genai_risk
        if np.isnan(combined_risk):
            raise ValueError("NaN in combined risk score")
        response_text = f"Hey, for that transfer from {transaction['sender_country']} to {transaction['receiver_country']} of ${transaction['amount']:.2f} for {transaction['remittance_purpose'].lower()}, there's a {combined_risk*100:.1f}% chance it could be fraudulent. Want me to check anything else?"
        return {
            "response": response_text,
            "risk_flag": bool(combined_risk > 0.5),
            "probability": float(combined_risk)
        }
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Remittance Risk Agent API"}