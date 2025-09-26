from fastapi import FastAPI, HTTPException
import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pydantic import BaseModel
from transformers import DistilBertTokenizer, DistilBertModel
import dvc.api
import io

app = FastAPI()

# Define model architecture (same as train_model.py)
class RiskModel(torch.nn.Module):
    def __init__(self, input_size=11):  # Adjust input_size based on features
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
with dvc.api.open(model_path, mode='rb') as f:
    model_state = torch.load(io.BytesIO(f.read()), map_location=torch.device('cpu'))
model = RiskModel()
model.load_state_dict(model_state)
model.eval()

# Load DistilBERT for GenAI
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Define input schema
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

# Preprocess for PyTorch model
def preprocess_transaction(data: dict):
    df = pd.DataFrame([data])
    df['log_amount'] = np.log(df['amount'] + 1)
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    categorical_cols = ['sender_country', 'receiver_country', 'currency', 'remittance_purpose', 
                       'payment_method', 'transaction_status', 'bank', 'agent', 'transaction_type']
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return torch.tensor(df[['log_amount', 'hour'] + categorical_cols].values, dtype=torch.float32)

# GenAI risk assessment with DistilBERT
def agentic_risk_assessment(transaction: dict):
    context = f"Sender: {transaction['sender_country']}, Receiver: {transaction['receiver_country']}, " \
              f"Amount: {transaction['amount']}, Agent: {transaction['agent']}, Purpose: {transaction['remittance_purpose']}"
    inputs = tokenizer(context, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = bert(**inputs)
    risk_score = outputs.last_hidden_state.mean(dim=1).numpy().mean()
    return risk_score

@app.post("/predict")
async def predict(transaction: Transaction):
    try:
        # PyTorch prediction
        X = preprocess_transaction(transaction.dict())
        with torch.no_grad():
            pred = model(X).numpy()
        pytorch_risk = {"risk_flag": bool(pred[0] > 0.5), "probability": float(pred[0])}
        
        # GenAI assessment
        genai_risk = agentic_risk_assessment(transaction.dict())
        
        # Combine results
        combined_risk = 0.7 * pytorch_risk['probability'] + 0.3 * genai_risk
        
        return {
            "risk_flag": bool(combined_risk > 0.5),
            "pytorch_probability": pytorch_risk['probability'],
            "genai_risk_score": float(genai_risk),
            "combined_probability": float(combined_risk)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))