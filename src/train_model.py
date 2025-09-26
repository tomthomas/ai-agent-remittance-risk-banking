import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature
from tqdm import tqdm

# Load and preprocess data
df = pd.read_csv('data/remittance_data.csv')
df['log_amount'] = np.log(df['amount'] + 1)
df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
categorical_cols = ['sender_country', 'receiver_country', 'currency', 'remittance_purpose', 
                    'payment_method', 'transaction_status', 'bank', 'agent', 'transaction_type']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df[['log_amount', 'hour'] + categorical_cols]
y = df['risk_flag']

# Apply SMOTE to balance classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# PyTorch Dataset
class RemittanceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = RemittanceDataset(X_train, y_train)
test_dataset = RemittanceDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# PyTorch Model
class RiskModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.sigmoid(self.fc4(x))
        return x

input_size = X.shape[1]
model = RiskModel(input_size)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Training with tqdm progress bar
with mlflow.start_run():
    model.train()
    for epoch in range(50):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/50", leave=False)
        total_loss = 0
        for batch_X, batch_y in progress_bar:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        avg_loss = total_loss / len(train_loader)
        mlflow.log_metric("train_loss", avg_loss, step=epoch)

        # Evaluate AUC per epoch
        model.eval()
        y_pred = []
        y_true = []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                y_pred.extend(outputs.numpy())
                y_true.extend(batch_y.numpy())
        auc = roc_auc_score(y_true, y_pred)
        mlflow.log_metric("test_auc", auc, step=epoch)
        print(f"Epoch {epoch+1}/50 - Loss: {avg_loss:.4f}, AUC: {auc:.4f}")
        model.train()

    # Final evaluation
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            y_pred.extend(outputs.numpy())
            y_true.extend(batch_y.numpy())
    
    auc = roc_auc_score(y_true, y_pred)
    mlflow.log_metric("final_auc", auc)
    
    # Infer model signature
    input_example = X_test.iloc[:1].values.astype(np.float32)
    predictions = model(torch.tensor(input_example)).detach().numpy()
    signature = infer_signature(X_test.iloc[:1], predictions)
    
    # Log model with signature
    mlflow.pytorch.log_model(model, "pytorch_model", signature=signature)
    print(f"\nFinal AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, np.round(y_pred)))

    # Save model for DVC
    torch.save(model.state_dict(), 'models/pytorch_model.pth')