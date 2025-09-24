import pandas as pd

# Load dataset
df = pd.read_csv('data/remittance_data.csv')

# Basic checks
print("Shape:", df.shape)  # Should be (100000, 14)
print("Columns:", df.columns.tolist())
print("Missing Values:\n", df.isnull().sum())
print("Sample Names:\n", df[['sender_name', 'sender_country', 'receiver_name', 'receiver_country']].head())
print("Currencies:\n", df['currency'].value_counts())
print("UAE Agent Weighting:\n", df[df['sender_country'] == 'UAE']['agent'].value_counts(normalize=True))

# Check risk_flag distribution
print("Risk Flag Distribution:\n", df['risk_flag'].value_counts(normalize=True))