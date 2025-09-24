import h2o
from h2o.automl import H2OAutoML
import pandas as pd

# Initialize H2O
h2o.init()

# load dataset
df = pd.read_csv('data/remittance_data.csv')

# convert to H2OFrame
h2o_df = h2o.H2OFrame(df)

# Define features and target
features = ['sender_country', 'receiver_country', 'amount', 'currency', 
            'remittance_purpose', 'payment_method', 'transaction_status', 
            'bank', 'agent', 'transaction_type']
target = 'risk_flag'

# Convert categorical columns to factors
for col in ['sender_country', 'receiver_country', 'currency', 'remittance_purpose', 
            'payment_method', 'transaction_status', 'bank', 'agent', 'transaction_type', 'risk_flag']:
    h2o_df[col] = h2o_df[col].asfactor()

# Split data
train, test = h2o_df.split_frame(ratios=[.8], seed=42)

# Train AutoML model
aml = H2OAutoML(max_models=10, seed = 42, max_runtime_secs= 600)
aml.train(x = features, y = target, training_frame= train)

# View leaderboard
print(aml.leaderboard)

# Evaluate on test set
preds = aml.leader.predict(test)
print(aml.leader.model_performance(test))

# Save the best model
h2o.save_model(aml.leader, path = 'models/h2o_model', 
               force = True)

# Shutdown H2O
h2o.cluster().shutdown()
