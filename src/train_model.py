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
features = ['log_amount', 'sender_country', 'receiver_country', 'currency', 
            'remittance_purpose', 'payment_method', 'transaction_status', 
            'bank', 'agent', 'transaction_type', 'hour']
target = 'risk_flag'

# Convert categorical columns to factors
for col in features:
    if col != 'log_amount' and col != 'hour':
        h2o_df[col] = h2o_df[col].asfactor()
h2o_df['risk_flag'] = h2o_df['risk_flag'].asfactor()

# Split data
train, test = h2o_df.split_frame(ratios=[0.8], seed=42)

# Train AutoML model
aml = H2OAutoML(max_models=20, seed=42, max_runtime_secs=1200, balance_classes=True)
aml.train(x=features, y='risk_flag', training_frame=train)

# View leaderboard
print(aml.leaderboard)
perf = aml.leader.model_performance(test)
print(perf)

# Evaluate on test set
preds = aml.leader.predict(test)
print(aml.leader.model_performance(test))

# Save the best model
h2o.save_model(aml.leader, path = 'models/h2o_model', 
               force = True)

# Shutdown H2O
h2o.cluster().shutdown()
