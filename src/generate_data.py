from faker import Faker
import pandas as pd
import random
import numpy as np
from datetime import datetime, timedelta
from unidecode import unidecode

# Initialize multiple Faker instances for diverse names
fakers = {
    'en_US': Faker('en_US'),    # Western names (USA, UK, etc.)
    'ar_AE': Faker('ar_AE'),    # Arabic names (UAE, Pakistan, etc.)
    'en_IN': Faker('en_IN'),    # Indian names
    'fil_PH': Faker('fil_PH'),  # Filipino names
    'en_NG': Faker('en_NG')     # East African names (Kenya, Nigeria)
}

# Define realistic parameters for UAE remittances
countries = ['India', 'Pakistan', 'Bangladesh', 'Philippines', 'Egypt', 
             'Sri Lanka', 'Nepal', 'Jordan', 'Lebanon', 'Yemen', 
             'UK', 'South Africa', 'Kenya', 'Nigeria', 'Sudan', 
             'USA', 'Germany', 'France', 'Italy', 'Australia', 
             'Canada', 'Spain', 'UAE']
currencies = ['INR', 'PKR', 'BDT', 'PHP', 'EGP', 'LKR', 'NPR', 
              'JOD', 'LBP', 'YER', 'GBP', 'ZAR', 'KES', 'NGN', 
              'SDG', 'USD', 'EUR', 'AUD', 'CAD', 'AED']
remittance_purposes = ['Family Support', 'Education', 'Medical Expenses', 
                       'Savings', 'Investment', 'Travel', 'Business', 
                       'Gifts', 'Charity']
payment_methods = ['Bank Transfer', 'Cash Pickup', 'Mobile Wallet', 
                   'Cheque', 'Direct Deposit', 'Prepaid Card']
transaction_statuses = ['Completed', 'Pending', 'Failed', 'Cancelled']
banks = ['Emirates NBD', 'Abu Dhabi Commercial Bank', 'First Abu Dhabi Bank', 
         'Mashreq Bank', 'Dubai Islamic Bank', 'RAK Bank']
agents = ['Western Union', 'MoneyGram', 'Xpress Money', 'Al Ansari Exchange', 
          'UAE Exchange', 'Al Fardan Exchange', 'BNP Paribas', 'HSBC', 
          'Citibank', 'Standard Chartered', 'Barclays', 'Deutsche Bank', 
          'Wells Fargo', 'Chase Bank', 'Bank of America', 'TD Bank', 
          'Scotiabank', 'Royal Bank of Canada', 'ING', 'Santander', 
          'UniCredit', 'Credit Agricole', 'Societe Generale', 'Nordea', 
          'Danske Bank', 'SEB']
uae_specific_agents = ['Al Ansari Exchange', 'UAE Exchange', 'Al Fardan Exchange']
transaction_types = ['Online', 'In-Person', 'Mobile App']
risk_flag_prob = 0.05  # Not used directly, but referenced for scaling

# Function to pick Faker locale based on country
def get_faker_for_country(country):
    if country in ['India', 'Sri Lanka', 'Nepal']:
        return fakers['en_IN']  # Indian subcontinent names
    elif country in ['Pakistan', 'UAE', 'Egypt', 'Jordan', 'Lebanon', 'Sudan', 'Yemen']:
        return fakers['ar_AE']  # Arabic names
    elif country == 'Philippines':
        return fakers['fil_PH']  # Filipino names
    elif country in ['Kenya', 'Nigeria']:
        return fakers['en_NG']  # East African names
    else:
        return fakers['en_US']  # Default to US names

# Function to get transliterated name
def get_transliterated_name(faker, country):
    name = faker.name()
    if country in ['Pakistan', 'UAE', 'Egypt', 'Jordan', 'Lebanon', 'Sudan', 'Yemen']:
        return unidecode(name)  # Use unidecode for ar_AE
    return name

# Function to pick agent with UAE weighting
def get_agent(sender_country):
    if sender_country == 'UAE':
        weights = []
        for agent in agents:
            if agent in uae_specific_agents:
                weights.append(0.7 / len(uae_specific_agents))  # 0.233 for each UAE agent
            else:
                weights.append(0.3 / (len(agents) - len(uae_specific_agents)))  # 0.013 for others
        return random.choices(agents, weights=weights, k=1)[0]
    return random.choice(agents)

# Generate 100K transaction records
num_transactions = 100_000
data = []

for _ in range(num_transactions):
    sender_country = random.choice(countries)
    receiver_country = random.choice([c for c in countries if c != sender_country])
    amount = round(random.uniform(100, 50_000), 2)
    
    # Match currency to receiver country
    if receiver_country in ['India', 'Pakistan', 'Bangladesh', 'Philippines', 'Egypt', 
                           'Sri Lanka', 'Nepal', 'Jordan', 'Lebanon', 'Yemen', 'Sudan']:
        currency = random.choice(['INR', 'PKR', 'BDT', 'PHP', 'EGP', 'LKR', 'NPR', 'JOD', 'LBP', 'YER', 'SDG'])
    elif receiver_country in ['Kenya', 'Nigeria']:
        currency = random.choice(['KES', 'NGN'])
    elif receiver_country in ['UK', 'USA', 'Germany', 'France', 'Italy', 'Australia', 
                             'Canada', 'Spain', 'South Africa']:
        currency = random.choice(['GBP', 'USD', 'EUR', 'AUD', 'CAD', 'ZAR'])
    else:  # UAE
        currency = 'AED'

    # Pick bank
    bank = random.choice(banks) if sender_country == 'UAE' else 'International Bank'

    # Pick agent
    agent = get_agent(sender_country)

    # Pick Faker for names
    sender_faker = get_faker_for_country(sender_country)
    receiver_faker = get_faker_for_country(receiver_country)

    # Compute risk flag
    risk_score = 0
    if amount > 20000:
        risk_score += 0.3
    if agent not in uae_specific_agents:
        risk_score += 0.2
    if sender_country != 'UAE' and receiver_country in ['Nigeria', 'Sudan']:
        risk_score += 0.3
    risk_flag = random.random() < risk_score * 0.2  # Adjust for ~5% base rate

    # Compute additional features
    timestamp = fakers['en_US'].date_time_between(start_date='-1y', end_date='now')
    log_amount = np.log(amount + 1)
    hour = timestamp.hour

    # Generate record
    data.append({
        'transaction_id': fakers['en_US'].uuid4(),
        'sender_name': get_transliterated_name(sender_faker, sender_country),
        'sender_country': sender_country,
        'receiver_name': get_transliterated_name(receiver_faker, receiver_country),
        'receiver_country': receiver_country,
        'amount': amount,
        'currency': currency,
        'remittance_purpose': random.choice(remittance_purposes),
        'payment_method': random.choice(payment_methods),
        'transaction_status': random.choice(transaction_statuses),
        'bank': bank,
        'agent': agent,
        'transaction_type': random.choice(transaction_types),
        'timestamp': timestamp,
        'risk_flag': risk_flag,
        'log_amount': log_amount,
        'hour': hour
    })

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('data/remittance_data.csv', index=False)
print(f"Generated {num_transactions} transactions and saved to data/remittance_data.csv")
print(df.head())