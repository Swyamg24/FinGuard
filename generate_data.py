import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random

# Initialize Faker for generating realistic-looking data
fake = Faker()

# --- Configuration ---
NUM_TRANSACTIONS = 2000
NUM_USERS = 100
NUM_MERCHANTS = 50
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2023, 1, 31)

# --- Data Generation ---
def generate_base_transactions(n):
    """Generates a base set of normal transactions."""
    users = [fake.uuid4() for _ in range(NUM_USERS)]
    merchants = [fake.uuid4() for _ in range(NUM_MERCHANTS)]
    
    data = []
    for _ in range(n):
        data.append({
            'transaction_id': fake.uuid4(),
            'user_id': random.choice(users),
            'timestamp': fake.date_time_between(start_date=START_DATE, end_date=END_DATE),
            'amount': round(random.uniform(5.0, 500.0), 2),
            'merchant_id': random.choice(merchants),
            'location': fake.city()
        })
    return pd.DataFrame(data)

def inject_anomalies(df):
    """Injects specific types of anomalies into the dataframe."""
    anomalies = []
    
    # --- Anomaly 1: High-Value Transactions ---
    # Select 5 random transactions to make high-value
    for _ in range(5):
        anomaly = df.sample(1).iloc[0].to_dict()
        anomaly['amount'] = round(random.uniform(6000.0, 10000.0), 2)
        anomaly['reason_true'] = 'High-Value Transaction'
        anomalies.append(anomaly)

    # --- Anomaly 2: Unusual Timings ---
    # Select 5 random transactions and set their time to be between 2-5 AM
    for _ in range(5):
        anomaly = df.sample(1).iloc[0].to_dict()
        # Create a new timestamp with an unusual hour
        original_date = anomaly['timestamp']
        anomaly['timestamp'] = original_date.replace(hour=random.randint(2, 4), minute=random.randint(0, 59))
        anomaly['reason_true'] = 'Unusual Timing'
        anomalies.append(anomaly)

    # --- Anomaly 3: Rapid-Fire Transactions ---
    # Create 3 transactions for a single user in a 2-minute window
    user_for_rapid_fire = df['user_id'].sample(1).iloc[0]
    base_time = fake.date_time_between(start_date=START_DATE, end_date=END_DATE)
    for i in range(3):
        anomaly = df.sample(1).iloc[0].to_dict()
        anomaly['user_id'] = user_for_rapid_fire
        anomaly['timestamp'] = base_time + timedelta(seconds=i * 45) # Transactions are 45s apart
        anomaly['amount'] = round(random.uniform(10.0, 100.0), 2)
        anomaly['reason_true'] = 'Rapid-Fire Transactions'
        anomalies.append(anomaly)
        
    # Append anomalies to the original dataframe
    anomalies_df = pd.DataFrame(anomalies)
    df_with_anomalies = pd.concat([df, anomalies_df], ignore_index=True)
    
    # Shuffle the dataset to mix anomalies with normal data
    return df_with_anomalies.sample(frac=1).reset_index(drop=True)

if __name__ == "__main__":
    print("Generating transaction data...")
    base_df = generate_base_transactions(NUM_TRANSACTIONS)
    final_df = inject_anomalies(base_df)
    
    # Ensure the 'data' directory exists
    import os
    if not os.path.exists('data'):
        os.makedirs('data')
        
    # Save to CSV
    output_path = 'data/transactions.csv'
    final_df.to_csv(output_path, index=False)
    print(f"Data generation complete. Saved {len(final_df)} transactions to {output_path}")
    print("\nSample of generated data:")
    print(final_df.head())