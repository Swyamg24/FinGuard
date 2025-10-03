import pandas as pd
import numpy as np
import yaml
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# --- 1. Data Loading and Preprocessing ---
def load_config(config_path):
    """Loads the configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_data(path):
    """Loads transaction data and converts timestamp to datetime objects."""
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

# --- 2. Rule-Based Anomaly Detection ---
def detect_high_value(df, threshold):
    """Rule 1: Flag transactions exceeding a value threshold."""
    return df[df['amount'] > threshold].copy()

def detect_unusual_timing(df, start_hour, end_hour):
    """Rule 2: Flag transactions occurring in unusual hours."""
    return df[(df['timestamp'].dt.hour >= start_hour) & (df['timestamp'].dt.hour < end_hour)].copy()

def detect_rapid_fire(df, window_minutes, count_threshold):
    """Rule 3: Flag transactions from a user within a short time window."""
    df_sorted = df.sort_values(by=['user_id', 'timestamp'])
    # Calculate time difference between consecutive transactions for the same user
    time_diff = df_sorted.groupby('user_id')['timestamp'].diff().dt.total_seconds() / 60
    
    # Identify transactions that are part of a rapid-fire sequence
    # A transaction is suspicious if the time diff to the PREVIOUS one is small
    suspicious_indices = time_diff[time_diff < window_minutes].index
    
    # We need to find groups of `count_threshold` or more such transactions
    flagged_ids = set()
    for idx in suspicious_indices:
        user_id = df_sorted.loc[idx, 'user_id']
        start_time = df_sorted.loc[idx, 'timestamp'] - pd.Timedelta(minutes=window_minutes)
        
        # Get all transactions for this user in the window
        user_window_txns = df_sorted[
            (df_sorted['user_id'] == user_id) &
            (df_sorted['timestamp'] >= start_time) &
            (df_sorted['timestamp'] <= df_sorted.loc[idx, 'timestamp'])
        ]
        
        if len(user_window_txns) >= count_threshold:
            # Flag all transactions in this identified window
            for i in user_window_txns.index:
                flagged_ids.add(i)
                
    return df.loc[list(flagged_ids)].copy()

# --- 3. Machine Learning Anomaly Detection ---
def feature_engineering(df):
    """Creates new features for the ML model."""
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek # Monday=0, Sunday=6
    
    # User-level aggregated features
    user_avg_amount = df.groupby('user_id')['amount'].transform('mean')
    df['amount_deviation_from_user_avg'] = df['amount'] - user_avg_amount
    
    return df

def detect_ml_anomalies(df, config):
    """Uses Isolation Forest to detect anomalies."""
    features = ['amount', 'hour_of_day', 'day_of_week', 'amount_deviation_from_user_avg']
    
    # Scale features for better model performance
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[features])
    
    # Train the model
    model = IsolationForest(
        contamination=config['isolation_forest']['contamination'],
        random_state=config['isolation_forest']['random_state']
    )
    df['ml_anomaly'] = model.fit_predict(df_scaled) # -1 for anomalies, 1 for inliers
    
    # Return anomalous transactions
    return df[df['ml_anomaly'] == -1].copy()

# --- 4. Reporting and Visualization ---
def generate_report(anomalies_dict, output_path):
    """Combines all detected anomalies into a single report."""
    # Add a 'reason' column to each anomaly dataframe
    for reason, df in anomalies_dict.items():
        df['reason'] = reason
    
    # Concatenate all dataframes
    all_anomalies = pd.concat(anomalies_dict.values(), ignore_index=True)
    
    # Drop duplicates, keeping the first reason a transaction was flagged
    final_report = all_anomalies.drop_duplicates(subset=['transaction_id'], keep='first')
    
    # Clean up ML-related columns before saving
    final_report = final_report.drop(columns=['ml_anomaly'], errors='ignore')
    
    final_report.to_csv(output_path, index=False)
    print(f"Anomaly report generated at: {output_path}")
    return final_report

def plot_anomaly_distribution(report_df, output_path):
    """Creates and saves a bar chart of anomaly types."""
    plt.figure(figsize=(10, 6))
    sns.countplot(y='reason', data=report_df, order=report_df['reason'].value_counts().index)
    plt.title('Distribution of Detected Anomaly Types')
    plt.xlabel('Number of Transactions')
    plt.ylabel('Reason for Flagging')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Anomaly distribution plot saved at: {output_path}")

# --- 5. Main Execution ---
def main():
    # Setup CLI argument parser
    parser = argparse.ArgumentParser(description="FinGuard: Transaction Anomaly Detection Tool")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file.')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load data
    df = load_data(config['input_csv_path'])
    
    # --- Execute Detection Phases ---
    anomalies_detected = {}
    
    # Phase 1: Rule-Based Detections
    anomalies_detected['High-Value Transaction'] = detect_high_value(df, config['high_value_threshold'])
    anomalies_detected['Unusual Timing (2-5 AM)'] = detect_unusual_timing(df, config['unusual_hours']['start'], config['unusual_hours']['end'])
    anomalies_detected['Rapid-Fire Transactions'] = detect_rapid_fire(df, config['rapid_fire']['window_minutes'], config['rapid_fire']['count_threshold'])
    
    # Phase 2: Machine Learning Detection
    df_features = feature_engineering(df.copy())
    anomalies_detected['ML Anomaly (Isolation Forest)'] = detect_ml_anomalies(df_features, config)
    
    # --- Generate Outputs ---
    # Phase 3: Reporting and Visualization
    report_df = generate_report(anomalies_detected, config['output_csv_path'])
    plot_anomaly_distribution(report_df, config['output_plot_path'])

if __name__ == '__main__':
    main()