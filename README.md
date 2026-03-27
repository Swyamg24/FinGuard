# FinGuard: Transaction Anomaly Detection Tool 🛡️

## Description

FinGuard is a command-line tool built in Python to analyze financial transaction data and identify potential anomalies or fraudulent activities. It employs a powerful hybrid approach, combining a fast, explicit rule-based engine with an unsupervised machine learning model (Isolation Forest) to detect suspicious patterns that rules might miss.

This tool is designed to be easily configurable and can process transaction data from a CSV file, outputting a detailed report of all flagged activities.

## Features

-   **Hybrid Detection Engine**: Uses both predefined rules and a machine learning model for comprehensive analysis.
-   **Rule-Based Detection**:
    -   Flags transactions exceeding a configurable value threshold.
    -   Flags transactions occurring during unusual hours (e.g., late at night).
    -   Flags rapid-fire transactions from a single user in a short time window.
-   **Machine Learning Detection**:
    -   Utilizes `scikit-learn`'s **Isolation Forest** algorithm, which is highly effective for anomaly detection in unlabeled datasets.
    -   Automatically engineers features like `hour_of_day` and `amount_deviation_from_user_avg` to improve model accuracy.
-   **Configurable Parameters**: Easily adjust detection thresholds and model parameters via a `config.yaml` file without changing the source code.
-   **Comprehensive Reporting**: Generates a clean CSV report (`anomalies_report.csv`) detailing each flagged transaction and the specific reason it was flagged.
-   **Data Visualization**: Automatically creates a bar chart (`anomaly_distribution.png`) showing the count of each type of anomaly detected.

## How It Works

The tool processes data in several stages:
1.  **Configuration**: Reads parameters from `config.yaml` (e.g., file paths, thresholds).
2.  **Data Ingestion**: Loads transaction data from the specified input CSV file.
3.  **Rule-Based Analysis**: The data is first passed through a series of checks for high values, unusual timings, and rapid succession of transactions.
4.  **Feature Engineering**: The tool then creates new, insightful features from the raw data to prepare it for the machine learning model.
5.  **ML Analysis**: An Isolation Forest model is trained on the engineered features to identify transactions that are outliers compared to the norm.
6.  **Report Generation**: All anomalies identified by both the rule-based engine and the ML model are compiled into a single, de-duplicated report and a summary visualization is saved.

## Installation & Usage

### Prerequisites
-   Python 3.8+
-   pip

### 1. Clone the Repository

```bash
git clone https://github.com/Swyamg24/FinGuard
cd FinGuard