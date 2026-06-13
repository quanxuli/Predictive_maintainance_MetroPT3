# Predictive Maintenance for Metro-PT3: Enterprise IoT & ML Pipeline

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-14.0+-336791.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E.svg)
![PowerBI](https://img.shields.io/badge/Power_BI-Dashboard-F2C811.svg)

## 📌 Executive Summary
This project simulates an enterprise-grade IoT Data Pipeline and Machine Learning system designed for industrial manufacturing predictive maintenance. It ingests simulated high-frequency sensor data from a Metro-PT3 compressor, processes it in real-time using a sliding window buffer, logs it to a PostgreSQL database, and applies a **Supervised Machine Learning** model to predict physical air leaks before total system failure, ultimately reducing equipment downtime.

## 🏗️ Data Architecture & Pipeline

The system is built on a decoupled, real-time streaming architecture:

```text
[IoT Sensor API (Replay)] --(Streaming)--> [Real-time Predictor & Buffer]
      (FastAPI / Pandas)                     (Scikit-Learn / Pandas)
                                                        |
                                                        v
[Power BI Dashboard] <----(DirectQuery)---- [PostgreSQL Database]
    (Real-time BI)                                      |
                                                        v
                                        (sensor_realtime_predictions)
```

## 🚀 Core Features

**1. Real-Time IoT Data Ingestion (FastAPI)**
* A custom **FastAPI** server (`api_server.py`) mocks a factory IoT streaming endpoint.
* Includes a robust simulation of **Gradual Degradation** (mechanical wear and tear, e.g., pressure drops, vibration changes) to simulate realistic anomalies.

**2. On-the-fly Feature Engineering (Sliding Window)**
* Engineered a RAM-based memory buffer in `realtime_predictor.py` that maintains a sliding window (Window Size = 10) of the latest incoming sensor readings.
* Dynamically calculates `Rolling Mean` and `Rolling Standard Deviation` to capture true physical degradation trends rather than reacting to isolated sensor noise.

**3. Machine Learning Anomaly Detection (SMOTE + Random Forest)**
* Processed highly imbalanced industrial data (normal vs. failure states) using the **SMOTE** (Synthetic Minority Over-sampling Technique) algorithm in `train_offline_smote.py`.
* Deployed a **Random Forest** classifier that scores live data streams, effectively isolating the physical signature of an air leak.

**4. Model Validation & Sensitivity Analysis**
* A dedicated script (`validate_sensitivity.py`) performs Time-Series Cross Validation.
* Analyzes Feature Importance and Threshold Sensitivity to fine-tune the precision-recall trade-off.

## 📂 Repository Structure

```bash
├── MetroPT3.csv                      # Raw dataset (Not included due to size)
├── dashboard.pbix                    # Power BI Real-time Dashboard
├── api_server.py                     # Mock IoT Sensor Streaming API (FastAPI)
├── realtime_predictor.py             # Continuous Inference & Logging Pipeline
├── train_offline_smote.py            # ML Model Training & SMOTE implementation
├── validate_sensitivity.py           # Model Evaluation & Feature Importance
├── rf_smote_model.pkl                # Trained Random Forest Model
└── README.md                         # Project documentation
```

## 🛠️ Installation & Setup

1. **Clone the repository and install dependencies:**
   Ensure you have Python installed. The required libraries include `fastapi`, `uvicorn`, `pandas`, `scikit-learn`, `imbalanced-learn`, `joblib`, `sqlalchemy`, `psycopg2`, and `requests`.
   
   ```bash
   pip install fastapi uvicorn pandas scikit-learn imbalanced-learn joblib sqlalchemy psycopg2-binary requests
   ```

2. **Database Setup:**
   Ensure you have a PostgreSQL database running and update the `DB_CONNECTION_URL` in `realtime_predictor.py`:
   ```python
   DB_CONNECTION_URL = "postgresql://username:password@localhost:5432/metro_pt3_db"
   ```

3. **Data Requirements:**
   Place the `MetroPT3.csv` dataset in the root directory.

## 🏃 How to Run the Pipeline

### Step 1: Train the Offline Model
If you want to re-train the model, run the offline training script. This will read the dataset, perform SMOTE, train the Random Forest model, and output `rf_smote_model.pkl`.
```bash
python train_offline_smote.py
```

### Step 2: Validate the Model (Optional)
Run the validation script to check Time-Series Cross Validation and sensitivity analysis.
```bash
python validate_sensitivity.py
```

### Step 3: Start the IoT Stream API Server
Launch the FastAPI mock sensor API. The server will run on `http://127.0.0.1:8000`.
```bash
uvicorn api_server:app --reload
```

### Step 4: Run the Real-time Predictor
In a separate terminal, start the inference pipeline. It will continuously fetch data from the API, perform rolling feature engineering, predict anomalies, and log the results to PostgreSQL.
```bash
python realtime_predictor.py
```

## 📊 Dashboard Visualization
Open `dashboard.pbix` in Power BI and connect it to your PostgreSQL database to monitor real-time anomaly detection and sensor trends.