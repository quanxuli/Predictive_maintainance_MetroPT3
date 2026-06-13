# Predictive Maintenance for Metro-PT3: Enterprise IoT & ML Pipeline

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-14.0+-336791.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E.svg)
![PowerBI](https://img.shields.io/badge/Power_BI-Dashboard-F2C811.svg)

## Executive Summary
This project simulates an enterprise-grade IoT Data Pipeline and Machine Learning system designed for industrial manufacturing. It ingests simulated high-frequency sensor data from a Metro-PT3 compressor, processes it in real-time using a sliding window buffer, logs it to a PostgreSQL database, and applies a **Supervised Machine Learning** model to predict physical air leaks before total system failure, ultimately reducing equipment downtime.

## Data Architecture & Pipeline

The system is built on a decoupled, real-time streaming architecture:

    [IoT Sensor API (Replay)] --(Streaming)--> [Real-time Predictor & Buffer]
          (FastAPI / Pandas)                     (Scikit-Learn / Pandas)
                                                            |
                                                            v
    [Power BI Dashboard] <----(DirectQuery)---- [PostgreSQL Database]
        (Real-time BI)                                      |
                                                            v
                                            (sensor_realtime_predictions)

## Core Features & Business Value

**1. Real-Time IoT Data Ingestion (FastAPI)**
* Built a custom **FastAPI** server to mock a factory IoT streaming endpoint.
* Configured a "Fast-forward" historical replay mechanism to stream data starting exactly 1 minute before a documented historical failure (April 17, 2020, 23:59:00), ensuring physical accuracy.

**2. On-the-fly Feature Engineering (Sliding Window)**
* Engineered a RAM-based memory buffer that maintains a sliding window (W=10) of the latest incoming sensor readings.
* Dynamically calculates `Rolling Mean` and `Rolling Standard Deviation` to capture the true physical degradation trend rather than reacting to isolated sensor noise.

**3. Machine Learning Anomaly Detection (SMOTE + RF)**
* Processed highly imbalanced industrial data (normal vs. failure states) using the **SMOTE** (Synthetic Minority Over-sampling Technique) algorithm.
* Deployed a **Random Forest** classifier that scores live data streams, effectively isolating the physical signature of an air leak (e.g., pressure drops combined with high vibration/H1 std).

**4. P-F Curve Simulation & Zero False Positives**
* The pipeline successfully simulates the **P-F Interval** (Potential to Functional Failure). The model remains silent during minor, non-critical fluctuations and only triggers a "Red Alert" when the degradation curve hits the critical operational threshold.

## Repository Structure

```bash
├── MetroPT3.csv                      # Raw dataset (Not uploaded due to size)
├── Dashboard.pbix                    # Power BI Real-time Dashboard
├── api_server.py                     # Mock IoT Sensor Streaming API (FastAPI)
├── realtime_predictor.py             # Continuous Inference & Logging Pipeline
├── train_offline_smote.py            # ML Model Training & SMOTE implementation
├── validate_sensitivity.py           # Model Evaluation & Feature Importance
├── rf_smote_model.pkl                # Trained Random Forest Model
└── README.md