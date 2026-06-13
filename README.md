##  Predictive Maintenance for Metro-PT3: Enterprise IoT & ML Pipeline

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-14.0+-336791.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E.svg)
![PowerBI](https://img.shields.io/badge/Power_BI-Dashboard-F2C811.svg)

## Executive Summary
This project simulates an enterprise-grade IoT Data Pipeline and Machine Learning system designed for industrial manufacturing. It ingests simulated high-frequency sensor data from a Metro-PT3 compressor, processes it in real-time using a sliding window buffer, logs it to a PostgreSQL database, and applies a **Supervised Machine Learning** model to predict physical air leaks before total system failure, ultimately reducing equipment downtime.

## Data Architecture & Pipeline

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
