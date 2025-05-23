# Flight Delay Prediction and Deployment Pipeline

**Author:** May Cooper

## Overview

This project implements a complete machine learning pipeline to predict average flight departure delays using historical airline performance data. It integrates data ingestion, preprocessing, model training with hyperparameter tuning, experiment tracking using MLflow, and deployment via a RESTful API served in a Docker container. It is designed to be modular, version-controlled, and portable.

The end product is a containerized, production-ready API that accepts flight schedule inputs and returns estimated average delays. The system emphasizes model reproducibility, modular automation, and multi-step orchestration via MLflow.

---

## Goals

* Construct a modular pipeline that includes data ingestion, transformation, training, evaluation, and inference.
* Build a Ridge regression model enhanced with polynomial features to predict departure delays.
* Leverage MLflow to log experiment parameters, metrics, and artifacts.
* Deploy the trained model through a FastAPI application for real-time inference.
* Test model input/output handling and ensure robust endpoint validation.
* Package and deploy the application using Docker to standardize environment and deployment.

---

## Project Structure

```
Flight-Delay-MLOps/
│
├── MLOps_Flight_Prediction/
│   ├── Scripts/
│   │   ├── data_import.py
│   │   ├── filter_clean_data.py
│   │   ├── finalized_model.pkl
│   │   ├── cleaned_data.csv
│   │   ├── airport_encodings.json
│   │   └── T_ONTIME_REPORTING.csv.dvc
│   ├── T_ONTIME_REPORTING.csv
│   ├── cleaned_data.csv
│   ├── finalized_model.pkl
│   ├── evaluate_model_test.ipynb
│   ├── airport_encodings.json
│   ├── model_performance_test.jpg
│   ├── polynomial_regression.txt
│   ├── main_pipeline_train_poly_ridge_model.py
│   ├── pipeline_env.yaml
│   ├── formatted_T_ONTIME_REPORTING.csv
│   ├── formatted_T_ONTIME_REPORTING.csv.dvc
│   └── MLproject
│
├── API_Deployment_Flight_Prediction/
│   ├── Scripts/
│   │   ├── activate
│   │   ├── activate.bat
│   │   ├── Activate.ps1
│   │   └── deactivate.bat
│   ├── T_ONTIME_REPORTING.csv
│   ├── airport_encodings.json
│   ├── finalized_model.pkl
│   ├── generate_encodings.py
│   ├── app_predict_delay_api.py
│   ├── dockerfile
│   ├── test_api.py
│   └── requirements.txt
│
├── README.md
```

---

## Technologies Used

* **Python** (pandas, numpy, scikit-learn)
* **FastAPI** for serving predictions
* **Uvicorn** as the ASGI server for FastAPI
* **MLflow** for experiment tracking and orchestration
* **Docker** for containerization and API deployment
* **DVC** (Data Version Control) for tracking dataset versions
* **Jupyter Notebook** for exploratory model evaluation and visualization
* **Pytest** for unit testing the API endpoints
* **Matplotlib & Seaborn** for performance and error visualization

---

## Pipeline Stages

### 1. Data Ingestion

`data_import.py` processes raw flight data, checks schema compliance, and renames fields to align with the model's structure. DVC is used to version the input dataset.

### 2. Data Cleaning

`filter_clean_data.py` filters data to include only flights departing from MIA (Miami), removes outliers and missing values, and adds features like an "ON\_TIME" binary flag.

### 3. Model Training

`main_pipeline_train_poly_ridge_model.py` trains a polynomial Ridge regression model using a series of alpha values. This script logs training runs, artifacts, and performance metrics to MLflow, ensuring experiment reproducibility.

Artifacts include:

* `finalized_model.pkl`: trained model file
* `airport_encodings.json`: one-hot encoding reference
* `model_performance_test.jpg`: visualization of predicted vs. actual delays

### 4. Pipeline Automation

The `MLproject` file links the scripts into a seamless pipeline and supports parameterized runs. Environment dependencies are pinned in `pipeline_env.yaml`.

### 5. API Deployment

`app_predict_delay_api.py` provides a lightweight RESTful API that receives flight schedule details and returns a predicted average delay in minutes. Time values are validated for strict formatting, and model inputs are preprocessed for consistency.

Dockerized deployment ensures consistency across environments. Unit testing is conducted via `pytest`, and the container exposes port 8000 for public interaction.

---

## API Usage Example
![image](https://github.com/user-attachments/assets/33147bad-f228-4fa6-b008-f37ed6906979)
![image](https://github.com/user-attachments/assets/2009c9e5-c23e-40c3-a2b4-9138dfaadad1)

**Endpoint:** `/predict/delays`
**Method:** GET

**Sample Query:**

```
http://localhost:8000/predict/delays?arrival_airport=JFK&departure_time=09:30&arrival_time=12:15
```

**Sample Response:**

```json
{
  "arrival_airport": "JFK",
  "departure_time": "09:30",
  "arrival_time": "12:15",
  "average_delay": 8.37
}
```

**Real Example**
![image](https://github.com/user-attachments/assets/125ed085-52b6-47f3-a7a8-ba1b921d555e)
**Response:**
![image](https://github.com/user-attachments/assets/b55ecde4-c029-4c29-ae9f-65b08040218a)

---

## Testing

`test_api.py` validates endpoint behavior, including correct responses to valid inputs and appropriate error handling for malformed or missing parameters.

```bash
pytest test_api.py
```

---

## Docker Instructions

To build and run the API in a Docker container:

```bash
docker build -t flight-delay-api .
docker run -p 8000:8000 flight-delay-api
```

Then access Swagger UI at `http://localhost:8000/docs`.
![image](https://github.com/user-attachments/assets/d7c02a7e-65d4-4087-805a-037d1cf40744)
![image](https://github.com/user-attachments/assets/853980c6-69b4-4ebd-9813-da0440a954bc)

---

## MLflow Tracking

The training process logs the following:

* Parameters: `alpha` values, polynomial degree
* Metrics: Mean Squared Error (MSE), Root Mean Squared Error (RMSE)
* Artifacts: trained model, performance charts, and logs

To launch MLflow UI:

```bash
mlflow ui
```

Then navigate to `http://localhost:5000`.

---

## Inputs & Outputs

| Input Columns          | Description                           |
| ---------------------- | ------------------------------------- |
| YEAR, MONTH, DAY       | Flight date                           |
| ORIGIN, DEST           | Airports (departure/arrival)          |
| SCHEDULED/ACTUAL TIMES | HHMM format times (departure/arrival) |
| DELAYS                 | Departure and arrival delays (min)    |

| Output         | Description                        |
| -------------- | ---------------------------------- |
| average\_delay | Predicted average delay in minutes |

---

## Results

The model was trained on flights departing from Miami International Airport (MIA) using a one-month snapshot of flight data. After preprocessing and cleaning, the model was trained using Ridge regression with polynomial feature expansion and evaluated using a held-out test set.

Across various alpha values, the model demonstrated minimal overfitting and maintained relatively stable generalization. The best performing model achieved a Mean Squared Error (MSE) below 20 on the test dataset, with predictions exhibiting a consistent linear relationship to actual delay values.

The performance evaluation was visualized in `model_performance_test.jpg`, where predicted delays closely tracked true values with little deviation.

This deployment-ready solution:

* Enables accurate and quick predictions for operational flight monitoring
* Provides a modular foundation for extending to other airports or variables
* Promotes version tracking and experimentation across development iterations

---

## Implications

The successful implementation of this flight delay prediction pipeline has practical implications for both technical teams and decision-makers:

* **For developers and MLOps teams**: The structure of this solution demonstrates how to move from raw CSV files to a production-ready API using tools like MLflow, DVC, and FastAPI. It illustrates the importance of modularization, reproducibility, and controlled experimentation.

* **For business users and analysts**: The API provides actionable insights in real time, helping operations teams anticipate delays and optimize scheduling. Even without integration into larger systems, it can serve as a standalone analytical utility for delay monitoring.

* **For future deployment strategies**: The containerized architecture allows for easy scaling and integration into broader airline or airport IT ecosystems. By adjusting minimal parameters, the model can be adapted to other origin airports, making it suitable for wider applications across regions or seasons.

---

## Future Improvements

* Extend model capabilities to support predictions for any origin airport
* Integrate weather, aircraft, and distance data as additional predictors
* Implement authentication and rate limiting for public API use
* Schedule automated retraining workflows using cron or Airflow
* Add database logging and prediction history tracking for auditability
