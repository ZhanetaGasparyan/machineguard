# Dataset

This project uses the AI4I 2020 Predictive Maintenance Dataset from the
UCI Machine Learning Repository.

- Source: https://archive.ics.uci.edu/dataset/601/ai4i
- Instances: 10,000
- Target: `Machine failure`
- License: Creative Commons Attribution 4.0
- Dataset type: Synthetic industrial predictive-maintenance data

The original, unmodified CSV is stored in `data/raw/ai4i2020.csv`.

## Important limitation

This is a synthetic dataset designed to resemble industrial
predictive-maintenance data. Model performance in this project should
not be interpreted as expected performance on real industrial machinery.

MachineGuard



An end-to-end machine-learning platform that estimates industrial equipment failure risk from operating measurements.

MachineGuard combines a reproducible scikit-learn pipeline, a validated FastAPI service, an interactive Streamlit dashboard, automated tests, and continuous integration.

Features

Predicts machine-failure probability from six operating measurements

Compares logistic regression and random forest models

Handles severe class imbalance using class weighting and appropriate metrics

Prevents target leakage by excluding failure-mode indicators

Separates training, validation, and untouched test data

Exposes predictions through a validated REST API

Provides an interactive Streamlit dashboard

Includes 11 automated API and prediction tests

Runs tests automatically with GitHub Actions

Architecture

AI4I Dataset
     |
     v
Data exploration and leakage analysis
     |
     v
Preprocessing pipeline
     |
     +--> Logistic Regression
     |
     +--> Random Forest
              |
              v
       Saved model pipeline
              |
              v
          FastAPI API
              |
              v
      Streamlit Dashboard

Model performance

The random forest was selected using validation PR-AUC and then evaluated once on an untouched test set.

Metric

Test result

Precision

72.13%

Recall

64.71%

F1 score

68.22%

ROC-AUC

96.11%

PR-AUC

75.92%

Accuracy is not treated as the primary metric because machine failures represent only 3.39% of the dataset.

At the current decision threshold of 0.50, the model detects approximately 65% of actual failures, while approximately 72% of its failure warnings are correct.

Target-leakage prevention

The original dataset includes five failure-mode columns:

TWF

HDF

PWF

OSF

RNF

These columns reveal information about whether a failure occurred. They are excluded from training to prevent artificially inflated model performance.

UDI and Product ID are also excluded because they are identifiers rather than meaningful predictive features.

The final model uses:

Product type

Air temperature

Process temperature

Rotational speed

Torque

Tool wear

Technology stack

Python

Pandas and NumPy

scikit-learn

FastAPI and Pydantic

Streamlit

Plotly

pytest

GitHub Actions

Joblib

API

The API provides:

Method

Endpoint

Purpose

GET

/

Service information

GET

/health

Service and model health

GET

/model-info

Model metadata and evaluation results

POST

/predict

Machine-failure prediction

Example request:

{
  "product_type": "L",
  "air_temperature": 303.5,
  "process_temperature": 312.5,
  "rotational_speed": 1200,
  "torque": 70.0,
  "tool_wear": 230
}

Example response:

{
  "failure_probability": 0.866667,
  "prediction": 1,
  "risk_level": "high",
  "decision_threshold": 0.5,
  "model_name": "random_forest",
  "model_version": "1.0.0"
}

Run locally

Clone the repository:

git clone https://github.com/ZhanetaGasparyan/machineguard.git
cd machineguard

Create and activate a virtual environment:

python3 -m venv .venv
source .venv/bin/activate

Install the dependencies:

python -m pip install -r requirements.txt

Start the API:

uvicorn app.api:app --reload

The interactive API documentation will be available at:

http://127.0.0.1:8000/docs

In a second terminal, start the dashboard:

source .venv/bin/activate
streamlit run app/dashboard.py

The dashboard will be available at:

http://localhost:8501

Run the tests

python -m pytest -v

The project currently includes 11 tests covering:

Low-risk predictions

High-risk predictions

Deterministic predictions

Model metadata

API health

Valid API requests

Invalid product types

Negative measurements

Missing required fields

Train the model again

python scripts/train.py

This command:

Loads and validates the dataset.

Creates stratified training, validation, and test sets.

Trains logistic regression and random forest pipelines.

Selects the model using validation PR-AUC.

Evaluates the selected model on the untouched test set.

Saves the trained pipeline and model metadata.

Dataset

This project uses the AI4I 2020 Predictive Maintenance Dataset from the UCI Machine Learning Repository.

Instances: 10,000

Target: Machine failure

Failure prevalence: 3.39%

License: Creative Commons Attribution 4.0

Dataset type: synthetic industrial predictive-maintenance data

Limitations

The dataset is synthetic and does not represent a particular production facility.

Evaluation results should not be interpreted as expected performance on real industrial machinery.

Risk categories are designed for demonstration purposes.

The model should not be used for real maintenance or safety decisions without external validation.

Changes in production sensor distributions would require drift monitoring and potentially retraining.

Future improvements

Optimize the decision threshold using maintenance-cost assumptions

Add local prediction explanations

Monitor incoming feature distributions for data drift

Record prediction history

Add model-version comparison

Evaluate the system on real industrial sensor data

License

The project code is available under the MIT License. The dataset is distributed separately under CC BY 4.0.