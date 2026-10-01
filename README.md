# MachineGuard

[![Tests](https://github.com/ZhanetaGasparyan/machineguard/actions/workflows/tests.yml/badge.svg)](https://github.com/ZhanetaGasparyan/machineguard/actions)

An end-to-end machine-learning application that estimates industrial equipment failure risk from operating measurements.

MachineGuard combines a reproducible scikit-learn pipeline, a validated FastAPI service, an interactive Streamlit dashboard, automated tests, and continuous integration.

## Highlights

- Prevents target leakage by excluding failure-mode indicators
- Compares logistic regression and random forest models
- Uses separate training, validation, and untouched test sets
- Evaluates an imbalanced target with precision, recall, F1, ROC-AUC, and PR-AUC
- Exposes predictions through a validated REST API
- Provides an interactive Streamlit dashboard
- Includes 11 automated tests and GitHub Actions CI

## Architecture

```text
AI4I dataset
     |
     v
Data exploration and leakage analysis
     |
     v
Preprocessing and model comparison
     |
     v
Saved random-forest pipeline
     |
     v
FastAPI service
     |
     v
Streamlit dashboard
```

## Model performance

The random forest was selected using validation PR-AUC and evaluated once on an untouched test set.

| Metric | Test result |
|---|---:|
| Precision | 72.13% |
| Recall | 64.71% |
| F1 score | 68.22% |
| ROC-AUC | 96.11% |
| PR-AUC | 75.92% |

Machine failures represent only 3.39% of the dataset, so accuracy is not used as the primary metric.

## Target-leakage prevention

The original dataset contains five failure-mode indicators: `TWF`, `HDF`, `PWF`, `OSF`, and `RNF`. These columns reveal information about the target and are excluded from training.

`UDI` and `Product ID` are also excluded because they are identifiers. The final model uses:

- Product type
- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear

## Technology stack

- Python, Pandas, NumPy, and scikit-learn
- FastAPI and Pydantic
- Streamlit and Plotly
- pytest and GitHub Actions
- Joblib

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Service information |
| `GET` | `/health` | Service and model health |
| `GET` | `/model-info` | Model metadata and evaluation results |
| `POST` | `/predict` | Machine-failure prediction |

Example request:

```json
{
  "product_type": "L",
  "air_temperature": 303.5,
  "process_temperature": 312.5,
  "rotational_speed": 1200,
  "torque": 70.0,
  "tool_wear": 230
}
```

Example response:

```json
{
  "failure_probability": 0.866667,
  "prediction": 1,
  "risk_level": "high",
  "decision_threshold": 0.5,
  "model_name": "random_forest",
  "model_version": "1.0.0"
}
```

## Run locally

```bash
git clone https://github.com/ZhanetaGasparyan/machineguard.git
cd machineguard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Start the API:

```bash
uvicorn app.api:app --reload
```

The API documentation is available at `http://127.0.0.1:8000/docs`.

In a second terminal, start the dashboard:

```bash
source .venv/bin/activate
streamlit run app/dashboard.py
```

The dashboard is available at `http://localhost:8501`.

## Testing

```bash
python -m pytest -v
```

The test suite covers prediction behavior, model metadata, API health, valid requests, and invalid inputs.

## Dataset

This project uses the [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i) from the UCI Machine Learning Repository.

- Instances: 10,000
- Target: `Machine failure`
- Failure prevalence: 3.39%
- Dataset license: CC BY 4.0
- Dataset type: synthetic industrial predictive-maintenance data

## Limitations

- The dataset is synthetic and does not represent a specific production facility.
- Results should not be interpreted as expected performance on real machinery.
- Risk categories are intended for demonstration purposes.
- The model requires external validation before any real maintenance or safety use.

## Future improvements

- Optimize the decision threshold using maintenance-cost assumptions
- Add local prediction explanations
- Monitor incoming feature distributions for data drift
- Record prediction history and model versions

## License

The project code is available under the MIT License. The dataset is distributed separately under CC BY 4.0.
