import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


API_URL = os.getenv(
    "MACHINEGUARD_API_URL",
    "http://127.0.0.1:8000",
)


st.set_page_config(
    page_title="MachineGuard",
    page_icon="⚙️",
    layout="wide",
)


def get_model_info():
    """Request model information from the FastAPI service."""

    response = requests.get(
        f"{API_URL}/model-info",
        timeout=10,
    )
    response.raise_for_status()

    return response.json()


def request_prediction(payload):
    """Send machine measurements to the prediction API."""

    response = requests.post(
        f"{API_URL}/predict",
        json=payload,
        timeout=10,
    )
    response.raise_for_status()

    return response.json()


st.title("MachineGuard")
st.subheader("Predictive maintenance for industrial equipment")

st.write(
    "Estimate machine-failure risk from operating measurements using "
    "a trained random-forest model."
)

prediction_tab, performance_tab, about_tab = st.tabs(
    [
        "Prediction",
        "Model performance",
        "About",
    ]
)


with prediction_tab:
    st.header("Machine measurements")

    with st.form("prediction_form"):
        first_column, second_column = st.columns(2)

        with first_column:
            product_type = st.selectbox(
                "Product type",
                options=["L", "M", "H"],
                help="L = low, M = medium, H = high product quality.",
            )

            air_temperature = st.number_input(
                "Air temperature (K)",
                min_value=250.0,
                max_value=400.0,
                value=300.0,
                step=0.1,
            )

            process_temperature = st.number_input(
                "Process temperature (K)",
                min_value=250.0,
                max_value=500.0,
                value=310.0,
                step=0.1,
            )

        with second_column:
            rotational_speed = st.number_input(
                "Rotational speed (rpm)",
                min_value=1,
                max_value=10000,
                value=1500,
                step=10,
            )

            torque = st.number_input(
                "Torque (Nm)",
                min_value=0.0,
                max_value=500.0,
                value=40.0,
                step=0.1,
            )

            tool_wear = st.number_input(
                "Tool wear (min)",
                min_value=0,
                max_value=1000,
                value=100,
                step=1,
            )

        submitted = st.form_submit_button(
            "Estimate failure risk",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        payload = {
            "product_type": product_type,
            "air_temperature": air_temperature,
            "process_temperature": process_temperature,
            "rotational_speed": rotational_speed,
            "torque": torque,
            "tool_wear": tool_wear,
        }

        try:
            result = request_prediction(payload)
            probability = result["failure_probability"]
            percentage = probability * 100
            risk_level = result["risk_level"].upper()

            st.divider()
            st.header("Prediction result")

            probability_column, risk_column, prediction_column = st.columns(3)

            probability_column.metric(
                "Failure probability",
                f"{percentage:.1f}%",
            )

            risk_column.metric(
                "Risk level",
                risk_level,
            )

            prediction_column.metric(
                "Model decision",
                (
                    "Inspection recommended"
                    if result["prediction"] == 1
                    else "No failure predicted"
                ),
            )

            st.progress(probability)

            if result["risk_level"] == "high":
                st.error(
                    "The supplied measurements indicate elevated failure risk. "
                    "A maintenance inspection is recommended."
                )
            elif result["risk_level"] == "medium":
                st.warning(
                    "The supplied measurements indicate moderate failure risk. "
                    "Continue monitoring the equipment."
                )
            else:
                st.success(
                    "The supplied measurements indicate low failure risk."
                )

            st.caption(
                f"Model: {result['model_name']} "
                f"v{result['model_version']} · "
                f"Decision threshold: {result['decision_threshold']:.2f}"
            )

        except requests.RequestException:
            st.error(
                "The prediction service is unavailable. "
                "Confirm that the FastAPI server is running."
            )


with performance_tab:
    st.header("Model performance")

    try:
        model_info = get_model_info()
        metrics = model_info["test_metrics"]

        metric_columns = st.columns(4)

        metric_columns[0].metric(
            "Precision",
            f"{metrics['precision']:.1%}",
        )
        metric_columns[1].metric(
            "Recall",
            f"{metrics['recall']:.1%}",
        )
        metric_columns[2].metric(
            "F1 score",
            f"{metrics['f1']:.1%}",
        )
        metric_columns[3].metric(
            "PR-AUC",
            f"{metrics['pr_auc']:.1%}",
        )

        chart_data = pd.DataFrame(
            {
                "Metric": [
                    "Precision",
                    "Recall",
                    "F1",
                    "ROC-AUC",
                    "PR-AUC",
                ],
                "Score": [
                    metrics["precision"],
                    metrics["recall"],
                    metrics["f1"],
                    metrics["roc_auc"],
                    metrics["pr_auc"],
                ],
            }
        )

        figure = px.bar(
            chart_data,
            x="Metric",
            y="Score",
            text_auto=".1%",
            color="Score",
            color_continuous_scale="Blues",
        )

        figure.update_layout(
            yaxis_range=[0, 1],
            coloraxis_showscale=False,
            title="Final test-set performance",
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )

        st.info(
            "The test set remained untouched during model selection. "
            "PR-AUC is emphasized because machine failures represent "
            "only 3.39% of the dataset."
        )

    except requests.RequestException:
        st.error(
            "Model information is unavailable. "
            "Confirm that the FastAPI server is running."
        )


with about_tab:
    st.header("About this project")

    st.markdown(
        """
        MachineGuard is an end-to-end machine-learning project that
        demonstrates:

        - Exploratory data analysis
        - Target-leakage prevention
        - Imbalanced classification
        - Train, validation, and test separation
        - Logistic-regression and random-forest comparison
        - Model serialization
        - REST API development with FastAPI
        - Input validation with Pydantic
        - Automated testing with pytest
        - Interactive model delivery with Streamlit

        ### Dataset

        The project uses the synthetic AI4I 2020 Predictive Maintenance
        Dataset from the UCI Machine Learning Repository.

        ### Important limitation

        This application is an educational demonstration. Its predictions
        should not be used to make real industrial maintenance or safety
        decisions.
        """
    )