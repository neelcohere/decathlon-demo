# ../app/app.py

import os
import cohere
import sklearn
import shap
import numpy as np
import pandas as pd
from typing import Optional, Union, Any

import streamlit as st

from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
load_dotenv()

co = cohere.Client(api_key="Ldp4S4gZ2FUAwZVvoievRf44t4Y8HPFDPsKcQNem", base_url="https://stg.api.cohere.com/v1")
FEATURE_NAMES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]

def generate(prompt: str, *args, **kwargs) -> str:
    return co.chat(
        message=prompt,
        **kwargs
    ).text


def setup_model(X: Union[pd.DataFrame, Any], y: Any) -> LinearRegression:
    model = LinearRegression()
    model.fit(X, y)
    return model


def calculate_shap_values(model: LinearRegression, X: Union[pd.DataFrame, Any]) -> shap.Explainer:
    X100 = shap.utils.sample(X, 100)
    explainer = shap.Explainer(model.predict, X100, feature_names=FEATURE_NAMES)
    shap_values = explainer(X).values
    return shap_values


def build_shap_value_prompt(shap_values: Any, focus_features: list[str]) -> str:
    PROMPT = \
"""You are a helpful data scientist assistant that will help answer questions about models and their explainability.

## Task
Provide a verbose explaination about model explainability based on the provided Features and their corresponding shapley values for 5 sampled predictions by the model.
After evaluating each prediction, provide a summary of findings over all the predictions and point out any relevant trends in the shapley values' impact to the model predictions.
{focus_features}

## Features and Shapley Values
{feats_and_vals}
"""
    feats_and_vals = ""
    for i, _shap_value in enumerate(shap_values):
        feats_and_vals += f"### Prediction {i + 1}\n"
        _feat_and_val = "   \n".join([f"{feat}: {val}" for feat, val in zip(FEATURE_NAMES, _shap_value)])
        feats_and_vals += _feat_and_val + "\n"
    
    if focus_features is not None:
        focus_features = \
"""Evaluate all features, but place a special focus on the following features: """ + ", ".join(focus_features)
    
    return PROMPT.format(feats_and_vals=feats_and_vals, focus_features=focus_features)

X, y = shap.datasets.california(n_points=1_000)
model = setup_model(X, y)
shap_values = calculate_shap_values(model, X)


def main():
    # Set the title of the app
    st.title("Model Explainability Assistant")

    if 'resp' not in st.session_state:
        st.session_state["resp"] = "Please generate the analysis first"
    if 'selected_method' not in st.session_state:
        st.session_state["selected_method"] = "Shapley Value"

    # Sidebar for model explainability methods
    st.sidebar.header("Choose Explainability Method")
    methods = ["Shapley Value", "LIME", "Partial Dependence Plot"]
    selected_method = st.sidebar.selectbox("Select Method", methods)
    st.session_state["selected_method"] = selected_method

    # Sidebar for file upload
    st.sidebar.header("Upload Data and Model")
    data_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])
    model_file = st.sidebar.file_uploader("Upload your model (pickle)", type=["pkl", "pickle"])

    # Main panel
    st.header(f"{st.session_state['selected_method']} Analysis")
    
    # Multi-select field
    focus_features = st.multiselect("**Select focus features**", FEATURE_NAMES, help="Select features that you want the analysis to be focused on. Leave blank to analyze all available features.")
    st.caption(f"All features: {FEATURE_NAMES}")
    num_preds = st.slider("**Num. of Predictions**", min_value=3, max_value=int(shap_values.shape[0] * 0.1), value=5, help="Select the number of predictions and corresponding shapley values to include in generation prompt. Can run for upto 10% of the total dataset.")
    if len(focus_features) == 0:
        focus_features = FEATURE_NAMES

    btn_generate = st.button("Generate")
    st.divider()

    # Generate button
    if btn_generate:
        with st.status("**Generating analysis...**") as status:
            prompt = build_shap_value_prompt(shap_values[:num_preds], focus_features)
            resp = generate(prompt, model="command-r-plus-08-2024")
            status.update(
                label="**Generated Analysis**", state="complete", expanded=True
            )
            st.success("Analysis complete!", icon="✅")
            st.session_state["resp"] = resp
            st.markdown(st.session_state["resp"])


if __name__ == "__main__":
    main()
