'''
Author: Oliver
Date: February 2022
'''
# Script to train machine learning model.
from matplotlib.font_manager import json_dump, json_load
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ml.data import cat_features
from ml.data import process_data
from ml.data import load_model_artifacts
from ml.model import compute_model_metrics
from ml.model import inference
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def inference(file_pth, X):
    
    # load model artifacts
    model, encoder, lb, score = load_model_artifacts(file_pth)
    preds, acts = inference(model, encoder, lb, X, cat_features, 'salary')
    
    logger.info(f"Prediction: {preds} vs. Actual Values: {acts}")

    # tbd - visualise


if __name__ == "__main__":

    try:
        # load test data
        X = pd.read_csv("./data/test_cleaned_census.csv")

        # model inference 
        inference("./model", X)

    except (Exception) as error:
        print("main error: %s", error)