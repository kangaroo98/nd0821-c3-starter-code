'''
Author: Oliver
Date: February 2022
'''
# Script to train machine learning model.
from matplotlib.font_manager import json_dump, json_load
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd

from ml.data import load_model_artifacts
from ml.model import compute_model_metrics
from ml.model import validate_model

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def evaluate(model, encoder, lb, score, X):
    '''
    evaluate the quality of the model using a test dataset X

    model: trained model
    encoder: OneHot encoder used to train the model
    lb: LabelBinarizer used to train the model
    score: score of the trained model
    X: test dataset  
    '''
    logger.info(f"Evaluating test dataset size: {X.shape}")

    preds, acts = validate_model(model, encoder, lb, X)
    logger.info(f"Prediction: {preds} vs. Actual Values: {acts}")

    precision, recall, fbeta = compute_model_metrics(acts, preds)
    test_score = {"name": score["name"], "precision": precision, "recall": recall, "fbeta": fbeta}
    logger.info(f"Train metrics: {score}")
    logger.info(f"Test metrics: {test_score}")

    # tbd - visualisation


if __name__ == "__main__":

    try:

        # load model artifacts
        model, encoder, lb, score = load_model_artifacts("./model")
        X = pd.read_csv("./data/test_cleaned_census.csv")
        logger.info(f"Test dataset ({X.shape})")

        # model evaluation 
        evaluate(model, encoder, lb, score, X)

        # model inference
        # preds = inference(model, encoder, lb, X)

    except (Exception) as error:
        print("main error: %s", error)