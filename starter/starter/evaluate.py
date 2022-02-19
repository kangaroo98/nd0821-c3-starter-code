'''
Author: Oliver
Date: February 2022
'''
# Add the necessary imports for the starter code.
import pandas as pd
import numpy as np

from ml.data import load_model_artifacts
from ml.model import compute_model_metrics
from ml.model import validate_model
from ml.model import inference

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
        model, encoder, lb, score = load_model_artifacts("/Users/Oliver/Development/nd0821-c3-starter-code/starter/model")
        X = pd.read_csv("/Users/Oliver/Development/nd0821-c3-starter-code/starter/data/test_cleaned_census.csv")
        logger.info(f"Test dataset ({X.shape})")

        # model evaluation 
        evaluate(model, encoder, lb, score, X)

        # model inference test
        data = {
            "workclass": ['Private', 'Private', 'Private'],
            "education": ['Never-married', 'Never-married', 'Never-married'],
            "marital-status": ['Bachelors', 'Bachelors', 'Bachelors'],
            "age": [55,20,5],
        }
        
        test_df = pd.DataFrame(data)
        logger.info(f"Predict test data: {test_df}")
        test_preds = inference(model, encoder, lb, test_df)
        logger.info(f"Test prediction: {test_preds}")

    except (Exception) as error:
        print("main error: %s", error)