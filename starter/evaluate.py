'''
Author: Oliver
Date: February 2022
'''
# Add the necessary imports for the starter code.
import pandas as pd
import numpy as np

from ml.data import load_model_artifacts
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

    test_score, preds, acts = validate_model(model, encoder, lb, score, X)
    logger.info(f"Prediction: {preds} vs. Actual Values: {acts}")
    logger.info(f"Train metrics: {score}")
    logger.info(f"Test metrics: {test_score}")

    # tbd- slices performance

    # tbd (opt.) - visualisation
    
    return test_score

if __name__ == "__main__":

    try:
        # load model artifacts
        model, encoder, lb, score = load_model_artifacts("./model")
        X = pd.read_csv("./data/test_cleaned_census.csv")
        logger.info(f"Test dataset ({X.shape})")

        # model evaluation 
        evaluate(model, encoder, lb, score, X)

    except (Exception) as error:
        print("main error: %s", error)