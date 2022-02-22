'''
Author: Oliver
Date: February 2022
'''
# Add the necessary imports for the starter code.
import pandas as pd
import numpy as np
import json

from ml.data import load_model_artifacts
from ml.data import cat_features
from ml.model import validate_model

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def evaluate(model_pth, data_pth):
    '''
    evaluate the quality of the model using a test dataset X

    model: trained model
    encoder: OneHot encoder used to train the model
    lb: LabelBinarizer used to train the model
    score: score of the trained model
    X: test dataset  
    '''

    model, encoder, lb, score = load_model_artifacts(model_pth)
    df = pd.read_csv(data_pth)
    logger.info(f"Evaluating test dataset size: {df.shape}")

    test_score, preds, acts = validate_model(model, encoder, lb, score, df)
    
    return score, test_score, preds, acts
    
def data_slicing(data_pth, cat, num):

    #model, encoder, lb, score = load_model_artifacts(model_pth)
    df = pd.read_csv(data_pth)

    # slices performance 
    for cat_feat in df[cat].unique():
        avg_value = df[df[cat] == cat_feat][num].mean()
        logger.info(f"{cat}: {cat_feat} {num}: {avg_value}")

def data_slicing_perf(model_pth, data_pth):

    model, encoder, lb, met = load_model_artifacts(model_pth)
    df = pd.read_csv(data_pth)

    # slices performance 
    scores = {}
    for cat_feat in cat_features:
        score = {}
        for cls in df[cat_feat].unique():
            test_score, _, _ = validate_model(model, encoder, lb, met, df[df[cat_feat] == cls])
            score[cls] = test_score['precision']
        scores[cat_feat] = score
    
    logger.info(f"Scores: {scores}")
    return scores


if __name__ == "__main__":

    try:
        
        # model evaluation 
        model_score, test_score, preds, acts = evaluate("./model", "./data/test_cleaned_census.csv")
        logger.info(f"Prediction: {preds} vs. Actual Values: {acts}")
        logger.info(f"Train metrics: {model_score}")
        logger.info(f"Test metrics: {test_score}")

        # slicing
        data_slicing("./data/cleaned_census.csv", "salary" , "age")
        data_slicing("./data/cleaned_census.csv", "salary" , "fnlgt")
        data_slicing("./data/cleaned_census.csv", "salary" , "education-num")
        data_slicing("./data/cleaned_census.csv", "salary" , "hours-per-week")
        data_slicing("./data/cleaned_census.csv", "salary" , "capital-gain")
        data_slicing("./data/cleaned_census.csv", "salary" , "capital-loss")
        data_slicing("./data/cleaned_census.csv", "sex" , "capital-gain")
        data_slicing("./data/cleaned_census.csv", "sex" , "capital-loss")
        data_slicing("./data/cleaned_census.csv", "race" , "capital-gain")
        data_slicing("./data/cleaned_census.csv", "race" , "capital-loss")

        # slices performance 
        scores = data_slicing_perf("./model", "./data/cleaned_census.csv")
        with open('./model/slice_output.txt', 'w') as file:
            json.dump(scores, file)
        

    except (Exception) as error:
        print("main error: %s", error)