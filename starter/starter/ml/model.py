'''
Author: Oliver
Date: February 2022
'''
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.metrics import classification_report, RocCurveDisplay
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import yaml
from yaml import CLoader as Loader

from starter.ml.data import process_data

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        - Trained machine learning model.
        - Name of the model
    """
    # logistic regression
    # lrc = LogisticRegression(max_iter=1000)
    # lrc.fit(X_train, y_train)

    # return lrc, str("LogisticRegression")

    # read model parameter
    with open("./params.yaml", "rb") as f:
        params = yaml.load(f, Loader=Loader)
    logger.info(f"DVC params.yaml: {params}")    

    # random forest
    rfc = RandomForestClassifier(random_state=params['train_model']['random_state'])
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=params['train_model']['param_grid'], cv=5)
    cv_rfc.fit(X_train, y_train)
    
    return cv_rfc.best_estimator_, str("RandomForest")


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def validate_model(model, encoder, lb, score, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model :
    encoder: 
    lb:
    score:
        Trained machine learning model.
    
    X : np.array
        Data used for validation.

    Returns
    -------
    preds : np.array-Predictions from the model.
    y_actual_results: np.array-Actual results of the dataset.
    """
    logger.info(f"Inference input dataset X: ({X.shape})")

    # preprocess and predict
    X_trained_columns, y_actual_results, _, _ = process_data(X, process_type='val_test', encoder=encoder, lb=lb)
    logger.info(f"X_trained_columns: ({X_trained_columns.shape}) y_actual_results: ({y_actual_results.shape}) ")

    # predict 
    predictions = model.predict(X_trained_columns)
    logger.info(f"Encoded predictions: {predictions} Encoded actual results: {y_actual_results}")

    precision, recall, fbeta = compute_model_metrics(y_actual_results, predictions)
    test_score = {"name": score["name"], "precision": precision, "recall": recall, "fbeta": fbeta}

    logger.info(f"Validated Model Precision: {test_score}")

    return test_score, lb.inverse_transform(predictions), lb.inverse_transform(y_actual_results)


def inference(model, encoder, lb, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    logger.info(f"Inference input dataset X columns: ({X.columns})")

    # preprocess and predict
    X_trans_cols, y_actual_results, _, _ = process_data(X, process_type='inference', encoder=encoder)
    logger.info(f"X_trained_columns: ({X_trans_cols.shape}) y_actual_results: ({y_actual_results.shape}) ")

    # predict 
    predictions = model.predict(X_trans_cols)

    return lb.inverse_transform(predictions)

