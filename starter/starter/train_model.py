'''
Author: Oliver
Date: February 2022
'''
# Script to train machine learning model.
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ml.data import process_data
from ml.data import save_model_artifacts
from ml.model import train_model
from ml.model import compute_model_metrics
from ml.model import inference
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

num_features = [
    "age",
    "fnlgt",
    "education-num",
    "capital-gain",
    "capital-loss",  
    "hours-per-week",
]


def start_train_pipeline(file_pth):

    # Add code to load in the data.
    df = pd.read_csv(file_pth)
    logger.info(f"File read, training process started: {file_pth}")
    
    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, val = train_test_split(df, test_size=0.20)
    logger.info(f"Data is split into train and test.")

    # Proces the train/test data with the process_data function.
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    X_val, y_val, _, _ = process_data(
        val, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )
    logger.info(f"Train {X_train.shape} and test {X_val.shape} data preprocessed with label salary.")

    # Train and save a model.
    model, model_type = train_model(X_train, y_train)
    logger.info(f"Model trained.")

    test_preds = model.predict(X_val)
    precision, recall, fbeta = compute_model_metrics(y_val, test_preds)
    score = {"name": model_type, "precision": precision, "recall": recall, "fbeta": fbeta}
    logger.info(f"Metrics: {score}")

    return model, encoder, lb, score


if __name__ == "__main__":

    try:
        # train model
        model, encoder, lb, score = start_train_pipeline("./data/train_cleaned_census.csv")

        # save model artifacts
        save_model_artifacts("./model", model, encoder, lb, score)
        logger.info(f"Model, encoder and score saved to: './model'")
        
    except (Exception) as error:
        print("main error: %s", error)