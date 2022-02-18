import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

import os
import joblib
from matplotlib.font_manager import json_dump, json_load

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



def save_model_artifacts(file_dir, model, encoder, lb, score):

    # save the model and the OneHot encoder
    model_pth = str(file_dir + '/' + 'model.pkl')
    encoder_pth = str(file_dir + '/' + 'encoder.pkl')
    lb_pth = str(file_dir + '/' + 'lb.pkl')
    score_pth = str(file_dir + '/' + 'score.json')

    if os.path.exists(file_dir):
        joblib.dump(model, model_pth)
        joblib.dump(encoder, encoder_pth)
        joblib.dump(encoder, lb_pth)
        json_dump(score, score_pth)
    else:
        logger.error("Failed to save the model, encoders or metrics. Filepath incorrect!")


def load_model_artifacts(file_dir):

    # load model and encoder
    model_pth = str(file_dir + '/' + 'model.pkl')
    encoder_pth = str(file_dir + '/' + 'encoder.pkl')
    lb_pth = str(file_dir + '/' + 'lb.pkl')
    score_pth = str(file_dir + '/' + 'score.json')

    if os.path.exists(file_dir):
        model = joblib.load(model_pth)
        encoder = joblib.load(encoder_pth)
        lb = joblib.load(lb_pth)
        score = json_load(score_pth)
        logger.info(f"Model, encoders and metrics loaded from directory {file_dir}")
    else:
        logger.error("Failed to load the model, encoders or metrics. Filepath incorrect!")

    return model, encoder, lb, score


def process_data(
    X, categorical_features=[], label=None, training=True, encoder=None, lb=None
):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. This can be used in either training or
    inference/validation.

    Note: depending on the type of model used, you may want to add in functionality that
    scales the continuous data.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    """

    logger.info(f"Dataset shape: {X.shape}")

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
        logger.info("Training true. Encoded.")
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass
        logger.info("Training false. Encoded.")

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb



