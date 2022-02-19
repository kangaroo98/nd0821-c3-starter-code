'''
Author: Oliver
Date: February 2022
'''
import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

import os
import joblib
from matplotlib.font_manager import json_dump, json_load

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# # all columsn representing the given dataset
# cat_features = [
#     "workclass",
#     "education",
#     "marital-status",
#     "occupation",
#     "relationship",
#     "race",
#     "sex",
#     "native-country",
# ]
# num_features = [
#     "age",
#     "fnlgt",
#     "education-num",
#     "capital-gain",
#     "capital-loss",  
#     "hours-per-week",
# ]
# target = "salary"
# used columns in model training
cat_features = [
    "workclass",
    "education",
    "marital-status",
]
num_features = [
    "age",
]
target = "salary"

process_type = [
    'train',
    'val_test',
    'inference'
]


def save_model_artifacts(file_dir, model, encoder, lb, score):
    '''
    save model artifacts in file_dir
    Naming Convention and artifacts of the model:
    model.pkl - model
    encoder.pkl - OneHot Encoder
    lb.pkl - Binarizer
    score - Metrics of the model

    Input: model, encoder, lb, score - as described above
    Output: -
    '''
    # save the model and the OneHot encoder
    model_pth = str(file_dir + '/' + 'model.pkl')
    encoder_pth = str(file_dir + '/' + 'encoder.pkl')
    lb_pth = str(file_dir + '/' + 'lb.pkl')
    score_pth = str(file_dir + '/' + 'score.json')

    if os.path.exists(file_dir):
        joblib.dump(model, model_pth)
        joblib.dump(encoder, encoder_pth)
        joblib.dump(lb, lb_pth)
        json_dump(score, score_pth)
    else:
        logger.error("Failed to save the model, encoders or metrics. Filepath incorrect!")


def load_model_artifacts(file_dir):
    '''
    load model artifacts stored in file_dir
    Naming Convention and artifacts of the model:
    model.pkl - model
    encoder.pkl - OneHot Encoder
    lb.pkl - Binarizer
    score - Metrics of the model

    Input: file_dir - Directory of the model
    Output: model, encoder, lb, score - as described above
    '''
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


def process_data(dataset, process_type='train', encoder=None, lb=None):
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
    assert(dataset.shape[0] > 1)
    assert(set(dataset[cat_features]).issubset(set(dataset)))
    
    logger.info(f"Preprocessing dataset shape: {dataset.shape}")

    if (process_type != 'inference'):
        # training/validation/test
        assert(target in set(dataset.columns))
        labels = dataset[target]
        features = dataset.drop([target], axis=1)
    else:
        # inference - no label
        labels = np.array([])
        features = dataset.copy()

    X_categorical = features[cat_features].values
    X_continuous = features[num_features].values
    logger.info(f"X_categorical: {X_categorical.shape} X_continuous: {X_continuous.shape} ")

    if (process_type == 'train'):
        # training
        logger.info("Preparing data for training...Encoding...")
        assert((encoder is None) and (lb is None))

        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        labels = lb.fit_transform(labels.values).ravel()

    else:
        # validation/test/inference
        logger.info("Preparing data for validation/test/inference...Encoding...")
        assert(encoder is not None)
        
        X_categorical = encoder.transform(X_categorical)
        if (process_type == 'val_test'):
            # validation/test
            logger.info("Preparing label data for validation/test...Encoding...")
            assert(lb is not None)
            labels = lb.transform(labels.values).ravel()
        
    logger.info("Dataset encoded.")
    features = np.concatenate([X_categorical, X_continuous], axis=1)

    return features, labels, encoder, lb



