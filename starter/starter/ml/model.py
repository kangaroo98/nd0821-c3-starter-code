from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.metrics import classification_report, RocCurveDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

param_grid = { 
    'n_estimators': [20, 50],
    'max_features': ['auto', 'sqrt'],
    'max_depth' : [4,5,10],
    'criterion' :['gini', 'entropy']
}

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
        Trained machine learning model.
    """
    # logistic regression
    # lrc = LogisticRegression(max_iter=1000)
    # lrc.fit(X_train, y_train)

    # return lrc, str("LogisticRegression")

    # random forest
    rfc = RandomForestClassifier(random_state=42)
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
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


def inference(model, encoder, lb, X, cat_features, label):
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
    # Preprocess and predict
    X_trained_columns, y_actual_results, _, _ = process_data(X, categorical_features=cat_features,
                                    label=label, training=False, encoder=encoder, lb=lb)
    predictions = model.predict(X_trained_columns)
    logger.info(f"Predicted: Salaray prediction: {predictions} Actual results: {y_actual_results}")

    return predictions, y_actual_results 

