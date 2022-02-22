'''
Author: Oliver
Date: February 2022
'''
# Add the necessary imports for the starter code.
from dataclasses import dataclass
import pandas as pd
import numpy as np
import json
import requests

from ml.data import load_model_artifacts
from ml.model import inference_current_model
from ml.model import inference

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# test data
json_data = {
    "age": 30,
    "fnlgt":100000,
    "education-num":10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 5000,
    "capital-loss": 300,
    "hours-per-week": 60,
    "native-country": "United-States"
}

# test data for inferencing
inference_test_data = {
#            "workclass": ['Private', 'Self-emp-not-inc', 'Local-gov'],
#            "education": ["HS-grad","Some-college","Bachelors"],
            "marital-status": ['Never-married', 'Married-civ-spouse', 'Divorced'],
            "occupation": ["Sales","Tech-support","Priv-house-serv"],
            "relationship": ["Not-in-family","Husband","Other-relative"],
            "race": ["Black","Asian-Pac-Islander","White"],
            "sex": ["Male","Female","Male"],
            "native-country": ["United-States","Nicaragua","Honduras"],
            "age": [44,36,26],
            "fnlgt": ["80000","160000","40000"],
            "education-num": ['9', '2', '12'],
            "capital-gain": [5000,500,0],
            "capital-loss": [0,0,0],
            "hours-per-week": ["25", "60","40"]
}

if __name__ == "__main__":

    try:

        # model inference test - inference_test_data defined in data.py
        model, encoder, lb, _ = load_model_artifacts("./model")
        test_df = pd.DataFrame(inference_test_data)
        logger.info(f"Predict test data: {test_df}")
        test_preds = inference(model, encoder, lb, test_df)
        logger.info(f"Test prediction: {test_preds}")

        # inference with json
        logger.info("Starting inference with json object....")
        salary = inference_current_model(json_data)
        logger.info(f"Salary prediction: {salary}")

    except (Exception) as error:
        print("main error: %s", error)