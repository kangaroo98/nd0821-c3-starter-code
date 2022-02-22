'''
clean data testing

Author: Oliver
Date: February 2022
'''
import pytest
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder

from app.ml.data import process_data
from app.inference import inference_test_data
from app.evaluate import validate_model

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()



def test_column_presence(data):
    '''
    check for matching columns & types
    '''
    required_columns = {
        "age": pd.api.types.is_integer_dtype,
        "workclass": pd.api.types.is_string_dtype,
        "fnlgt": pd.api.types.is_integer_dtype,
        "education": pd.api.types.is_string_dtype,
        "education-num": pd.api.types.is_integer_dtype,
        "marital-status": pd.api.types.is_string_dtype,
        "occupation": pd.api.types.is_string_dtype,
        "relationship": pd.api.types.is_string_dtype,
        "race": pd.api.types.is_string_dtype,
        "sex": pd.api.types.is_string_dtype,
        "capital-gain": pd.api.types.is_integer_dtype,
        "capital-loss": pd.api.types.is_integer_dtype,  
        "hours-per-week": pd.api.types.is_integer_dtype,
        "native-country": pd.api.types.is_string_dtype,
        "salary": pd.api.types.is_string_dtype
    }

    # Check column presence  
    assert set(data.columns).issuperset(set(required_columns.keys()))

def test_column_presence_and_type(data):
    '''
    check for matching columns & types
    '''
    required_columns = {
        "age": pd.api.types.is_integer_dtype,
        "workclass": pd.api.types.is_string_dtype,
        "fnlgt": pd.api.types.is_integer_dtype,
        "education": pd.api.types.is_string_dtype,
        "education-num": pd.api.types.is_integer_dtype,
        "marital-status": pd.api.types.is_string_dtype,
        "occupation": pd.api.types.is_string_dtype,
        "relationship": pd.api.types.is_string_dtype,
        "race": pd.api.types.is_string_dtype,
        "sex": pd.api.types.is_string_dtype,
        "capital-gain": pd.api.types.is_integer_dtype,
        "capital-loss": pd.api.types.is_integer_dtype,  
        "hours-per-week": pd.api.types.is_integer_dtype,
        "native-country": pd.api.types.is_string_dtype,
        "salary": pd.api.types.is_string_dtype
    }

    for col_name, format_verification_funct in required_columns.items():
        assert format_verification_funct(data[col_name]), f"Column {col_name} failed test {format_verification_funct}"

def test_check_duplicates(data):
    '''
    check for duplicate rows
    '''
    assert(data.duplicated().any() == False)

def test_check_countries(data):
    '''
    check for unknown countries 
    '''
    assert(data['native-country'].loc[data['native-country'] == '?'].any() == False)

def test_preprocess_train(test_data):
    '''
    check for preprocessing function for parameter handling
    '''
    try:
        process_data(test_data, process_type='train', encoder=None, lb=None)

    except Exception as err:
        logger.error(f"TEST FAILED: {err}")
        assert False

def test_preprocess_train_1(test_data):
    '''
    check for preprocessing function for parameter handling
    '''
    try:
        process_data(test_data, process_type='train', encoder=OneHotEncoder(), lb=None)
        pytest.fail("process_data passed without AssertionError")

    except AssertionError:
        logger.info("TEST SUCCESSFUL: Encoder must be None.")
    except Exception as err:
        logger.error(f"TEST FAILED: {err}")
        assert False

def test_preprocess_train_2(test_data):
    '''
    check for preprocessing function for parameter handling
    '''
    try:
        process_data(test_data, process_type='train', encoder=None, lb=LabelBinarizer())
        pytest.fail("process_data passed without AssertionError")

    except AssertionError:
        logger.info("TEST SUCCESSFUL: Encoder must be None.")
    except Exception as err:
        logger.error(f"TEST FAILED: {err}")
        assert False

def test_preprocess_val(test_data):
    '''
    check for preprocessing function for parameter handling
    '''
    try:
        process_data(test_data, process_type='val_test', encoder=None, lb=None)
        pytest.fail("process_data passed without AssertionError")

    except AssertionError:
        logger.info("TEST SUCCESSFUL: Encoders must not be None.")
    except Exception as err:
        logger.error(f"TEST FAILED: {err}")
        assert False

def test_preprocess_inf(model_artifacts):
    '''
    check for preprocessing function for parameter handling
    '''
    try:
        process_data(pd.DataFrame(inference_test_data), process_type='inference', encoder=model_artifacts[1], lb=model_artifacts[2])
        pytest.fail("process_data passed without AssertionError")

    except AssertionError:
        logger.info("TEST SUCCESSFUL: LabelBinarizes must be None.")
    except Exception as err:
        logger.error(f"TEST FAILED: {err}")
        assert False

def test_model_score(test_data, model_artifacts):
    '''
    check test data complies with trained model score (range within 5%)     
    '''
    test_score, _, _ = validate_model(
            model_artifacts[0], 
            model_artifacts[1], 
            model_artifacts[2], 
            model_artifacts[3], 
            test_data
        )
    range_min = float(model_artifacts[3]['precision']-0.05)
    range_max = float(model_artifacts[3]['precision']+0.05)
    logger.info(f"Range Min: {range_min} Range Max: {range_max}")
    logger.info(f"Testing metrics train vs test data: {model_artifacts[3]['precision']} vs {test_score['precision']}")

    if not ((test_score["precision"] > range_min) and (test_score["precision"] < range_max)):
        assert False

def test_data_shape(data):
    """ If your data is assumed to have no null values then this is a valid test. """
    assert data.shape == data.dropna().shape, "Dropping null changes shape."