'''
pytest configuration

Author: Oliver
Date: February 2022

'''
import pytest
import pandas as pd

from starter.ml.data import load_model_artifacts

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def pytest_addoption(parser):
    parser.addoption("--cleaned_data", action="store")
    parser.addoption("--test_data", action="store")
    parser.addoption("--model_artifacts", action="store")

    
@pytest.fixture(scope="session")
def data(request):

    file_path = request.config.option.cleaned_data
    logger.info(f"MyPara: {file_path}")

    if file_path is None:
        pytest.fail("--file missing on command line")

    raw_data = pd.read_csv(file_path)

    return raw_data

@pytest.fixture(scope="session")
def test_data(request):

    file_path = request.config.option.test_data
    logger.info(f"MyPara: {file_path}")

    if file_path is None:
        pytest.fail("--file missing on command line")

    test_data = pd.read_csv(file_path)

    return test_data

@pytest.fixture(scope="session")
def model_artifacts(request):

    file_dir = request.config.option.model_artifacts
    logger.info(f"MyPara: {file_dir}")

    if file_dir is None:
        pytest.fail("--file missing on command line")

    model, enocder, lb, score = load_model_artifacts(file_dir)

    return model, enocder, lb, score