'''
clean data testing

Author: Oliver
Date: February 2022
'''
from http.client import responses
from urllib import response
from fastapi.testclient import TestClient

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

from main import app

client = TestClient(app)


def test_get_path():
    '''
    check standard get method and its response 
    '''
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello!"}

def test_post_salary_invalid_cat_value():
    '''
    post method with wrong parameter value for Realtionship-Literal - HTTP Status Coder 422 - Unprocessable Entity
    '''
    # realtionship invalid - 'best friend'
    response = client.post(
        "/salary/",
        json={
            "age": 60,
            "fnlgt":100000,
            "education_num":10,
            "marital_status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "best friend",
            "race": "Black",
            "sex": "Male",
            "capital_gain": 5000,
            "capital_loss": 300,
            "hours_per_week": 60,
            "native_country": "United-States"
        }
    )
    assert response.status_code == 422


def test_post_salary_1():
    '''
    post method check with a valid json body  
    '''
    response = client.post(
        "/salary/",
        json={
            "age": 22,
            "fnlgt":254547,
            "education_num":10,
            "marital_status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Other-relative",
            "race": "Black",
            "sex": "Female",
            "capital_gain": 0,
            "capital_loss": 0,
            "hours_per_week": 30,
            "native_country": "Jamaica"
        }
    )
    logger.info(f"Response should be <=50K?: {response.json()}")
    assert response.status_code == 200  
    assert response.json() == {"salary": "Predicted Salary: ['<=50K']"}


def test_post_salary_2():
    '''
    post method check with a valid json body  
    '''
    response = client.post(
        "/salary/",
        json={
            "age": 47,
            "fnlgt":294913,
            "education_num":15,
            "marital_status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital_gain": 99999,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "United-States"
        }
    )
    logger.info(f"Response should be >50K?: {response.json()}")
    assert response.status_code == 200
    assert response.json() == {"salary": "Predicted Salary: ['>50K']"}

