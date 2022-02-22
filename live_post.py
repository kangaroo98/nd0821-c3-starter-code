import requests

response = requests.get('http://udacity-salary.herokuapp.com/')
print(response.status_code)
print(response.json())

body = {
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
response = requests.post('https://udacity-salary.herokuapp.com/salary/', json=body)
print(response.status_code)
print(response.json())