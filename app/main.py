'''
Author: Oliver
Date: February 2022
'''
# Put the code for your API here.
from fastapi import Body, FastAPI

# Import Union since our Item object will have tags that can be strings or a list.
from typing import Literal, Union 
from shtab import Optional

from pydantic import BaseModel

import os

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

# Pydanitc Model
Workclass = Literal['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked']
Education = Literal['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'] 
Marital_status = Literal['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
Occupation = Literal['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'] 
Relationship = Literal['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
Race = Literal['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
Sex = Literal['Female', 'Male']
Native_country = Literal['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']

# Declare the data object with its components and their type.
class SalaryReq(BaseModel):
    age: int
    #workclass: Workclass
    fnlgt: int
    #education: Education
    education_num: int
    marital_status: Marital_status
    occupation: Occupation
    relationship: Relationship
    race: Race
    sex: Sex
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: Native_country

    class Config:
        schema_extra = {
            "example": {
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
        }

from app.ml.model import inference_current_model

# Instantiate the app.
app = FastAPI()

# Define a GET on the specified endpoint.
@app.get("/")
async def root():
    return {"message": "Hello!"}

@app.post("/salary/")
async def salary_req(item: SalaryReq): 
    # predict salary
    salary = inference_current_model(convert_pydantic2schema(item.dict()))
    print(f"Predicted salary {salary}")
    
    return {"salary": f"Predicted Salary: {salary}"}


def convert_pydantic2schema(json_req: dict): 
    json_req['education-num'] = json_req.pop('education_num')
    json_req['marital-status'] = json_req.pop('marital_status')
    json_req['capital-gain'] = json_req.pop('capital_gain')
    json_req['capital-loss'] = json_req.pop('capital_loss')
    json_req['hours-per-week'] = json_req.pop('hours_per_week')
    json_req['native-country'] = json_req.pop('native_country')
    return json_req

def main_app():
    pass

if __name__ == "__main__":
    try:
        main_app()
        
    except (Exception) as error:
        print("Main error: %s", error)