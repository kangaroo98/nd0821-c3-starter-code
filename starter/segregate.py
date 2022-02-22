'''
Author: Oliver
Date: February 2022
'''
from sklearn.model_selection import train_test_split
import pandas as pd

import yaml
from yaml import CLoader as Loader

import os
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def segregate(artifact_pth):
    '''
    segregate train and test data
    Naming convention of the split files:
    OriginalDirectory/train_OriginalFileName
    OriginalDirectory/test_OriginalFileName

    Input:
        - artifact_pth - path to the dataset to be split
    Output:
        - two files split according to the test_size in params.yaml and 
        stored in the same dir as of the passed file (artifact_pth)
    '''
    # get dvc parameters 
    with open("./params.yaml", "rb") as f:
        params = yaml.load(f, Loader=Loader)

    # read dataset to split into train and test data  
    file_dir, file_name = os.path.split(artifact_pth)
    assert(os.path.exists(file_dir))
    logger.info(f"Directory: {file_dir} File: {file_name}")
    df = pd.read_csv(artifact_pth)

    # Split first in model_dev/test, then we further divide model_dev in train and validation
    logger.info("Splitting data into train/val and test")
    splits = {}

    splits["train"], splits["test"] = train_test_split(
        df,
        test_size=params['segregate_data']['test_size'],
        random_state=params['segregate_data']['random_state'],
        #stratify=df[stratify] if stratify != 'null' else None,
    )

    # Save the artifacts. 
    for split_name, split in splits.items():

        # Set the path on disk 
        data_pth = str(split_name + '_' + file_name)
        temp_path = os.path.join(file_dir, data_pth)
        split.to_csv(temp_path, index=False)
        logger.info(f"Saved {split_name} dataset wiht shape {split.shape} to {temp_path}")


if __name__ == "__main__":
    try:
        segregate("./data/cleaned_census.csv")

    except (Exception) as error:
        print("Main error: %s", error)