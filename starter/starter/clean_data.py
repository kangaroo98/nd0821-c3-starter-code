'''
Cleaning a dataset

Author: Oliver
Date: February 2022

'''
import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def remove_duplicates(file_pth):
    '''
    Drop the duplicates of a given dataframe.
    Naming Convention of the cleaned dataset:
    OriginalDirectory/cleaned_OriginalFileName

    Input: file_pth - raw_data
    Output: cleaned_data stored in the same directory 
    '''
    # Add code to load in the data.
    file_dir, file_name = os.path.split(file_pth)
    df = pd.read_csv(file_pth, sep = '\s*,\s*', engine = 'python')
    logger.info(f"Read raw dataset ({df.shape})")

    if (df.duplicated().any()):
        df.drop_duplicates(keep='first', inplace=True)
        df.reset_index(drop=True, inplace=True)
        logger.info("Dropped duplicates...")
        cleaned_pth = str(file_dir+'/cleaned_'+file_name)
        df.to_csv(cleaned_pth, index=False)
        logger.info(f"Saved cleaned dataset ({df.shape}) to: {cleaned_pth}")

if __name__ == "__main__":
    try:
        remove_duplicates("./data/census.csv")

    except (Exception) as error:
        print("Main error: %s", error)
    


