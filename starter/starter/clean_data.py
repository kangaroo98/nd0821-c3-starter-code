'''
Cleaning a dataset

Author: Oliver
Date: February 2022

'''
import pandas as pd

import os
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def remove_duplicates(df):
    '''
    Drop the duplicates of a given dataframe.

    Input: df - raw_data
    Output: df - cleaned_data  
    '''
    # remove duplicates but keep the first occurance 
    if (df.duplicated().any()):
        df.drop_duplicates(keep='first', inplace=True)
        df.reset_index(drop=True, inplace=True)
        logger.info(f"Dropped duplicates...New size: {df.shape}")

def remove_unknown_country(df):
    '''
    Drop all unkonwn countries.

    Input: df - raw_data
    Output: df - cleaned_data  
    '''
    # remove duplicates but keep the first occurance 
    if (df['native-country'].loc[df['native-country'] == '?'].any()):
        df.drop(df.loc[df['native-country'] == '?'].index, inplace=True)
        df.reset_index(drop=True, inplace=True)
        logger.info(f"Dropped unkoknwn countries...New size: {df.shape}")


if __name__ == "__main__":
    try:
        '''
        Naming Convention of the cleaned dataset:
        OriginalDirectory/cleaned_OriginalFileName
        '''
        file_pth = "./data/census.csv"
        file_dir, file_name = os.path.split(file_pth)
        df = pd.read_csv(file_pth, sep = '\s*,\s*', engine = 'python')
        logger.info(f"Read raw dataset ({df.shape})")

        # clean data
        remove_duplicates(df)
        remove_unknown_country(df)

        cleaned_pth = str(file_dir+'/cleaned_'+file_name)
        df.to_csv(cleaned_pth, index=False)
        logger.info(f"Saved cleaned dataset ({df.shape}) to: {cleaned_pth}")

    except (Exception) as error:
        print("Main error: %s", error)
    


