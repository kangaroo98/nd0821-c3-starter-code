from sklearn.model_selection import train_test_split
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def segregate(artifact_pth, test_size=0.2, random_state=42):

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
        test_size=test_size,
        random_state=random_state,
        #stratify=df[stratify] if stratify != 'null' else None,
    )

    # Save the artifacts. 
    for split_name, split in splits.items():

        # Set the path on disk 
        data_pth = str(split_name + '_' + file_name)
        temp_path = os.path.join(file_dir, data_pth)
        split.to_csv(temp_path)
        logger.info(f"Saved {split_name} dataset wiht shape {split.shape} to {temp_path}")


if __name__ == "__main__":
    try:
        segregate("./data/cleaned_census.csv")

    except (Exception) as error:
        print("Main error: %s", error)