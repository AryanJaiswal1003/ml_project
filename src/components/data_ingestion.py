import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Paths for the three files this step writes. The raw file is the full dataset. Train and test are the split used by later steps.
# dataclass gives these defaults without writing an __init__.
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")


class DataIngestion:
    def __init__(self):
        # Keep the output paths on the object so the method below can use them.
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        # Read the source file, store a copy, then split it into train and test files.
        # Returns the train and test paths for the next component.
        logging.info("Entered the Data Ingestion Method or Component")

        try:
            # Source table of student scores and the categorical columns around them.
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the Dataset as DataFrame')

            # train.csv, test.csv, and data.csv share the artifacts folder.
            # Create that folder once. exist_ok keeps a second run from failing.
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok = True)

            # Save the untouched table before any split, so the original rows stay available.
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)
            logging.info("Train Test Split Initiated")

            # 20% of the rows go to test. random_state=42 makes the same split on every run.
            train_split, test_split = train_test_split(df, test_size = 0.2, random_state = 42)

            # train_test_split returns the same kind of table it received.
            # Wrapping each piece in DataFrame keeps the column names for to_csv.
            train_set = pd.DataFrame(train_split)
            test_set = pd.DataFrame(test_split)

            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)
            logging.info("Data Ingestion Completed")

            # Later steps open these files. They do not receive the tables themselves.
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            # sys gives CustomException the active traceback, including this file and line.
            raise CustomException(e, sys)


if __name__ == '__main__':
    obj = DataIngestion()
    obj.initiate_data_ingestion()