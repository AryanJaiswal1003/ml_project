import os
import sys
import pandas as pd
from src.components import data_transformation
from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation, DataTransformationConfig

# This file is the first step of the project.
# It reads the original student table, keeps a full copy, and writes two smaller files:
# one for training a model and one for testing that model.
# The next step, data transformation, opens those two files. It does not receive the tables directly.


# Output locations for the three CSV files.
# dataclass fills these paths in automatically, so this class does not need its own __init__.
# All three files go in the artifacts folder, next to the project root when you run from D:\ML-PROJECT.
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")


class DataIngestion:
    def __init__(self):
        # Keep the three paths on this object so initiate_data_ingestion can use them.
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        '''
        Load notebook/data/stud.csv.

        Save every row as artifacts/data.csv.
        Then hold out 20 percent of the rows as artifacts/test.csv
        and write the other 80 percent as artifacts/train.csv.
        Return those two paths.
        '''
        logging.info("Entered the Data Ingestion Method or Component")

        try:
            # One row is one student.
            # Columns include gender, race, parental education, lunch, test preparation,
            # and the math, reading, and writing scores.
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Read the Dataset as DataFrame')

            # train.csv, test.csv, and data.csv share the artifacts folder.
            # Create that folder if it is missing.
            # exist_ok=True means a second run does not crash when the folder already exists.
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok = True)

            # Save the full table before the split.
            # index=False leaves out the pandas row numbers.
            # header=True keeps the column names in the first line of the file.
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)
            logging.info("Train Test Split Initiated")

            # test_size=0.2 sends 20 percent of the students to the test set and 80 percent to training.
            # random_state=42 shuffles the rows the same way every time, so the files stay repeatable.
            train_split, test_split = train_test_split(df, test_size = 0.2, random_state = 42)

            # The split already returns tables. Wrapping them in DataFrame keeps the column names
            # attached, which to_csv needs in order to write a header row.
            train_set = pd.DataFrame(train_split)
            test_set = pd.DataFrame(test_split)

            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)
            logging.info("Data Ingestion Completed")

            # Return paths, not the tables.
            # Data transformation will open train.csv and test.csv from these paths.
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            # Pass sys so CustomException can read the traceback and report this file and line.
            raise CustomException(e, sys)


# This block runs only when you start this file directly:
# python src/components/data_ingestion.py
# Importing DataIngestion from another file does not run the lines below.
if __name__ == '__main__':
    # Write artifacts/data.csv, artifacts/train.csv, and artifacts/test.csv.
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # Hand the two paths to the next step.
    # That step encodes the columns and saves artifacts/preprocessor.pkl.
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)
