import os
import sys
import pandas as pd
from src.components import data_transformation
from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer

# Workflow when this file is started directly:
# 1. Read the student CSV and write artifacts/data.csv, train.csv, and test.csv.
# 2. Data transformation turns those files into numeric arrays and saves artifacts/preprocessor.pkl.
# 3. Model training fits several models, keeps the best one in artifacts/model.pkl,
#    and prints that model's test R2 score.


# Output locations for the three CSV files.
# dataclass fills these paths in automatically, so this class does not need its own __init__.
# All three files go in the artifacts folder when you run from the project root.
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
        Step 1 of the project.

        Load notebook/data/stud.csv.
        Save every row as artifacts/data.csv.
        Hold out 20 percent of the rows as artifacts/test.csv.
        Write the other 80 percent as artifacts/train.csv.
        Return those two paths. The tables themselves stay in the files.
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

            # Save the full table before the split, so the original rows stay available.
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
            # Data transformation opens train.csv and test.csv from these paths.
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
    # Step 1. Write artifacts/data.csv, artifacts/train.csv, and artifacts/test.csv.
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # Step 2. Encode the columns.
    # train_arr and test_arr are numeric tables. math_score is the last column of each.
    # The third return value is the preprocessor path. It is unused here, so it is named _.
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # Step 3. Fit the models, save the best one, and print its test R2 score.
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))
