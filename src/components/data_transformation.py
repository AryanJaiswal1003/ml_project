import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


# This file turns the raw train.csv and test.csv into numeric arrays a model can learn from.
# It also saves the exact cleaning steps, so a later prediction uses the same rules.


# Path of the saved preprocessor. Prediction loads this file instead of rebuilding the steps.
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        # Store the output path on the object so both methods can reach it.
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        '''
        Build the cleaning steps, but do not apply them yet.

        Number columns and category columns need different steps.
        math_score is what the model will predict, so it is left out of both lists.
        The returned object only remembers which steps belong to which columns.
        The actual numbers (medians, scales, category columns) are learned later,
        when fit_transform runs on the training file.
        '''
        try:
            # writing_score and reading_score are already numbers, such as 72 or 90.
            numerical_cols = ["writing_score", "reading_score"]
            # These columns hold labels, such as "female", "group B", or "standard".
            categorical_cols = [
                "gender", "race_ethnicity",
                "lunch", "test_preparation_course",
                "parental_level_of_education"
            ]

            # A Pipeline runs its steps in order on the same columns.
            # 1. median fill: a blank score becomes the middle score from the training rows.
            # 2. StandardScaler: subtract the training average and divide by the spread,
            #    so a column like 0-100 does not overpower smaller columns.
            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy = "median")),
                    ("scaler", StandardScaler())
                ])

            # Category steps, also in order:
            # 1. most_frequent fill: a blank label becomes the label that appears most often.
            # 2. one-hot encoding: one text column becomes several 0/1 columns.
            #    Example: gender "female"/"male" becomes two columns, one of them 1.
            #    sparse_output=False stores a normal table of numbers.
            #    The default sparse form cannot be passed to StandardScaler.
            # 3. StandardScaler: scale those 0/1 columns the same way as the scores.
            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy = "most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(sparse_output=False)),
                    ("scaler", StandardScaler())
                ])

            logging.info(f"Number columns: {numerical_cols}")
            logging.info(f"Category columns: {categorical_cols}")

            # ColumnTransformer sends each column list through its own pipeline,
            # then places the results side by side in one numeric table.
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_cols),
                ("cat_pipeline", cat_pipeline, categorical_cols)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        '''
        Read the two files written by data ingestion.

        Learn the cleaning rules from the training rows only.
        Apply those same rules to the test rows.
        Stick math_score back onto the end of each result.
        Save the fitted preprocessor for prediction.
        '''
        try:
            # These CSVs are artifacts/train.csv and artifacts/test.csv.
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test files loaded")
            logging.info("Building the preprocessor")

            # This call only builds the empty steps. Nothing is learned yet.
            preprocessing_obj = self.get_data_transformer_obj()

            # The model should predict math_score from the other columns.
            # Drop it from the inputs, and keep a separate copy as the answer column.
            target_column = "math_score"

            input_feature_train_df = train_df.drop(columns = [target_column], axis = 1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns = [target_column], axis = 1)
            target_feature_test_df = test_df[target_column]

            logging.info("Fitting the preprocessor on the training data and applying it to both sets")

            # fit_transform looks at the training rows and remembers:
            # the median of each score, the most common label in each category,
            # which category columns to create, and the average and spread used for scaling.
            # transform uses those remembered values on the test rows.
            # It must not learn from the test rows, or the test would no longer be unseen data.
            input_feature_train_array = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_array = preprocessing_obj.transform(input_feature_test_df)

            # The arrays above contain only input columns.
            # np.c_ adds math_score as the last column of each array.
            # The training step can then take every column except the last as X,
            # and take the last column as y.
            train_arr = np.c_[input_feature_train_array, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_array, np.array(target_feature_test_df)]

            logging.info("Saving the preprocessor")

            # Save the fitted object, not the arrays.
            # A new student row later must be encoded with these same medians, categories, and scales.
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            # Caller receives the ready training array, the ready test array,
            # and the path of artifacts/preprocessor.pkl.
            return (
                train_arr, test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
