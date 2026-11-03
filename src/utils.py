import os
import sys
import dill

import numpy as np
import pandas as pd

from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    '''
    Save a Python object to disk so another run can load it later.

    Data transformation calls this with the fitted preprocessor.
    Model training calls this with the chosen model.
    The files are artifacts/preprocessor.pkl and artifacts/model.pkl.
    dill can store those objects, including values they learned from the training rows.
    '''
    try:
        # file_path looks like artifacts/preprocessor.pkl or artifacts/model.pkl.
        # dirname is the folder part: artifacts.
        dir_path = os.path.dirname(file_path)
        # Create the folder when this function is used before that folder exists.
        # exist_ok=True lets the folder already exist.
        os.makedirs(dir_path, exist_ok = True)

        # "wb" means write binary. The file holds bytes, not readable text.
        # dill.dump writes the whole object, including the numbers it learned while fitting.
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        # Report the original error together with this file and line number.
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, models, param):
    '''
    Train every model in the dictionary and score it on the test rows.

    X_train and X_test are the input columns.
    y_train and y_test are the math_score answers.
    models maps a name, such as "Linear Regression", to an unfitted model.

    Returns a dictionary of name -> test R2 score.
    R2 is 1 when predictions match the answers, and lower when they do not.
    The training score is calculated as well, but the report keeps the test score,
    because that is the score on students the model did not train on.
    '''
    try:
        report = {}
        # Walk the dictionary by position so the name and the model stay paired.
        for i in range(len(list(models))):
            model = list(models.values())[i]
            params = param[list(models.keys())[i]]

            grid_search = GridSearchCV(model, params, cv = 3)
            grid_search.fit(X_train, y_train)


            # fit teaches this model using the training rows only.
            model.set_params(**grid_search.best_params_)
            model.fit(X_train, y_train)

            # Predict math_score for the rows the model has seen and for the held-out rows.
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store the test score under the model's name.
            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
