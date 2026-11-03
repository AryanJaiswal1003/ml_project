import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

# This is the last step of the direct run started from data_ingestion.py.
# It receives the numeric train and test arrays, compares several models,
# saves the best one, and returns that model's test R2 score.


# Where the chosen model is stored. Prediction loads this file later.
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        '''
        train_array and test_array come from data transformation.
        Every column except the last is an input. The last column is math_score.

        Fit each model, keep the one with the highest test R2,
        and refuse to save a model whose test R2 is below 0.6.
        '''
        try:
            logging.info("Splitting Training and Test Imput Data")
            # :-1 means "all columns except the last". -1 means "the last column only".
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Each entry is a fresh, unfitted model.
            # evaluate_model fits them on X_train and y_train.
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbour Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose = False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            params = {
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter': ['best', 'random'],
                    # 'max_features': ['sqrt', 'log2'],
                },
                "Random Forest": {
                    # 'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features': ['sqrt', 'log2', None],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    # 'loss': ['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    # 'criterion': ['squared_error', 'friedman_mse'],
                    # 'max_features': ['auto', 'sqrt', 'log2'],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "K-Neighbour Regressor": {
                    'n_neighbors': [5, 7, 9, 11],
                    # 'weights': ['uniform', 'distance'],
                    # 'algorithm': ['ball_tree', 'kd_tree', 'brute']
                },
                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    # 'loss': ['linear', 'square', 'exponential']
                }
            }



            # report looks like {"Linear Regression": 0.85, "Random Forest": 0.87, ...}.
            # The number is the test R2, not the training R2.
            model_report: dict = evaluate_model(
                X_train = X_train, y_train = y_train,
                X_test = X_test, y_test = y_test,
                models = models, param = params
            )

            # Highest test R2 in the report.
            best_model_score = max(sorted(model_report.values()))

            # Name of the model that produced that score, then the fitted model object itself.
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            # 0.6 is the minimum test R2 accepted for a saved model.
            # Below that, stop instead of writing a weak model to disk.
            if best_model_score < 0.6:
                raise CustomException("No Best Model Found", sys)
            logging.info("Best Model found on both Training & Testing Dataset")

            # Save the already-fitted model so prediction does not have to train again.
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            # Score the saved model once more on the test rows.
            # This is the number printed when data_ingestion.py is run directly.
            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)

            return r2

        except Exception as e:
            raise CustomException(e, sys)
