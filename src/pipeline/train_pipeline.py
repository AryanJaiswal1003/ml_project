import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging


class TrainPipeline:
    '''
    Run the training steps in order.

    Data ingestion writes artifacts/train.csv and artifacts/test.csv.
    Data transformation turns those files into numeric arrays and saves artifacts/preprocessor.pkl.
    Model training compares the models, saves the best one as artifacts/model.pkl,
    and returns that model's test R2 score.
    '''

    def initiate_training(self):
        try:
            logging.info("Training pipeline started")

            ingestion = DataIngestion()
            train_data_path, test_data_path = ingestion.initiate_data_ingestion()

            transformation = DataTransformation()
            train_array, test_array, _ = transformation.initiate_data_transformation(
                train_data_path, test_data_path
            )

            trainer = ModelTrainer()
            r2_score = trainer.initiate_model_trainer(train_array, test_array)

            logging.info("Training pipeline completed")
            return r2_score

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    print(TrainPipeline().initiate_training())
