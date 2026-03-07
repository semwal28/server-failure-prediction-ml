import sys
from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


if __name__ == "__main__":

    try:

        logging.info("Training pipeline started")

        # DATA INGESTION
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()

        logging.info("Data ingestion completed")


        # DATA TRANSFORMATION
        data_transformation = DataTransformation()

        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
            train_path, test_path
        )

        logging.info("Data transformation completed")


        # MODEL TRAINING
        model_trainer = ModelTrainer()

        model_score = model_trainer.initiate_model_trainer(
            train_arr, test_arr
        )

        logging.info(f"Model training completed. Best accuracy: {model_score}")


    except Exception as e:
        raise CustomException(e, sys)