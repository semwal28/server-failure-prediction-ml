import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")

        try:
            project_root = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../../")
            )
            dataset_path = os.path.join(
                project_root, "notebook", "data", "predictive_maintenance.csv"
            )

            logging.info(f"Reading dataset from {dataset_path}")

            df = pd.read_csv(dataset_path)

            # Create artifacts folder if not exists
            os.makedirs("artifacts", exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            logging.info("Train-test split initiated")

            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42
            )

            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False
            )

            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False
            )

            logging.info("Data ingestion completed successfully")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer()
    r2 = modeltrainer.initiate_model_trainer(train_arr, test_arr)

    print(f"Final R2 Score: {r2}")