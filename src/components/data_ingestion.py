import pandas as pd
import os

class DataIngestion:

    def initiate_data_ingestion(self):

        df = pd.read_csv(r"C:\Users\semwa\OneDrive\Desktop\Server Failure  prediction\data\predictive_maintenance.csv")

        os.makedirs("artifacts", exist_ok=True)

        df.to_csv("artifacts/raw_data.csv", index=False)

        return "artifacts/raw_data.csv"


if __name__ == "__main__":

    obj = DataIngestion()

    obj.initiate_data_ingestion()

    print("Data ingestion completed")