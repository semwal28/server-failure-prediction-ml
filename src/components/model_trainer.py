import os
import sys
from dataclasses import dataclass

from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):

      try:

            logging.info("Splitting training and test input data")

            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )


            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier()
            }
            params = {

                "Logistic Regression": {
                    "penalty": ["l2"],
                    "C": [0.01, 0.1, 1, 10],
                    "solver": ["lbfgs", "liblinear"],
                    "max_iter": [100, 200, 500]
                },

                "Decision Tree": {
                    "criterion": ["gini", "entropy"],
                    "max_depth": [None, 5, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },

                "Random Forest": {
                    "n_estimators": [50, 100, 200],
                    "criterion": ["gini", "entropy"],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2]
                }

}

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)
            best_model_score = max(model_report.values())

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            logging.info(f"Best model found, model name: {best_model_name}, R2 score: {best_model_score}")
            best_model_score = max(model_report.values())

         

            save_object(

                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model

            )
            predicted = best_model.predict(X_test)
            accuracy_score1 = accuracy_score(y_test, predicted)
            return accuracy_score1
       
       
      except Exception as e:
        raise CustomException(e,sys)