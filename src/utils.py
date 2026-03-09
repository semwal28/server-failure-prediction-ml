import os
import sys
import pandas as pd
import numpy as np
import dill
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,accuracy_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    from src.exception import CustomException
    '''This function is responsible for saving the object in the file path'''

    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models,params):
    from src.exception import CustomException
    '''This function is responsible for evaluating the models'''

    try:
        report = {}

        for i in range(len(models)):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)

            test_model_score = accuracy_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    
    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    from src.exception import CustomException
    '''This function is responsible for loading the object from the file path'''

    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    