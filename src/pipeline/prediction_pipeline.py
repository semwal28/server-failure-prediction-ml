import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self, features):
        try:
            model_path = 'C:\\Users\\semwa\\OneDrive\\Desktop\\MLproject\\src\\components\\artifacts\\model.pkl'
            preprocessor_path = 'C:\\Users\\semwa\\OneDrive\\Desktop\\MLproject\\src\\components\\artifacts\\preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)

            return pred

        except Exception as e:
            raise CustomException(e, sys)
        
class CustomData:
    def __init__(self,
                 Type,
                 Air_temperature,
                 Process_temperature,
                 Rotational_speed,
                 Torque,
                 Tool_wear):
        self.Type = Type
        self.Air_temperature = Air_temperature
        self.Process_temperature = Process_temperature
        self.Rotational_speed = Rotational_speed
        self.Torque = Torque
        self.Tool_wear = Tool_wear
    def get_data_as_dataframe(self):
      try:
        custom_data_input_dict = {
            "Type": [self.Type],
            "Air_temperature": [self.Air_temperature],
            "Process_temperature": [self.Process_temperature],
            "Rotational_speed": [self.Rotational_speed],
            "Torque": [self.Torque],
            "Tool_wear": [self.Tool_wear]
        }

        return pd.DataFrame(custom_data_input_dict)

      except Exception as e:
        raise CustomException(e, sys)


    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)

            return pred

        except Exception as e:
            raise CustomException(e, sys)
        
    if __name__ == "__main__":

     data = {
        "Air temperature [K]":[300],
        "Process temperature [K]":[310],
        "Rotational speed [rpm]":[1500],
        "Torque [Nm]":[40],
        "Tool wear [min]":[10]
    }

    df = pd.DataFrame(data)

    pipeline = PredictPipeline()

    print(pipeline.predict(df))