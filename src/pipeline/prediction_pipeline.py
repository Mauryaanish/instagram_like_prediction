import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd

class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self , features):
        try:
            preprocessor_path = os.path.join('artifacts' , 'preprocessor.pkl')
            model_path = os.path.join('artifacts' , 'model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)

            return pred

        except Exception as e:
            logging.info('Exception occured in prediction')
            raise CustomException(e,sys)     

class CustomData:

    def __init__(self,
                Followers : int,
                Time_since_posted : int):
        
        self. Followers =  Followers
        self.Time_since_posted = Time_since_posted
        

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Followers' : [self.Followers],
                'Time_since_posted' : [self.Time_since_posted],
        
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info('DataFrame Gathered')
            return df
        
        except Exception as e:
            logging.info('Exception occured in prediction pipeline')
            raise CustomException(e ,sys)


         