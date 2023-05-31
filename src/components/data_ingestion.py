import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from dataclasses import dataclass

## initialize the data ingestion Configuration

@dataclass

class DataIngestionconfig:
    train_data_path = os.path.join('artifacts' , 'train.csv')
    test_data_path = os.path.join('artifacts' , 'test.csv')
    raw_data_path = os.path.join('artifacts' , 'raw.csv')

## create a class for data ingestion

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion Method Starts')

        try:
            df = pd.read_csv(os.path.join('notebooks/data' , 'instagram.csv'))
            logging.info('Data Cleaing Part Starts')

            # columns name rename
            df.rename(columns = {"Time since posted" : 'Time_since_posted'} , inplace= True)

            # this feature is not useful so drop this feature

            df.drop(labels = ['Unnamed: 0' , 'S.No' , 'USERNAME' ,'Caption' ,'Hashtags' ] ,axis = 1 , inplace = True )

            # remove hours in data 

            df['Time_since_posted'] =df['Time_since_posted'].str.replace('hours' , " ")

            # Change data type object to int

            df['Time_since_posted'] = df['Time_since_posted'].astype("int")

            logging.info('Data Cleaing Partm Done')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok= True)
            df.to_csv(self.ingestion_config.raw_data_path , index = False)

            ## train test split

            logging.info('Train test Split')

            train_set, test_set=train_test_split(df , test_size = 0.2 , random_state=40)

            train_set.to_csv(self.ingestion_config.train_data_path , index =False , header = True)
            test_set.to_csv(self.ingestion_config.test_data_path , index = False , header = True)

            logging.info("Ingestion of data is complete")

            return(
                self.ingestion_config.train_data_path ,
                self.ingestion_config.test_data_path
            )



        except Exception as e:
            logging.info('Exception Occured at data ingestion Stage')
            raise CustomException(e,sys)


