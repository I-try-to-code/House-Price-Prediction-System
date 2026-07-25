import os
import pandas as pd
from sklearn.model_selection import train_test_split

from src.logger import logger
from src.exception import CustomException
from src.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):
        try:
            logger.info("Data ingestion started")

            data_source = self.config.source_data_path
            df = pd.read_csv(data_source)
            logger.info("data recieved")

            df.to_csv(self.config.raw_data_path, index=False)
            logger.info("Reading dataset")

            train_set, test_set = train_test_split(df,test_size=0.2, random_state=42)
            logger.info("Train-test split completed")

            train_set.to_csv(self.config.train_data_path, index=False)
            test_set.to_csv(self.config.test_data_path, index=False)
            logger.info("Data ingestion completed")
            return (
               self.config.train_data_path,
            self.config.test_data_path
            )
        
        except Exception as e:
            logger.error(f"Error in data ingestion: {e}")
            raise CustomException(e)

		