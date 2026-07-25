import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.utils import save_object
from src.logger import logger
from src.exception import CustomException
from src.config.configuration import Configuration

class DataTransformation:

    def __init__(self):
        self.config = Configuration()

    def get_preprocessor(self, X_train):
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])
        num_cols = X_train.select_dtypes(
            include=["int64", "float64"]
        ).columns

        cat_cols = X_train.select_dtypes(
            include=["object"]
        ).columns

        preprocessor = ColumnTransformer([
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols)
        ])
        return preprocessor

    def initiate_data_transformation(self):
        try:
            logger.info("Reading train and test data")
            train_df = pd.read_csv(self.config.get_data_ingestion_config().train_data_path)
            test_df = pd.read_csv(self.config.get_data_ingestion_config().test_data_path)
            target_column = "SalePrice"

            X_train = train_df.drop(columns=[target_column], axis=1)
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column], axis=1)
            y_test = test_df[target_column]
            logger.info("Applying preprocessing pipeline")
            preprocessor = self.get_preprocessor(X_train)

            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)
                        
            train_arr = np.c_[X_train, y_train]
            test_arr = np.c_[X_test, y_test]
            save_object(    file_path=os.path.join("artifacts", "preprocessor.pkl"),    obj=preprocessor)
            logger.info("Preprocessor saved successfully")
            return X_train, X_test, y_train, y_test

        except Exception as e:
            logger.exception("Error in data transformation")
            raise CustomException(e)

