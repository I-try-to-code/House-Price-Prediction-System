import os
import pandas as pd

from src.utils import load_object
from src.exception import CustomException


class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, features):
        try:
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)

            transformed_features = preprocessor.transform(features)
            prediction = model.predict(transformed_features)

            return prediction[0]

        except Exception as e:
            raise CustomException(e)


class CustomData:
    def __init__(self, feature_dict: dict):
        """
        feature_dict should contain all input features with the
        exact column names used during training.

        Example:
        {
            "MS SubClass": 20,
            "MS Zoning": "RL",
            "Lot Area": 8450,
            ...
        }
        """
        self.feature_dict = feature_dict

    def get_data_as_dataframe(self):
        try:
            df = pd.DataFrame([self.feature_dict])

            required_columns = [
                "Overall Qual", "Garage Cars", "Kitchen Qual", "Bsmt Qual",
                "MS Zoning", "Neighborhood", "Exter Qual", "Full Bath",
                "Garage Finish", "Gr Liv Area", "1st Flr SF", "Total Bsmt SF",
                "Land Contour", "Fireplaces", "Central Air", "Bsmt Exposure",
                "Sale Condition", "Exterior 1st", "Exterior 2nd", "2nd Flr SF",
                "Paved Drive", "Kitchen AbvGr", "Sale Type", "Foundation",
                "BsmtFin SF 1", "Roof Matl", "Roof Style", "Year Built",
                "Bsmt Full Bath", "Garage Type"
            ]
            
            missing = set(required_columns) - set(df.columns)

            if missing:
                raise ValueError(
                    f"Missing {len(missing)} required columns: {sorted(missing)}"
                )

            return df[required_columns]

        except Exception as e:
            raise CustomException(e)