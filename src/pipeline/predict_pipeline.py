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
                "MS SubClass",
                "MS Zoning",
                "Lot Frontage",
                "Lot Area",
                "Street",
                "Alley",
                "Lot Shape",
                "Land Contour",
                "Utilities",
                "Lot Config",
                "Land Slope",
                "Neighborhood",
                "Condition 1",
                "Condition 2",
                "Bldg Type",
                "House Style",
                "Overall Qual",
                "Overall Cond",
                "Year Built",
                "Year Remod/Add",
                "Roof Style",
                "Roof Matl",
                "Exterior 1st",
                "Exterior 2nd",
                "Mas Vnr Type",
                "Mas Vnr Area",
                "Exter Qual",
                "Exter Cond",
                "Foundation",
                "Bsmt Qual",
                "Bsmt Cond",
                "Bsmt Exposure",
                "BsmtFin Type 1",
                "BsmtFin SF 1",
                "BsmtFin Type 2",
                "BsmtFin SF 2",
                "Bsmt Unf SF",
                "Total Bsmt SF",
                "Heating",
                "Heating QC",
                "Central Air",
                "Electrical",
                "1st Flr SF",
                "2nd Flr SF",
                "Low Qual Fin SF",
                "Gr Liv Area",
                "Bsmt Full Bath",
                "Bsmt Half Bath",
                "Full Bath",
                "Half Bath",
                "Bedroom AbvGr",
                "Kitchen AbvGr",
                "Kitchen Qual",
                "TotRms AbvGrd",
                "Functional",
                "Fireplaces",
                "Fireplace Qu",
                "Garage Type",
                "Garage Yr Blt",
                "Garage Finish",
                "Garage Cars",
                "Garage Area",
                "Garage Qual",
                "Garage Cond",
                "Paved Drive",
                "Wood Deck SF",
                "Open Porch SF",
                "Enclosed Porch",
                "3Ssn Porch",
                "Screen Porch",
                "Pool Area",
                "Pool QC",
                "Fence",
                "Misc Feature",
                "Misc Val",
                "Mo Sold",
                "Yr Sold",
                "Sale Type",
                "Sale Condition"
            ]

            missing = set(required_columns) - set(df.columns)

            if missing:
                raise ValueError(
                    f"Missing {len(missing)} required columns: {sorted(missing)}"
                )

            return df[required_columns]

        except Exception as e:
            raise CustomException(e)