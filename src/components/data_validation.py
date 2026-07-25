from src.config.configuration import Configuration
import pandas as pd
from src.logger import logger
from src.exception import CustomException


class DataValidation:

    def __init__(self):
        self.config = Configuration()
        self.schema = self.config.get_schema()

    def validate(self):
        try:
            logger.info("Starting data validation")

            df = pd.read_csv(
                self.config.get_data_ingestion_config().raw_data_path
            )

            dtype_map = {
                "int64": "int",
                "float64": "float",
                "object": "object",
                "str": "object",
                "string": "object",
                "O": "object",
            }

            for column, expected_dtype in self.schema.items():

                if column not in df.columns:
                    raise CustomException(f"Column '{column}' is missing")

                actual_dtype = dtype_map.get(
                    str(df[column].dtype),
                    str(df[column].dtype)
                )

                if actual_dtype != expected_dtype:
                    raise CustomException(
                        f"Column '{column}' has incorrect dtype. "
                        f"Expected: {expected_dtype}, Got: {actual_dtype}"
                    )

            logger.info("Data validation completed successfully")
            return True

        except Exception as e:
            logger.exception("Error during data validation")
            raise CustomException(e)