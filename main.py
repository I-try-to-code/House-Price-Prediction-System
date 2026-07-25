from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.config.configuration import Configuration

from src.logger import logger
from src.exception import CustomException

def main():

    try:
        logger.info("=" * 50)
        logger.info("House Price Prediction Pipeline Started")
        logger.info("=" * 50)

        # Config setup
        config = Configuration()

        # Data Ingestion
        ingestion = DataIngestion(config=config.get_data_ingestion_config())
        ingestion.initiate_data_ingestion()

        # Data Validation
        validation = DataValidation()
        validation.validate()

        # Data Transformation
        transformation = DataTransformation()
        X_train, X_test, y_train, y_test = transformation.initiate_data_transformation()

        # Model Training
        trainer = ModelTrainer()

        report, trained_models = trainer.initiate_model_training(
            X_train, y_train, X_test, y_test
        )

        # Get the best model
        best_model_name = max(report, key=lambda x: report[x]["R2"])
        best_score = report[best_model_name]["R2"]

        logger.info(f"Best Model : {best_model_name}")
        logger.info(f"Best R2 Score : {best_score}")

        logger.info("=" * 50)
        logger.info("Pipeline Completed Successfully")
        logger.info("=" * 50)

    except Exception as e:
        logger.exception(e)
        raise CustomException(e)

if __name__ == "__main__":
    main()