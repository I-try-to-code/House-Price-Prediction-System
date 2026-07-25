import os
import numpy as np

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)
from src.utils import evaluate_models
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from src.logger import logger
from src.exception import CustomException
from src.utils import save_object

class ModelTrainer:

    def __init__(self):
        pass

    def initiate_model_training(self, X_train, y_train, X_test, y_test):
        
		models = {
			"Linear Regression": LinearRegression(),
			"Ridge": Ridge(),
			"Lasso": Lasso(),
			"ElasticNet": ElasticNet(),
			"Decision Tree": DecisionTreeRegressor(random_state=42),
			"Random Forest": RandomForestRegressor(random_state=42),
			"XGBoost": XGBRegressor(random_state=42)
		}
		trained_models = {}

		for name, model in models.items():

			model.fit(X_train, y_train)
			logger.info(f"Training {name}")


			predictions = model.predict(X_test)

			logger.info(f"{name} R2 Score: {score:.4f}")


		report = evaluate_models(
			X_train,
			y_train,
			X_test,
			y_test,
			models
		)
		best_model = max(
			report,
			key=lambda x: report[x]["R2"]
		)
		return report, trained_models
