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
            "Lasso": Lasso(max_iter=100000,selection="cyclic"),
            "ElasticNet": ElasticNet(max_iter=100000,selection="cyclic"),
            "Decision Tree": DecisionTreeRegressor(random_state=42,criterion='poisson'),
            "Random Forest": RandomForestRegressor(bootstrap=False,n_estimators=153,random_state=42),
            "XGBoost": XGBRegressor(random_state=42)
        }
        
        report = evaluate_models(
            X_train,
            y_train,
            X_test,
            y_test,
            models
        )
        logger.info(" ")        
        logger.info(" ")
        logger.info(" ")

        for name, scores in report.items():
            logger.info(f"{name} R2 Score: {scores['R2']:.4f}")
            
        best_model = max(
            report,
            key=lambda x: report[x]["R2"]
        )
        return report, models
