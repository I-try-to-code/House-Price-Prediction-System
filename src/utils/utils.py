import os
import pickle
from src.logger import logger
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e)


import pickle

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e)

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        logger.info("Evaluating models...")
        

        for name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            
            r2 = r2_score(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            cv_scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=5,
                scoring="r2"
            )
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            report[name] = {
                "R2": r2,
                "MAE": mae,
                "RMSE": rmse,
                "CV Mean": cv_mean,
                "CV Std": cv_std
            }
       
                
        return report
    except Exception as e:
        raise CustomException(e)