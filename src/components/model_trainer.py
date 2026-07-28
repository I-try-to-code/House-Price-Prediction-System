import os
import numpy as np

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)
import pandas as pd
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("sqlite:///mlflow.db")
experiment_name = "House_Price_Prediction"
try:
    mlflow.create_experiment(
        experiment_name,
        artifact_location=f"file:///{os.path.abspath('mlruns')}"
    )
except Exception:
    pass
mlflow.set_experiment(experiment_name)

from src.utils import evaluate_models
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from src.logger import logger
from src.exception import CustomException
from src.utils import save_object, load_object

class ModelTrainer:

    def __init__(self):
        pass

    def initiate_model_training(self, X_train, y_train, X_test, y_test):
        
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(max_iter=100000,selection="cyclic"),
            "ElasticNet": ElasticNet(max_iter=100000,selection="cyclic"),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(random_state=42),
            "XGBoost": XGBRegressor(random_state=42,booster="gbtree",learning_rate=0.3)
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
            logger.info(f"{name} R2 Score: {scores['R2']:.4f} | MAE: {scores['MAE']:.4f} | RMSE: {scores['RMSE']:.4f} | CV Mean: {scores['CV Mean']:.4f} | CV Std: {scores['CV Std']:.4f}")
            
        best_model_name = max(
            report,
            key=lambda x: report[x]["R2"]
        )
        best_model = models[best_model_name]
        baseline_r2 = report[best_model_name]["R2"]
        model_path = os.path.join("artifacts", "model.pkl")

        if best_model_name in ["Random Forest", "XGBoost"]:
            logger.info(f"Starting Hyperparameter Tuning for {best_model_name}...")
            
            if best_model_name == "Random Forest":
                param_grid = {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [10, 20, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                }
                search = GridSearchCV(
                    estimator=RandomForestRegressor(random_state=42),
                    param_grid=param_grid,
                    scoring="r2",
                    cv=5,
                    n_jobs=-1,
                    verbose=1
                )
                search.fit(X_train, y_train)

            elif best_model_name == "XGBoost":
                param_dist = {
                    "n_estimators": [100, 200, 300, 500],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
                    "max_depth": [3, 5, 7, 10],
                    "subsample": [0.6, 0.8, 1.0],
                    "colsample_bytree": [0.6, 0.8, 1.0],
                    "min_child_weight": [1, 3, 5],
                    "gamma": [0, 0.1, 0.3]
                }
                search = RandomizedSearchCV(
                    estimator=XGBRegressor(random_state=42, objective="reg:squarederror"),
                    param_distributions=param_dist,
                    n_iter=25,
                    scoring="r2",
                    cv=5,
                    random_state=42,
                    n_jobs=-1,
                    verbose=1
                )
                search.fit(X_train, y_train)

            best_model = search.best_estimator_
            best_params = search.best_params_
            best_cv_score = search.best_score_
            y_pred = best_model.predict(X_test)
            tuned_r2 = r2_score(y_test, y_pred)
            
            logger.info(f"--- TUNING RESULTS FOR {best_model_name} ---")
            logger.info(f"Best Hyperparameters: {best_params}")
            logger.info(f"Best Tuned CV R2 Score: {best_cv_score:.4f}")
            logger.info(f"Best Tuned Test R2 Score: {tuned_r2:.4f}")
            logger.info("------------------------------------------")
            
            if tuned_r2 > baseline_r2:
                logger.info("Tuned model performed better. Saving tuned model.")
                final_model = best_model
                final_r2 = tuned_r2
                final_mae = mean_absolute_error(y_test, y_pred)
                final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                
                # Update report so main.py prints the tuned score
                report[best_model_name]["R2"] = final_r2
                report[best_model_name]["MAE"] = final_mae
                report[best_model_name]["RMSE"] = final_rmse
            else:
                logger.info("Baseline model performed better. Saving baseline model.")
                final_model = models[best_model_name]
                final_r2 = baseline_r2
                final_mae = report[best_model_name]["MAE"]
                final_rmse = report[best_model_name]["RMSE"]
                best_model = final_model  # For feature importance extraction

            # Save the final best model (tuned or base)
            save_object(file_path=model_path, obj=final_model)
            logger.info(f"Saved {best_model_name} to {model_path}")
            
            with mlflow.start_run(run_name=f"{best_model_name}_Tuned"):
                mlflow.log_param("model", best_model_name)
                mlflow.log_params(best_params)
                mlflow.log_metric("R2", final_r2)
                mlflow.log_metric("MAE", final_mae)
                mlflow.log_metric("RMSE", final_rmse)
                if best_model_name == "XGBoost":
                    mlflow.sklearn.log_model(sk_model=final_model, artifact_path="model", skops_trusted_types=["xgboost.core.Booster", "xgboost.sklearn.XGBRegressor"])
                else:
                    mlflow.sklearn.log_model(sk_model=final_model, artifact_path="model")
            
            logger.info("Extracting feature importances...")
            preprocessor = load_object(os.path.join("artifacts", "preprocessor.pkl"))
            feature_names = preprocessor.get_feature_names_out()

            importance_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": best_model.feature_importances_
            })
            importance_df = importance_df.sort_values(by="Importance", ascending=False)
            importance_df["Original Feature"] = importance_df["Feature"]

            importance_df.loc[
                importance_df["Original Feature"].str.startswith("num__"), "Original Feature"
            ] = importance_df.loc[
                importance_df["Original Feature"].str.startswith("num__"), "Original Feature"
            ].str.replace("num__", "", regex=False)

            importance_df.loc[
                importance_df["Original Feature"].str.startswith("cat__"), "Original Feature"
            ] = (
                importance_df.loc[
                    importance_df["Original Feature"].str.startswith("cat__"), "Original Feature"
                ]
                .str.replace("cat__", "", regex=False)
                .str.rsplit("_", n=1)
                .str[0]
            )

            grouped = (
                importance_df.groupby("Original Feature")["Importance"]
                .sum()
                .sort_values(ascending=False)
            )

            
            return report, models
