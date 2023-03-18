import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException 
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test data")

            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Linear Regression": LinearRegression(),
                "Gradient Boossting": GradientBoostingRegressor(),
                "KNN": KNeighborsRegressor(),
                "XGB": XGBRegressor(),
                "Cat Boosting": CatBoostRegressor(),
                "AdaBoost": AdaBoostRegressor()
            }

            model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            best_ms = max(sorted(model_report.values()))

            best_name = list(model_report.keys())[
                list(model_report.values()).index(best_ms)
            ]
            best_model = models[best_name]

            if best_ms < 0.6:
                raise CustomException("NO best Model")
            logging.info("Best model found on train and test data")


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model,
            )

            predicted = best_model.predict(X_test)

            r2scr = r2_score(y_test,predicted)
            return r2scr


        except Exception as e:
            CustomException(e,sys)