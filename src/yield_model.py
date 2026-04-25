import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, KFold
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YieldModel:
    def __init__(self):
        self.models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
            'LightGBM': LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
            'CatBoost': CatBoostRegressor(iterations=100, random_state=42, verbose=0)
        }
        self.best_model_name = None
        self.best_model = None
        self.results = {}
        
    def evaluate_model(self, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # MAPE
        # Add epsilon to avoid divide by zero if y_true can be zero.
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        return mae, rmse, r2, mape

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        logger.info("Starting model training and evaluation...")
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mae, rmse, r2, mape = self.evaluate_model(y_test, y_pred)
            
            # 5-fold CV R2
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2', n_jobs=-1)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            self.results[name] = {
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2,
                'MAPE': mape,
                'CV_R2_mean': cv_mean,
                'CV_R2_std': cv_std,
                'model': model
            }
            logger.info(f"{name} R2: {r2:.4f}, CV R2: {cv_mean:.4f} ± {cv_std:.4f}")
            
        return self.results
            
    def select_best_model(self):
        best_r2 = -float('inf')
        best_rmse = float('inf')
        
        for name, res in self.results.items():
            # primary metric R2, secondary RMSE
            if res['R2'] > best_r2:
                best_r2 = res['R2']
                best_rmse = res['RMSE']
                self.best_model_name = name
            elif np.isclose(res['R2'], best_r2, atol=1e-4) and res['RMSE'] < best_rmse:
                best_r2 = res['R2']
                best_rmse = res['RMSE']
                self.best_model_name = name
                
        self.best_model = self.results[self.best_model_name]['model']
        return self.best_model_name, best_r2, best_rmse
        
    def tune_best_model(self, X_train, y_train):
        logger.info(f"Tuning {self.best_model_name}...")
        
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        if self.best_model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
        elif self.best_model_name == 'XGBoost':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            }
        elif self.best_model_name == 'LightGBM':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7, -1],
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [31, 63, 127],
                'subsample': [0.6, 0.8, 1.0]
            }
        elif self.best_model_name == 'CatBoost':
            param_grid = {
                'iterations': [100, 200, 300],
                'depth': [4, 6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'l2_leaf_reg': [1, 3, 5, 7]
            }
        else:
            logger.info("No tuning for Linear Regression.")
            return None
            
        search = RandomizedSearchCV(self.models[self.best_model_name], param_distributions=param_grid, 
                                    n_iter=20, scoring='r2', cv=kf, random_state=42, n_jobs=-1)
        search.fit(X_train, y_train)
        
        self.best_model = search.best_estimator_
        logger.info(f"Best params: {search.best_params_}")
        return search.best_params_

    def save_model(self, path):
        joblib.dump(self.best_model, path)
        logger.info(f"Saved best model to {path}")
