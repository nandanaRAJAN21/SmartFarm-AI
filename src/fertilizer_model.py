import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, StratifiedKFold
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FertilizerModel:
    def __init__(self):
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
            'LightGBM': LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
            'CatBoost': CatBoostClassifier(iterations=100, random_state=42, verbose=0)
        }
        self.best_model_name = None
        self.best_model = None
        self.results = {}
        
    def evaluate_model(self, y_true, y_pred, y_proba):
        acc = accuracy_score(y_true, y_pred) * 100
        # average='weighted' for multiclass
        prec = precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100
        rec = recall_score(y_true, y_pred, average='weighted') * 100
        f1 = f1_score(y_true, y_pred, average='weighted') * 100
        
        try:
            roc_auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
        except:
            roc_auc = 0.0
            
        return acc, prec, rec, f1, roc_auc

    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        logger.info("Starting fertilizer model training...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
            else:
                y_proba = np.zeros((len(y_test), len(np.unique(y_train))))
                
            acc, prec, rec, f1, roc_auc = self.evaluate_model(y_test, y_pred, y_proba)
            
            # 5-fold CV Accuracy
            cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy', n_jobs=-1)
            cv_mean = cv_scores.mean() * 100
            cv_std = cv_scores.std() * 100
            
            self.results[name] = {
                'Accuracy': acc,
                'Precision': prec,
                'Recall': rec,
                'F1': f1,
                'ROC_AUC': roc_auc,
                'CV_Acc_mean': cv_mean,
                'CV_Acc_std': cv_std,
                'model': model
            }
            logger.info(f"{name} Acc: {acc:.2f}%, CV Acc: {cv_mean:.2f} ± {cv_std:.2f}%")
            
        return self.results
            
    def select_best_model(self):
        best_f1 = -1
        best_acc = -1
        
        for name, res in self.results.items():
            if res['F1'] > best_f1:
                best_f1 = res['F1']
                best_acc = res['Accuracy']
                self.best_model_name = name
            elif np.isclose(res['F1'], best_f1, atol=1e-4) and res['Accuracy'] > best_acc:
                best_f1 = res['F1']
                best_acc = res['Accuracy']
                self.best_model_name = name
                
        self.best_model = self.results[self.best_model_name]['model']
        return self.best_model_name, best_f1, best_acc
        
    def tune_best_model(self, X_train, y_train):
        logger.info(f"Tuning {self.best_model_name}...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
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
                'max_depth': [3, 5, 7],
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
            logger.info("No tuning for Logistic Regression.")
            return None
            
        search = RandomizedSearchCV(self.models[self.best_model_name], param_distributions=param_grid, 
                                    n_iter=20, scoring='f1_weighted', cv=skf, random_state=42, n_jobs=-1)
        search.fit(X_train, y_train)
        
        self.best_model = search.best_estimator_
        logger.info(f"Best params: {search.best_params_}")
        return search.best_params_

    def save_model(self, path):
        joblib.dump(self.best_model, path)
        logger.info(f"Saved best model to {path}")
