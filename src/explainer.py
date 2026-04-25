import shap
import lime
import lime.lime_tabular
import numpy as np
import pandas as pd
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class Explainer:
    def __init__(self, hybrid_pipeline):
        self.pipeline = hybrid_pipeline
        self.yield_model = hybrid_pipeline.yield_model
        self.fert_model = hybrid_pipeline.fert_model
        
        # Load sample training data for LIME
        self.yield_train = pd.read_csv(os.path.join(BASE, 'data', 'processed', 'yield_train.csv'), nrows=500)
        self.fert_train = pd.read_csv(os.path.join(BASE, 'data', 'processed', 'fertilizer_train.csv'), nrows=500)
        
        self.yield_X_train = self.yield_train.drop(columns=['Yield_ton_per_ha'], errors='ignore')
        self.fert_X_train = self.fert_train.drop(columns=['Recommended_Fertilizer'], errors='ignore')

        # Initialize LIME explainers
        self.lime_yield = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.yield_X_train.values,
            feature_names=self.yield_X_train.columns.tolist(),
            mode='regression',
            random_state=42
        )
        
        self.lime_fert = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.fert_X_train.values,
            feature_names=self.fert_X_train.columns.tolist(),
            mode='classification',
            class_names=[str(c) for c in self.pipeline.fert_classes],
            random_state=42
        )

    def _get_shap_explainer(self, model, background_data):
        # We assume Random Forest/XGBoost/LightGBM/CatBoost are TreeExplainer compatible
        # If it's a linear model like Logistic regression or Linear regression we use LinearExplainer
        model_name = type(model).__name__.lower()
        if 'linear' in model_name or 'logistic' in model_name:
            if hasattr(model, 'coef_'):
                # Some sklearn linear models can be explained by LinearExplainer, 
                # but if that fails we fall back to a simpler explainer
                try:
                    return shap.LinearExplainer(model, background_data)
                except:
                    pass
            # General fallback
            return shap.Explainer(model.predict, background_data)
        else:
            try:
                return shap.TreeExplainer(model)
            except Exception:
                return shap.Explainer(model.predict, background_data)

    def explain_yield(self, input_dict: dict, method: str = 'shap') -> dict:
        df = pd.DataFrame([input_dict])
        
        # apply same mapping as predict_yield_only
        if 'Irrigation' not in df.columns and 'Irrigation_Type' in df.columns:
            df['Irrigation'] = df['Irrigation_Type']
        if 'Crop' not in df.columns and 'Crop_Type' in df.columns:
            df['Crop'] = df['Crop_Type']
        if 'Rainfall_mm' not in df.columns and 'Rainfall' in df.columns:
            df['Rainfall_mm'] = df['Rainfall']
        if 'Temperature_C' not in df.columns and 'Temperature' in df.columns:
            df['Temperature_C'] = df['Temperature']
        if 'Humidity_pct' not in df.columns and 'Humidity' in df.columns:
            df['Humidity_pct'] = df['Humidity']
        if 'Fertilizer_Used_kg' not in df.columns and 'Fertilizer_Used_Last_Season' in df.columns:
            df['Fertilizer_Used_kg'] = df['Fertilizer_Used_Last_Season']
            
        X_trans = self.pipeline._transform_yield(df)
        feature_names = X_trans.columns.tolist()
        
        importances = []
        
        if method == 'shap':
            explainer = self._get_shap_explainer(self.yield_model, self.yield_X_train)
            shap_values = explainer.shap_values(X_trans)
            
            # Handle list output for some tree explainers based on version
            if isinstance(shap_values, list):
                sv = shap_values[0]
            else:
                sv = shap_values
                
            sv_row = sv[0]
            for i, name in enumerate(feature_names):
                val = sv_row[i]
                importances.append({
                    'feature': name,
                    'importance': float(abs(val)),
                    'direction': 'positive' if val > 0 else 'negative'
                })
        else:
            # LIME
            pred_func = self.yield_model.predict
            exp = self.lime_yield.explain_instance(
                X_trans.values[0],
                pred_func,
                num_features=8
            )
            for feature_desc, weight in exp.as_list():
                # LIME returns conditions like "Feature X > 0.5". Extract the feature name roughly
                found_feat = "Unknown"
                for fn in feature_names:
                    if fn in feature_desc:
                        found_feat = fn
                        break
                
                importances.append({
                    'feature': found_feat,
                    'importance': float(abs(weight)),
                    'direction': 'positive' if weight > 0 else 'negative'
                })
                
        # Sort and take top 8
        importances = sorted(importances, key=lambda x: x['importance'], reverse=True)[:8]
        
        top_positive = [i['feature'] for i in importances if i['direction'] == 'positive']
        top_negative = [i['feature'] for i in importances if i['direction'] == 'negative']
        
        summary = "The prediction was driven "
        if top_positive:
            summary += f"positively by {', '.join(top_positive[:2])}. "
        if top_negative:
            summary += f"However, it was negatively impacted by {', '.join(top_negative[:2])}."
            
        return {
            'method': method,
            'feature_importances': importances,
            'summary': summary.strip()
        }

    def explain_fertilizer(self, input_dict: dict, method: str = 'shap') -> dict:
        df = pd.DataFrame([input_dict])
        
        # Mappings
        if 'Irrigation_Type' not in df.columns and 'Irrigation' in df.columns:
            df['Irrigation_Type'] = df['Irrigation']
        if 'Crop_Type' not in df.columns and 'Crop' in df.columns:
            df['Crop_Type'] = df['Crop']
        if 'Rainfall' not in df.columns and 'Rainfall_mm' in df.columns:
            df['Rainfall'] = df['Rainfall_mm']
        if 'Temperature' not in df.columns and 'Temperature_C' in df.columns:
            df['Temperature'] = df['Temperature_C']
        if 'Humidity' not in df.columns and 'Humidity_pct' in df.columns:
            df['Humidity'] = df['Humidity_pct']
        if 'Fertilizer_Used_Last_Season' not in df.columns and 'Fertilizer_Used_kg' in df.columns:
            df['Fertilizer_Used_Last_Season'] = df['Fertilizer_Used_kg']
            
        X_trans = self.pipeline._transform_fertilizer(df)
        feature_names = X_trans.columns.tolist()
        importances = []
        
        # Predict class index to explain specifically that class
        pred_idx = self.fert_model.predict(X_trans)[0]
        
        if method == 'shap':
            explainer = self._get_shap_explainer(self.fert_model, self.fert_X_train)
            shap_values = explainer.shap_values(X_trans)
            
            if isinstance(shap_values, list):
                # multi-class: choose the array for the predicted class
                sv = shap_values[pred_idx]
                sv_row = sv[0]
            elif len(shap_values.shape) == 3:
                # Some versions return 3D array: (samples, features, classes)
                sv_row = shap_values[0, :, pred_idx]
            else:
                # Binary or flattened
                sv_row = shap_values[0]
                
            for i, name in enumerate(feature_names):
                val = sv_row[i]
                importances.append({
                    'feature': name,
                    'importance': float(abs(val)),
                    'direction': 'positive' if val > 0 else 'negative'
                })
        else:
            # LIME
            pred_func = self.fert_model.predict_proba
            exp = self.lime_fert.explain_instance(
                X_trans.values[0],
                pred_func,
                num_features=8,
                labels=(pred_idx,)
            )
            for feature_desc, weight in exp.as_list(label=pred_idx):
                found_feat = "Unknown"
                for fn in feature_names:
                    if fn in feature_desc:
                        found_feat = fn
                        break
                
                importances.append({
                    'feature': found_feat,
                    'importance': float(abs(weight)),
                    'direction': 'positive' if weight > 0 else 'negative'
                })

        importances = sorted(importances, key=lambda x: x['importance'], reverse=True)[:8]
        
        top_positive = [i['feature'] for i in importances if i['direction'] == 'positive']
        top_negative = [i['feature'] for i in importances if i['direction'] == 'negative']
        
        summary = f"The recommendation for {self.pipeline.fert_classes[pred_idx]} was supported "
        if top_positive:
            summary += f"by the values of {', '.join(top_positive[:2])}. "
        if top_negative:
            summary += f"Evidence against this choice came from {', '.join(top_negative[:2])}."
            
        return {
            'method': method,
            'feature_importances': importances,
            'summary': summary.strip()
        }
