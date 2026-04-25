import os
import sys
import pandas as pd
import numpy as np
import joblib

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class HybridPipeline:
    def __init__(self):
        # Load models and preprocessors from BASE/models/
        self.yield_model = joblib.load(os.path.join(BASE, 'models', 'yield_best_model.pkl'))
        self.yield_prep = joblib.load(os.path.join(BASE, 'models', 'yield_preprocessor.pkl'))
        
        self.fert_model = joblib.load(os.path.join(BASE, 'models', 'fertilizer_best_model.pkl'))
        self.fert_prep = joblib.load(os.path.join(BASE, 'models', 'fertilizer_preprocessor.pkl'))
        
        # Determine fertilizer classes
        self.fert_classes = self.fert_prep.label_encoders['Recommended_Fertilizer'].classes_

    def _categorize_yield(self, value: float) -> str:
        if value < 87:
            return 'Low'
        elif value < 149:
            return 'Medium'
        else:
            return 'High'

    def _transform_yield(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        prep = self.yield_prep
        
        # Add defaults for missing numerics and categoricals to avoid errors
        for col in prep.num_cols:
            if col not in df.columns:
                df[col] = 0.0
        for col in prep.cat_cols:
            if col not in df.columns:
                df[col] = 'Unknown'
                
        for col in prep.num_cols:
            if col in prep.iqr_bounds and col in df.columns:
                lower, upper = prep.iqr_bounds[col]
                df[col] = np.clip(df[col], lower, upper)
                
        for col in prep.cat_cols:
            if col in prep.label_encoders and col in df.columns:
                le = prep.label_encoders[col]
                df[col] = df[col].map(lambda s, le=le: s if s in le.classes_ else le.classes_[0])
                df[col] = le.transform(df[col].astype(str))
                
        df[prep.num_cols] = prep.scalers.transform(df[prep.num_cols])
        
        train_cols = pd.read_csv(
            os.path.join(BASE, 'data', 'processed', 'yield_train.csv'),
            nrows=1
        ).drop(columns=['Yield_ton_per_ha'], errors='ignore').columns.tolist()
        
        # Ensure correct column order and existence
        for col in train_cols:
            if col not in df.columns:
                df[col] = 0
                
        return df[train_cols]

    def _transform_fertilizer(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        prep = self.fert_prep
        
        # Add defaults for missing numerics and categoricals
        for col in prep.num_cols:
            if col not in df.columns:
                df[col] = 0.0
        for col in prep.cat_cols:
            if col not in df.columns:
                df[col] = 'Unknown'

        for col in prep.num_cols:
            if col in prep.iqr_bounds and col in df.columns:
                lower, upper = prep.iqr_bounds[col]
                df[col] = np.clip(df[col], lower, upper)
                
        for col in prep.cat_cols:
            if col in prep.label_encoders and col in df.columns:
                le = prep.label_encoders[col]
                df[col] = df[col].map(lambda s, le=le: s if s in le.classes_ else le.classes_[0])
                df[col] = le.transform(df[col].astype(str))
                
        df[prep.num_cols] = prep.scalers.transform(df[prep.num_cols])
        
        feature_columns = [
            'Soil_Type', 'Soil_pH', 'Soil_Moisture', 'Organic_Carbon',
            'Electrical_Conductivity', 'Nitrogen_Level', 'Phosphorus_Level',
            'Potassium_Level', 'Temperature', 'Humidity', 'Rainfall',
            'Crop_Type', 'Crop_Growth_Stage', 'Season', 'Irrigation_Type',
            'Previous_Crop', 'Region', 'Fertilizer_Used_Last_Season',
            'Yield_Last_Season'
        ]
        
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
                
        return df[feature_columns]

    def predict_yield_only(self, input_dict: dict) -> dict:
        df = pd.DataFrame([input_dict])
        
        # Map frontend 'Irrigation_Type' to 'Irrigation' if missing
        if 'Irrigation' not in df.columns and 'Irrigation_Type' in df.columns:
            df['Irrigation'] = df['Irrigation_Type']
            
        # Map frontend field names to match model
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
            
        X = self._transform_yield(df)
        pred = self.yield_model.predict(X)[0]
        category = self._categorize_yield(pred)
        
        return {
            'predicted_value': float(pred),
            'category': category
        }

    def predict_fertilizer_only(self, input_dict: dict) -> dict:
        df = pd.DataFrame([input_dict])
        
        # Map yield attributes to fertilizer attributes if missing
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
            
        X = self._transform_fertilizer(df)
        pred_idx = self.fert_model.predict(X)[0]
        
        if hasattr(self.fert_model, 'predict_proba'):
            proba = self.fert_model.predict_proba(X)[0]
        else:
            proba = np.zeros(len(self.fert_classes))
            proba[pred_idx] = 1.0
            
        predicted_class = self.fert_classes[pred_idx]
        confidence = float(proba[pred_idx])
        
        top3_indices = np.argsort(proba)[::-1][:3]
        top3 = [
            {
                'name': str(self.fert_classes[idx]),
                'probability': float(proba[idx])
            }
            for idx in top3_indices
        ]
        
        return {
            'recommended': str(predicted_class),
            'confidence': confidence,
            'top3': top3
        }

    def predict(self, input_dict: dict) -> dict:
        yield_results = self.predict_yield_only(input_dict)
        fert_results = self.predict_fertilizer_only(input_dict)
        
        adv = f"Based on conditions, the expected yield is {yield_results['category'].lower()} ({yield_results['predicted_value']:.2f} tons/ha)."
        adv += f" We recommend applying {fert_results['recommended']} with {fert_results['confidence']*100:.1f}% confidence."
        
        return {
            'yield': yield_results,
            'fertilizer': fert_results,
            'combined_advice': adv
        }
