import pandas as pd
import numpy as np
import joblib
import logging
from typing import Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import sys
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YieldPreprocessor:
    """Preprocessor class for crop yield dataset."""
    
    def __init__(self):
        self.num_cols = ['Soil_pH', 'Rainfall_mm', 'Temperature_C', 'Humidity_pct', 
                         'Fertilizer_Used_kg', 'Pesticides_Used_kg', 'Planting_Density']
        self.cat_cols = ['Crop', 'Region', 'Soil_Type', 'Irrigation', 'Previous_Crop']
        self.target_col = 'Yield_ton_per_ha'
        
        self.imputers = {}
        self.scalers = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.iqr_bounds = {}
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        logger.info("Starting preprocessing fit_transform...")
        
        # 1. Handle Missing Values
        mode_irr = df['Irrigation'].mode()[0]
        mode_prev = df['Previous_Crop'].mode()[0]
        self.imputers['Irrigation'] = mode_irr
        self.imputers['Previous_Crop'] = mode_prev
        
        df = df.copy()
        df['Irrigation'] = df['Irrigation'].fillna(mode_irr)
        df['Previous_Crop'] = df['Previous_Crop'].fillna(mode_prev)
        logger.info(f"Imputed missing values. Remaining nulls: {df.isnull().sum().sum()}")
        
        # 2. Outlier Treatment (IQR capping)
        for col in self.num_cols + [self.target_col]:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            self.iqr_bounds[col] = (lower, upper)
            # Count capped
            capped_idx = (df[col] < lower) | (df[col] > upper)
            logger.info(f"Capping {capped_idx.sum()} outliers in {col}")
            
            df[col] = np.clip(df[col], lower, upper)
            
        # 3. Label Encode
        for col in self.cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
            logger.info(f"Encoded {col} with {len(le.classes_)} classes.")
            
        # 4. Feature Scaling (only on features, not target)
        df[self.num_cols] = self.scalers.fit_transform(df[self.num_cols])
        logger.info("Scaled numerical features.")
        
        # 5. Split
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        
        return X, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data without target."""
        df = df.copy()
        
        # Impute
        df['Irrigation'] = df['Irrigation'].fillna(self.imputers.get('Irrigation', 'Drip'))
        df['Previous_Crop'] = df['Previous_Crop'].fillna(self.imputers.get('Previous_Crop', 'Wheat'))
        
        # Outliers cap (except target)
        for col in self.num_cols:
            if col in self.iqr_bounds:
                lower, upper = self.iqr_bounds[col]
                df[col] = np.clip(df[col], lower, upper)
                
        # Label Encode
        for col in self.cat_cols:
            if col in self.label_encoders:
                # Handle unseen labels by assigning a default or taking modulo/error
                # For simplicity, we assume robust input in API or handle silently
                le = self.label_encoders[col]
                df[col] = df[col].map(lambda s: s if s in le.classes_ else le.classes_[0])
                df[col] = le.transform(df[col])
                
        # Scale
        df[self.num_cols] = self.scalers.transform(df[self.num_cols])
        
        return df
