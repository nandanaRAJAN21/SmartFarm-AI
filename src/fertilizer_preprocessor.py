import pandas as pd
import numpy as np
import joblib
import logging
from typing import Tuple, Dict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FertilizerPreprocessor:
    """Preprocessor class for fertilizer recommendation dataset."""
    
    def __init__(self):
        self.num_cols = ['Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 'Electrical_Conductivity', 
                         'Nitrogen_Level', 'Phosphorus_Level', 'Potassium_Level', 'Temperature', 
                         'Humidity', 'Rainfall', 'Fertilizer_Used_Last_Season', 'Yield_Last_Season']
        self.cat_cols = ['Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season', 
                         'Irrigation_Type', 'Previous_Crop', 'Region']
        self.target_col = 'Recommended_Fertilizer'
        
        self.scalers = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.iqr_bounds = {}
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        logger.info("Starting fertilizer preprocessing fit_transform...")
        df = df.copy()
        
        # 1. Outlier Treatment (IQR capping)
        for col in self.num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            self.iqr_bounds[col] = (lower, upper)
            
            capped_idx = (df[col] < lower) | (df[col] > upper)
            logger.info(f"Capping {capped_idx.sum()} outliers in {col}")
            df[col] = np.clip(df[col], lower, upper)
            
        # 2. Label Encode Features
        for col in self.cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
            
        # 3. Target Label Encode
        target_le = LabelEncoder()
        df[self.target_col] = target_le.fit_transform(df[self.target_col])
        self.label_encoders[self.target_col] = target_le
        
        classes = target_le.classes_
        logger.info(f"Target classes encoding map: {dict(enumerate(classes))}")
            
        # 4. Feature Scaling (num_cols)
        df[self.num_cols] = self.scalers.fit_transform(df[self.num_cols])
        
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        
        # 5. Check Class Imbalance and apply SMOTE
        class_dist = y.value_counts(normalize=True)
        min_class_ratio = class_dist.min() * 100
        logger.info(f"Minimum class proportion: {min_class_ratio:.2f}%")
        
        if min_class_ratio < 15.0:
            logger.info("Applying SMOTE due to class imbalance (< 15%)...")
            logger.info(f"Shape before SMOTE: {X.shape}")
            smote = SMOTE(random_state=42)
            X, y = smote.fit_resample(X, y)
            logger.info(f"Shape after SMOTE: {X.shape}")
        
        return X, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data without target."""
        df = df.copy()
        
        # Outliers cap
        for col in self.num_cols:
            if col in self.iqr_bounds:
                lower, upper = self.iqr_bounds[col]
                df[col] = np.clip(df[col], lower, upper)
                
        # Label Encode Features
        for col in self.cat_cols:
            if col in self.label_encoders:
                le = self.label_encoders[col]
                df[col] = df[col].map(lambda s: s if s in le.classes_ else le.classes_[0])
                df[col] = le.transform(df[col])
                
        # Scale
        df[self.num_cols] = self.scalers.transform(df[self.num_cols])
        
        return df
