# %% [markdown]
# # Yield Data Preprocessing
# 
# This notebook handles missing values, outliers, encoding, scaling, and generates train/test sets.

# %%
import sys
import os
import io

sys.path.append(os.path.abspath('src'))

from data_loader import DataLoader
from yield_preprocessor import YieldPreprocessor
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

out_dir_models = r"C:\Users\NANDANA\OneDrive\Desktop\smartfarm\crop yield prediction\models"
out_dir_data = r"C:\Users\NANDANA\OneDrive\Desktop\smartfarm\crop yield prediction\data\processed"

# Initialize loader
loader = DataLoader()
df = loader.load_yield()

# %% [markdown]
# ## 1. Missing Values and Preprocessing
# We use the custom YieldPreprocessor class to perform all preprocessing steps at once.
# %%
preprocessor = YieldPreprocessor()

# Check Nulls before
print("Nulls before:")
print(df[['Irrigation', 'Previous_Crop']].isnull().sum())

import numpy as np
# Fit and transform
X, y = preprocessor.fit_transform(df)

# Check Nulls after
print("\nNulls after imputation in numericals/categoricals:", X.isnull().sum().sum())

# %% [markdown]
# ## 2. Feature Encoders Information
# Let's verify the classes learned by label encoders.
# %%
for col, encoder in preprocessor.label_encoders.items():
    print(f"{col} classes: {encoder.classes_}")

# %% [markdown]
# ## 3. Train / Test Split
# We split 80% train and 20% test data with a fixed random state.
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nShapes:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

# Combine for saving
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# %% [markdown]
# ## 4. Saving Processed Data and Preprocessor
# Models are saved for the predictive pipeline to use.
# %%
os.makedirs(out_dir_models, exist_ok=True)
os.makedirs(out_dir_data, exist_ok=True)

# Save Pickles
preprocessor_path = os.path.join(out_dir_models, 'yield_preprocessor.pkl')
with open(preprocessor_path, 'wb') as f:
    joblib.dump(preprocessor, f)

print(f"Preprocessor saved to {preprocessor_path}")

# Save CSVs
loader.save_processed(train_df, "yield_train.csv")
loader.save_processed(test_df, "yield_test.csv")

print("Processing complete and data saved.")
