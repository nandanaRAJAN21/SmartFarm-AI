# %% [markdown]
# # Fertilizer Data Preprocessing
# 
# This notebook handles outliers, encoding, scaling, SMOTE (if needed) and generates train/test sets.

# %%
import sys
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath('src'))
from data_loader import DataLoader
from fertilizer_preprocessor import FertilizerPreprocessor

out_dir_models = r"C:\Users\NANDANA\OneDrive\Desktop\smartfarm\models"
out_dir_data = r"C:\Users\NANDANA\OneDrive\Desktop\smartfarm\crop yield prediction\data\processed"
os.makedirs(out_dir_models, exist_ok=True)
os.makedirs(out_dir_data, exist_ok=True)

loader = DataLoader()
df = loader.load_fertilizer()

# %% [markdown]
# ## 1. Preprocessing and SMOTE
# %%
preprocessor = FertilizerPreprocessor()
X, y = preprocessor.fit_transform(df)

print("Nulls after preprocessing:", X.isnull().sum().sum())

# %% [markdown]
# ## 2. Feature Encoders Information
# %%
for col, encoder in preprocessor.label_encoders.items():
    if col != preprocessor.target_col:
        print(f"{col} classes: {encoder.classes_}")

print("\nTarget Classes:", preprocessor.label_encoders[preprocessor.target_col].classes_)

# %% [markdown]
# ## 3. Train / Test Split
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print("\nShapes:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# %% [markdown]
# ## 4. Save Models and Data
# %%
preprocessor_path = os.path.join(out_dir_models, 'fertilizer_preprocessor.pkl')
with open(preprocessor_path, 'wb') as f:
    joblib.dump(preprocessor, f)
print(f"Preprocessor saved to {preprocessor_path}")

loader.save_processed(train_df, "fertilizer_train.csv")
loader.save_processed(test_df, "fertilizer_test.csv")
print("Processing complete and data saved.")
