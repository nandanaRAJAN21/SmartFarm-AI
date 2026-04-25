import sys
import os

# Fix path — must be before any src imports
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'src'))

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (mean_squared_error,
                              mean_absolute_error,
                              r2_score)

# ── LOAD MODELS ────────────────────────────────────────────
try:
    model = joblib.load(os.path.join(BASE, 'models',
                        'yield_best_model.pkl'))
    prep  = joblib.load(os.path.join(BASE, 'models',
                        'yield_preprocessor.pkl'))
    test  = pd.read_csv(os.path.join(BASE, 'data',
                        'processed', 'yield_test.csv'))
    print("✅ Models and test data loaded successfully")
except Exception as e:
    print(f"❌ Error loading: {e}")
    sys.exit(1)

# ── CATEGORIZE HELPER ──────────────────────────────────────
def categorize(val):
    if val < 87:    return 'Low'
    elif val < 149: return 'Medium'
    else:           return 'High'

# ── TRANSFORM HELPER ───────────────────────────────────────
def transform(df, prep_obj):
    """Transform input using YieldPreprocessor class or dict."""
    df = df.copy()

    if isinstance(prep_obj, dict):
        label_encoders  = prep_obj['label_encoders']
        scaler          = prep_obj['scaler']
        impute_values   = prep_obj.get('impute_values', {})
        numerical_cols  = prep_obj['num_cols']
        feature_columns = prep_obj['feature_columns']

        for col, val in impute_values.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)

        for col, le in label_encoders.items():
            if col in df.columns and col != 'Yield_ton_per_ha':
                df[col] = df[col].map(
                    lambda s, le=le: s if s in le.classes_
                    else le.classes_[0]
                )
                df[col] = le.transform(df[col].astype(str))

        df[numerical_cols] = scaler.transform(df[numerical_cols])
        return df[feature_columns]

    else:
        # YieldPreprocessor class object
        # Check actual attribute names
        scaler = (
            getattr(prep_obj, 'scaler',          None) or
            getattr(prep_obj, 'scalers',         None) or
            getattr(prep_obj, 'standard_scaler', None)
        )
        num_cols = (
            getattr(prep_obj, 'num_cols',        None) or
            getattr(prep_obj, 'numerical_cols',  None) or
            getattr(prep_obj, 'numeric_cols',    None)
        )
        cat_cols = (
            getattr(prep_obj, 'cat_cols',        None) or
            getattr(prep_obj, 'categorical_cols',None)
        )
        label_encoders = (
            getattr(prep_obj, 'label_encoders',  None) or
            getattr(prep_obj, 'encoders',        None)
        )
        iqr_bounds = getattr(prep_obj, 'iqr_bounds', {})

        if scaler is None or num_cols is None:
            print("⚠️  Could not find scaler/num_cols. Attributes:")
            print([a for a in dir(prep_obj)
                   if not a.startswith('_')])
            sys.exit(1)

        # Step 1 — IQR capping
        for col in num_cols:
            if col in iqr_bounds and col in df.columns:
                lower, upper = iqr_bounds[col]
                df[col] = np.clip(df[col], lower, upper)

        # Step 2 — Encode categoricals safely
        for col in cat_cols:
            if col in label_encoders and col in df.columns:
                le = label_encoders[col]
                df[col] = df[col].map(
                    lambda s, le=le: s if s in le.classes_
                    else le.classes_[0]
                )
                df[col] = le.transform(df[col].astype(str))

        # Step 3 — Scale numericals
        df[num_cols] = scaler.transform(df[num_cols])

        # Step 4 — Return in exact training column order
        train_path = os.path.join(BASE, 'data', 'processed',
                                  'yield_train.csv')
        train_cols = pd.read_csv(
            train_path, nrows=1).drop(
            columns=['Yield_ton_per_ha'],
            errors='ignore').columns.tolist()
        return df[train_cols]

# ── PREPARE TEST DATA ──────────────────────────────────────
X = test.drop('Yield_ton_per_ha', axis=1)
y = test['Yield_ton_per_ha']

# ── PREDICT ────────────────────────────────────────────────
y_pred = model.predict(X)

# ── METRICS ────────────────────────────────────────────────
mae  = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2   = r2_scorepyth(y, y_pred)
mape = np.mean(np.abs((y - y_pred) / y)) * 100

actual_cat    = y.apply(categorize)
predicted_cat = pd.Series(y_pred).apply(categorize)
cat_accuracy  = (actual_cat == predicted_cat).mean() * 100

# ── SAMPLE PREDICTIONS TABLE ───────────────────────────────
results = pd.DataFrame({
    'Actual_Yield'   : y.values[:10],
    'Predicted_Yield': y_pred[:10],
    'Error'          : (y.values[:10] - y_pred[:10]),
    'Actual_Category': actual_cat.values[:10],
    'Pred_Category'  : predicted_cat.values[:10],
    'Category_Match' : (actual_cat.values[:10] ==
                        predicted_cat.values[:10])
})

# ── PRINT REPORT ───────────────────────────────────────────
print("=" * 55)
print(" YIELD MODEL — TEST SET EVALUATION REPORT")
print("=" * 55)
print(f" Total Test Samples   : {len(y)}")
print(f" MAE                  : {mae:.4f} tons/ha")
print(f" RMSE                 : {rmse:.4f} tons/ha")
print(f" R²                   : {r2:.4f}")
print(f" MAPE                 : {mape:.2f} %")
print(f" Category Accuracy    : {cat_accuracy:.2f} %")
print("-" * 55)
print(" SAMPLE PREDICTIONS (first 10 rows):")
print(results.to_string(index=False))
print("=" * 55)

# ── EDGE CASE TESTS ────────────────────────────────────────
print("\n EDGE CASE TESTS:")
print("-" * 55)

edge_cases = [
    {
        'label': 'Max rainfall + High fertilizer (expect High)',
        'input': {
            'Crop': 'Maize', 'Region': 'Region_A',
            'Soil_Type': 'Loam', 'Soil_pH': 7.0,
            'Rainfall_mm': 1499.0, 'Temperature_C': 25.0,
            'Humidity_pct': 80.0, 'Fertilizer_Used_kg': 300.0,
            'Irrigation': 'Drip', 'Pesticides_Used_kg': 40.0,
            'Planting_Density': 25.0, 'Previous_Crop': 'Wheat'
        }
    },
    {
        'label': 'Min rainfall + Low fertilizer (expect Low)',
        'input': {
            'Crop': 'Barley', 'Region': 'Region_B',
            'Soil_Type': 'Sandy', 'Soil_pH': 5.8,
            'Rainfall_mm': 200.0, 'Temperature_C': 35.0,
            'Humidity_pct': 20.0, 'Fertilizer_Used_kg': 50.0,
            'Irrigation': 'Flood', 'Pesticides_Used_kg': 5.0,
            'Planting_Density': 5.0, 'Previous_Crop': 'Rice'
        }
    },
    {
        'label': 'Average conditions (expect Medium)',
        'input': {
            'Crop': 'Wheat', 'Region': 'Region_C',
            'Soil_Type': 'Clay', 'Soil_pH': 6.5,
            'Rainfall_mm': 843.0, 'Temperature_C': 27.0,
            'Humidity_pct': 55.0, 'Fertilizer_Used_kg': 175.0,
            'Irrigation': 'Sprinkler', 'Pesticides_Used_kg': 20.0,
            'Planting_Density': 15.0, 'Previous_Crop': 'Maize'
        }
    },
    {
        'label': 'Mode-filled values (expect Medium)',
        'input': {
            'Crop': 'Rice', 'Region': 'Region_D',
            'Soil_Type': 'Loam', 'Soil_pH': 6.5,
            'Rainfall_mm': 843.0, 'Temperature_C': 25.0,
            'Humidity_pct': 60.0, 'Fertilizer_Used_kg': 175.0,
            'Irrigation': 'Sprinkler', 'Pesticides_Used_kg': 20.0,
            'Planting_Density': 15.0, 'Previous_Crop': 'Wheat'
        }
    }
]

for ec in edge_cases:
    try:
        sample    = pd.DataFrame([ec['input']])
        processed = transform(sample, prep)
        pred      = model.predict(processed)[0]
        cat       = categorize(pred)
        print(f" Test : {ec['label']}")
        print(f"   → Predicted : {pred:.2f} tons/ha  [{cat}]")
        print()
    except Exception as e:
        print(f" ❌ Failed: {ec['label']}")
        print(f"    Error: {e}\n")




# import sys, os
# sys.path.append(os.path.abspath(os.path.join('..', 'src')))
# import pandas as pd, numpy as np, joblib
# from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score)
# import os

# BASE = r"C:\Users\Nowshad\Desktop\Antigravity_projects\crop yield prediction"

# # Load
# model = joblib.load(f"{BASE}/models/yield_best_model.pkl")
# test  = pd.read_csv(f"{BASE}/data/processed/yield_test.csv")
# X     = test.drop('Yield_ton_per_ha', axis=1)
# y     = test['Yield_ton_per_ha']

# # Predict
# y_pred = model.predict(X)

# # Metrics
# mae   = mean_absolute_error(y, y_pred)
# rmse  = np.sqrt(mean_squared_error(y, y_pred))
# r2    = r2_score(y, y_pred)
# mape  = np.mean(np.abs((y - y_pred) / y)) * 100

# # Yield category accuracy
# def categorize(val):
#     if val < 87:   return 'Low'
#     elif val < 149: return 'Medium'
#     else:           return 'High'

# actual_cat    = y.apply(categorize)
# predicted_cat = pd.Series(y_pred).apply(categorize)
# cat_accuracy  = (actual_cat == predicted_cat).mean() * 100

# # Sample predictions table (first 10)
# results = pd.DataFrame({
#     'Actual_Yield'    : y.values[:10],
#     'Predicted_Yield' : y_pred[:10],
#     'Error'           : (y.values[:10] - y_pred[:10]),
#     'Actual_Category' : actual_cat.values[:10],
#     'Pred_Category'   : predicted_cat.values[:10],
#     'Category_Match'  : (actual_cat.values[:10] == predicted_cat.values[:10])
# })

# # Print full test report
# print("=" * 55)
# print(" YIELD MODEL — TEST SET EVALUATION REPORT")
# print("=" * 55)
# print(f" Total Test Samples   : {len(y)}")
# print(f" MAE                  : {mae:.4f} tons/ha")
# print(f" RMSE                 : {rmse:.4f} tons/ha")
# print(f" R²                   : {r2:.4f}")
# print(f" MAPE                 : {mape:.2f} %")
# print(f" Category Accuracy    : {cat_accuracy:.2f} %")
# print("-" * 55)
# print(" SAMPLE PREDICTIONS (first 10 rows):")
# print(results.to_string(index=False))
# print("=" * 55)

# # Edge case tests
# print("\n EDGE CASE TESTS:")
# print("-" * 55)

# edge_cases = [
#     {
#         'label': 'Max rainfall + High fertilizer (expect High yield)',
#         'input': {
#             'Crop': 'Maize', 'Region': 'Region_A', 'Soil_Type': 'Loam',
#             'Soil_pH': 7.0, 'Rainfall_mm': 1499.0, 'Temperature_C': 25.0,
#             'Humidity_pct': 80.0, 'Fertilizer_Used_kg': 300.0,
#             'Irrigation': 'Drip', 'Pesticides_Used_kg': 40.0,
#             'Planting_Density': 25.0, 'Previous_Crop': 'Wheat'
#         }
#     },
#     {
#         'label': 'Min rainfall + Low fertilizer (expect Low yield)',
#         'input': {
#             'Crop': 'Barley', 'Region': 'Region_B', 'Soil_Type': 'Sandy',
#             'Soil_pH': 5.8, 'Rainfall_mm': 200.0, 'Temperature_C': 35.0,
#             'Humidity_pct': 20.0, 'Fertilizer_Used_kg': 50.0,
#             'Irrigation': 'Flood', 'Pesticides_Used_kg': 5.0,
#             'Planting_Density': 5.0, 'Previous_Crop': 'Rice'
#         }
#     },
#     {
#         'label': 'Average conditions (expect Medium yield)',
#         'input': {
#             'Crop': 'Wheat', 'Region': 'Region_C', 'Soil_Type': 'Clay',
#             'Soil_pH': 6.5, 'Rainfall_mm': 843.0, 'Temperature_C': 27.0,
#             'Humidity_pct': 55.0, 'Fertilizer_Used_kg': 175.0,
#             'Irrigation': 'Sprinkler', 'Pesticides_Used_kg': 20.0,
#             'Planting_Density': 15.0, 'Previous_Crop': 'Maize'
#         }
#     },
#     {
#         'label': 'Null-like inputs (mode-filled values)',
#         'input': {
#             'Crop': 'Rice', 'Region': 'Region_D', 'Soil_Type': 'Loam',
#             'Soil_pH': 6.5, 'Rainfall_mm': 843.0, 'Temperature_C': 25.0,
#             'Humidity_pct': 60.0, 'Fertilizer_Used_kg': 175.0,
#             'Irrigation': 'Sprinkler', 'Pesticides_Used_kg': 20.0,
#             'Planting_Density': 15.0, 'Previous_Crop': 'Wheat'
#         }
#     }
# ]

# prep = joblib.load(f"{BASE}/models/yield_preprocessor.pkl")
# for ec in edge_cases:
#     sample    = pd.DataFrame([ec['input']])
#     processed = prep.transform(sample)
#     pred      = model.predict(processed)[0]
#     cat       = categorize(pred)
#     print(f" Test : {ec['label']}")
#     print(f"   → Predicted: {pred:.2f} tons/ha  [{cat}]")
#     print()
