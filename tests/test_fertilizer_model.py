import sys
import os

# Fix path — must be before any src imports
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'src'))

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (accuracy_score, f1_score,
                              classification_report,
                              confusion_matrix)

# ── LOAD MODELS ────────────────────────────────────────────
try:
    model = joblib.load(os.path.join(BASE, 'models',
                        'fertilizer_best_model.pkl'))
    prep  = joblib.load(os.path.join(BASE, 'models',
                        'fertilizer_preprocessor.pkl'))
    test  = pd.read_csv(os.path.join(BASE, 'data',
                        'processed', 'fertilizer_test.csv'))
    print("✅ Models and test data loaded successfully")
except Exception as e:
    print(f"❌ Error loading: {e}")
    sys.exit(1)

# ── TRANSFORM HELPER ───────────────────────────────────────
# Works whether prep is a class object OR a plain dict

def transform(df, prep_obj):
    """Transform input df using preprocessor (class or dict)."""
    df = df.copy()

    # Get components whether prep is class or dict
    if isinstance(prep_obj, dict):
        label_encoders  = prep_obj['label_encoders']
        scaler          = prep_obj['scalers']
        impute_values   = prep_obj.get('impute_values', {})
        numerical_cols  = prep_obj['num_cols']
        feature_columns = prep_obj['feature_columns']
    else:
        label_encoders  = prep_obj.label_encoders
        scaler          = prep_obj.scalers
        impute_values   = getattr(prep_obj, 'impute_values', {})
        numerical_cols  = prep_obj.num_cols
        feature_columns = [
    'Soil_Type', 'Soil_pH', 'Soil_Moisture', 'Organic_Carbon',
    'Electrical_Conductivity', 'Nitrogen_Level', 'Phosphorus_Level',
    'Potassium_Level', 'Temperature', 'Humidity', 'Rainfall',
    'Crop_Type', 'Crop_Growth_Stage', 'Season', 'Irrigation_Type',
    'Previous_Crop', 'Region', 'Fertilizer_Used_Last_Season',
    'Yield_Last_Season'
]

    # Impute missing values
    for col, val in impute_values.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    # Label encode categoricals
    for col, le in label_encoders.items():
        if col in df.columns and col != 'Recommended_Fertilizer':
            df[col] = df[col].map(
                lambda s, le=le: s if s in le.classes_
                else le.classes_[0]
            )
            df[col] = le.transform(df[col].astype(str))

    # Scale numericals
    df[numerical_cols] = scaler.transform(df[numerical_cols])

    return df[feature_columns]

# ── GET FERTILIZER CLASS NAMES ─────────────────────────────
try:
    if isinstance(prep, dict):
        fert_names = prep['target_classes']
    else:
        fert_names = prep.label_encoders[
                     'Recommended_Fertilizer'].classes_
except Exception as e:
    print(f"❌ Could not get fertilizer class names: {e}")
    sys.exit(1)

# ── PREPARE TEST DATA ──────────────────────────────────────
X = test.drop('Recommended_Fertilizer', axis=1)
y = test['Recommended_Fertilizer']

# ── PREDICT ────────────────────────────────────────────────
y_pred  = model.predict(X)
y_proba = model.predict_proba(X) if hasattr(
          model, 'predict_proba') else None

if y_proba is None:
    y_proba = np.zeros((len(y), len(fert_names)))
    for i in range(len(y_pred)):
        y_proba[i, y_pred[i]] = 1.0

# ── METRICS ────────────────────────────────────────────────
acc = accuracy_score(y, y_pred) * 100
f1  = f1_score(y, y_pred, average='weighted') * 100

top3_correct = 0
for i, true_label in enumerate(y):
    top3_idx = np.argsort(y_proba[i])[::-1][:3]
    if true_label in top3_idx:
        top3_correct += 1
top3_acc = (top3_correct / len(y)) * 100

# ── SAMPLE PREDICTIONS TABLE ───────────────────────────────
results = pd.DataFrame({
    'Actual'    : [fert_names[i] for i in y.values[:10]],
    'Predicted' : [fert_names[i] for i in y_pred[:10]],
    'Confidence': [f"{max(y_proba[i])*100:.1f}%"
                   for i in range(10)],
    'Correct'   : [y.values[i] == y_pred[i]
                   for i in range(10)]
})

# ── PRINT REPORT ───────────────────────────────────────────
print("=" * 60)
print(" FERTILIZER MODEL — TEST SET EVALUATION REPORT")
print("=" * 60)
print(f" Total Test Samples  : {len(y)}")
print(f" Accuracy            : {acc:.2f} %")
print(f" F1-Score (weighted) : {f1:.2f} %")
print(f" Top-3 Accuracy      : {top3_acc:.2f} %")
print("-" * 60)
print(" CLASSIFICATION REPORT:")
print(classification_report(y, y_pred,
      target_names=fert_names))
print("-" * 60)
print(" SAMPLE PREDICTIONS (first 10 rows):")
print(results.to_string(index=False))
print("=" * 60)

# ── PER-CLASS ACCURACY ─────────────────────────────────────
print("\n PER-CLASS ACCURACY:")
print("-" * 40)
cm = confusion_matrix(y, y_pred)
for i, fname in enumerate(fert_names):
    class_acc = (cm[i, i] / cm[i].sum() * 100
                 if cm[i].sum() > 0 else 0.0)
    print(f"  {fname:<20} : {class_acc:.1f} %")

# ── EDGE CASE TESTS ────────────────────────────────────────
print("\n EDGE CASE TESTS:")
print("-" * 60)

edge_cases = [
    {
        'label': 'High N, low P/K → expect Urea',
        'input': {
            'Soil_Type': 'Loamy', 'Soil_pH': 6.5,
            'Soil_Moisture': 35.0, 'Organic_Carbon': 0.8,
            'Electrical_Conductivity': 1.2,
            'Nitrogen_Level': 10, 'Phosphorus_Level': 80,
            'Potassium_Level': 75, 'Temperature': 25.0,
            'Humidity': 60.0, 'Rainfall': 900.0,
            'Crop_Type': 'Wheat',
            'Crop_Growth_Stage': 'Vegetative',
            'Season': 'Rabi', 'Irrigation_Type': 'Sprinkler',
            'Previous_Crop': 'Rice', 'Region': 'North',
            'Fertilizer_Used_Last_Season': 150.0,
            'Yield_Last_Season': 4.5
        }
    },
    {
        'label': 'Low N/P/K, organic soil → expect Compost',
        'input': {
            'Soil_Type': 'Clay', 'Soil_pH': 6.0,
            'Soil_Moisture': 50.0, 'Organic_Carbon': 2.5,
            'Electrical_Conductivity': 0.5,
            'Nitrogen_Level': 20, 'Phosphorus_Level': 15,
            'Potassium_Level': 20, 'Temperature': 22.0,
            'Humidity': 75.0, 'Rainfall': 1100.0,
            'Crop_Type': 'Rice',
            'Crop_Growth_Stage': 'Sowing',
            'Season': 'Kharif', 'Irrigation_Type': 'Canal',
            'Previous_Crop': 'Wheat', 'Region': 'East',
            'Fertilizer_Used_Last_Season': 50.0,
            'Yield_Last_Season': 2.5
        }
    },
    {
        'label': 'High P deficiency → expect DAP',
        'input': {
            'Soil_Type': 'Sandy', 'Soil_pH': 7.0,
            'Soil_Moisture': 30.0, 'Organic_Carbon': 0.6,
            'Electrical_Conductivity': 1.0,
            'Nitrogen_Level': 60, 'Phosphorus_Level': 10,
            'Potassium_Level': 70, 'Temperature': 28.0,
            'Humidity': 50.0, 'Rainfall': 700.0,
            'Crop_Type': 'Maize',
            'Crop_Growth_Stage': 'Flowering',
            'Season': 'Zaid', 'Irrigation_Type': 'Drip',
            'Previous_Crop': 'Cotton', 'Region': 'West',
            'Fertilizer_Used_Last_Season': 120.0,
            'Yield_Last_Season': 3.8
        }
    },
    {
        'label': 'High K deficiency → expect MOP',
        'input': {
            'Soil_Type': 'Silt', 'Soil_pH': 6.8,
            'Soil_Moisture': 40.0, 'Organic_Carbon': 1.0,
            'Electrical_Conductivity': 1.3,
            'Nitrogen_Level': 65, 'Phosphorus_Level': 55,
            'Potassium_Level': 10, 'Temperature': 30.0,
            'Humidity': 65.0, 'Rainfall': 850.0,
            'Crop_Type': 'Sugarcane',
            'Crop_Growth_Stage': 'Vegetative',
            'Season': 'Kharif', 'Irrigation_Type': 'Flood',
            'Previous_Crop': 'Maize', 'Region': 'South',
            'Fertilizer_Used_Last_Season': 200.0,
            'Yield_Last_Season': 5.5
        }
    }
]

for ec in edge_cases:
    try:
        sample    = pd.DataFrame([ec['input']])
        processed = transform(sample, prep)
        pred      = model.predict(processed)[0]
        proba     = (model.predict_proba(processed)[0]
                     if hasattr(model, 'predict_proba')
                     else np.zeros(len(fert_names)))
        top3_idx  = np.argsort(proba)[::-1][:3]

        print(f" Test : {ec['label']}")
        print(f"   → Top : {fert_names[pred]}"
              f" ({proba[pred]*100:.1f}%)")
        print(f"   → Top 3:")
        for rank, idx in enumerate(top3_idx, 1):
            bar = '█' * int(proba[idx] * 20)
            print(f"       #{rank} {fert_names[idx]:<18}"
                  f" {bar:<20} {proba[idx]*100:.1f}%")
        print()
    except Exception as e:
        print(f" ❌ Edge case failed: {ec['label']}")
        print(f"    Error: {e}\n")



# import sys, os
# sys.path.append(os.path.abspath(os.path.join('..', 'src')))
# import pandas as pd, numpy as np, joblib
# from sklearn.metrics import (accuracy_score, f1_score, classification_report, confusion_matrix)
# import os

# BASE = r"C:\Users\Nowshad\Desktop\Antigravity_projects\crop yield prediction"

# # Load
# try:
#     model      = joblib.load(f"{BASE}/models/fertilizer_best_model.pkl")
#     prep       = joblib.load(f"{BASE}/models/fertilizer_preprocessor.pkl")
#     test       = pd.read_csv(f"{BASE}/data/processed/fertilizer_test.csv")
# except Exception as e:
#     print(f"Error loading models or test data: {e}. Ensure you've run the training notebooks/scripts first.")
#     import sys; sys.exit(1)

# fert_names = prep.label_encoders['Recommended_Fertilizer'].classes_
# X          = test.drop('Recommended_Fertilizer', axis=1)
# y          = test['Recommended_Fertilizer']

# # Predict
# y_pred  = model.predict(X)
# if hasattr(model, 'predict_proba'):
#     y_proba = model.predict_proba(X)
# else:
#     # Dummy probabilities if not supported (e.g. some CatBoost setups or others)
#     y_proba = np.zeros((len(y), len(fert_names)))
#     for i in range(len(y_pred)):
#         y_proba[i, y_pred[i]] = 1.0

# # Metrics
# acc = accuracy_score(y, y_pred) * 100
# f1  = f1_score(y, y_pred, average='weighted') * 100

# # Top-3 accuracy
# top3_correct = 0
# for i, true_label in enumerate(y):
#     top3_idx = np.argsort(y_proba[i])[::-1][:3]
#     if true_label in top3_idx:
#         top3_correct += 1
# top3_acc = (top3_correct / len(y)) * 100

# # Sample predictions (first 10)
# results = pd.DataFrame({
#     'Actual'      : [fert_names[i] for i in y.values[:10]],
#     'Predicted'   : [fert_names[i] for i in y_pred[:10]],
#     'Confidence'  : [f"{max(y_proba[i])*100:.1f}%" for i in range(10)],
#     'Correct'     : [y.values[i] == y_pred[i] for i in range(10)]
# })

# # Print full test report
# print("=" * 60)
# print(" FERTILIZER MODEL — TEST SET EVALUATION REPORT")
# print("=" * 60)
# print(f" Total Test Samples   : {len(y)}")
# print(f" Accuracy             : {acc:.2f} %")
# print(f" F1-Score (weighted)  : {f1:.2f} %")
# print(f" Top-3 Accuracy       : {top3_acc:.2f} %")
# print("-" * 60)
# print(" CLASSIFICATION REPORT:")
# print(classification_report(y, y_pred, target_names=fert_names))
# print("-" * 60)
# print(" SAMPLE PREDICTIONS (first 10 rows):")
# print(results.to_string(index=False))
# print("=" * 60)

# # Per-class accuracy
# print("\n PER-CLASS ACCURACY:")
# print("-" * 40)
# cm = confusion_matrix(y, y_pred)
# for i, fname in enumerate(fert_names):
#     if cm[i].sum() > 0:
#         class_acc = cm[i, i] / cm[i].sum() * 100
#     else:
#         class_acc = 0.0
#     print(f"  {fname:<20} : {class_acc:.1f} %")

# # Edge case tests
# print("\n EDGE CASE TESTS:")
# print("-" * 60)

# edge_cases = [
#     {
#         'label': 'High N, low P/K → expect Urea (N-focused)',
#         'input': {
#             'Soil_Type': 'Loam', 'Soil_pH': 6.5, 'Soil_Moisture': 35.0, 'Organic_Carbon': 0.8,
#             'Electrical_Conductivity': 1.2, 'Nitrogen_Level': 10, 'Phosphorus_Level': 80, 
#             'Potassium_Level': 75, 'Temperature': 25.0, 'Humidity': 60.0, 'Rainfall': 900.0, 
#             'Crop_Type': 'Wheat', 'Crop_Growth_Stage': 'Vegetative', 'Season': 'Rabi',
#             'Irrigation_Type': 'Sprinkler', 'Previous_Crop': 'Rice', 'Region': 'North', 
#             'Fertilizer_Used_Last_Season': 150.0, 'Yield_Last_Season': 4.5
#         }
#     },
#     {
#         'label': 'Low N/P/K, organic soil → expect Compost',
#         'input': {
#             'Soil_Type': 'Clay', 'Soil_pH': 6.0, 'Soil_Moisture': 50.0, 'Organic_Carbon': 2.5,
#             'Electrical_Conductivity': 0.5, 'Nitrogen_Level': 20, 'Phosphorus_Level': 15, 
#             'Potassium_Level': 20, 'Temperature': 22.0, 'Humidity': 75.0, 'Rainfall': 1100.0, 
#             'Crop_Type': 'Rice', 'Crop_Growth_Stage': 'Sowing', 'Season': 'Kharif',
#             'Irrigation_Type': 'Canal', 'Previous_Crop': 'Wheat', 'Region': 'East', 
#             'Fertilizer_Used_Last_Season': 50.0, 'Yield_Last_Season': 2.5
#         }
#     },
#     {
#         'label': 'High P deficiency → expect DAP (P-focused)',
#         'input': {
#             'Soil_Type': 'Sandy', 'Soil_pH': 7.0, 'Soil_Moisture': 30.0, 'Organic_Carbon': 0.6,
#             'Electrical_Conductivity': 1.0, 'Nitrogen_Level': 60, 'Phosphorus_Level': 10, 
#             'Potassium_Level': 70, 'Temperature': 28.0, 'Humidity': 50.0, 'Rainfall': 700.0, 
#             'Crop_Type': 'Maize', 'Crop_Growth_Stage': 'Flowering', 'Season': 'Zaid',
#             'Irrigation_Type': 'Drip', 'Previous_Crop': 'Cotton', 'Region': 'West', 
#             'Fertilizer_Used_Last_Season': 120.0, 'Yield_Last_Season': 3.8
#         }
#     },
#     {
#         'label': 'High K deficiency → expect MOP (K-focused)',
#         'input': {
#             'Soil_Type': 'Silt', 'Soil_pH': 6.8, 'Soil_Moisture': 40.0, 'Organic_Carbon': 1.0,
#             'Electrical_Conductivity': 1.3, 'Nitrogen_Level': 65, 'Phosphorus_Level': 55, 
#             'Potassium_Level': 10, 'Temperature': 30.0, 'Humidity': 65.0, 'Rainfall': 850.0, 
#             'Crop_Type': 'Sugarcane', 'Crop_Growth_Stage': 'Vegetative', 'Season': 'Kharif',
#             'Irrigation_Type': 'Flood', 'Previous_Crop': 'Maize', 'Region': 'South', 
#             'Fertilizer_Used_Last_Season': 200.0, 'Yield_Last_Season': 5.5
#         }
#     }
# ]

# for ec in edge_cases:
#     sample    = pd.DataFrame([ec['input']])
#     processed = prep.transform(sample)
#     pred      = model.predict(processed)[0]
#     if hasattr(model, 'predict_proba'):
#         proba = model.predict_proba(processed)[0]
#     else:
#         proba = np.zeros(len(fert_names))
#         proba[pred] = 1.0
        
#     top3_idx  = np.argsort(proba)[::-1][:3]
#     print(f" Test : {ec['label']}")
#     print(f"   → Top Recommendation : {fert_names[pred]} ({proba[pred]*100:.1f}% confidence)")
#     print(f"   → Top 3:")
#     for rank, idx in enumerate(top3_idx, 1):
#         print(f"       #{rank} {fert_names[idx]:<20} {proba[idx]*100:.1f}%")
#     print()
