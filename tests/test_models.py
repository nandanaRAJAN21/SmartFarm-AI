import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'src')))
import pandas as pd, numpy as np, joblib
import os

BASE = r"C:\Users\NANDANA\OneDrive\Desktop\smartfarm\crop yield prediction"

# Load all models
try:
    yield_model  = joblib.load(f"{BASE}/models/yield_best_model.pkl")
    yield_prep   = joblib.load(f"{BASE}/models/yield_preprocessor.pkl")
    fert_model   = joblib.load(f"{BASE}/models/fertilizer_best_model.pkl")
    fert_prep    = joblib.load(f"{BASE}/models/fertilizer_preprocessor.pkl")
    fert_names   = fert_prep.label_encoders['Recommended_Fertilizer'].classes_
except Exception as e:
    print(f"Error loading models: {e}. Ensure you've run the training notebooks/scripts first.")
    import sys; sys.exit(1)

def categorize_yield(val):
    if val < 87:    return 'Low'
    elif val < 149: return 'Medium'
    else:           return 'High'

# 6 combined test scenarios
scenarios = [
    {
        'label': 'Maize | High rainfall | Sandy | Region_A',
        'yield_input': {
            'Crop': 'Maize', 'Region': 'Region_A',
            'Soil_Type': 'Sandy', 'Soil_pH': 6.8,
            'Rainfall_mm': 1200.0, 'Temperature_C': 22.0,
            'Humidity_pct': 70.0, 'Fertilizer_Used_kg': 200.0,
            'Irrigation': 'Drip', 'Pesticides_Used_kg': 15.0,
            'Planting_Density': 20.0, 'Previous_Crop': 'Rice'
        },
        'fert_input': {
            'Soil_Type': 'Sandy', 'Soil_pH': 6.8,
            'Soil_Moisture': 40.0, 'Organic_Carbon': 0.9,
            'Electrical_Conductivity': 1.0,
            'Nitrogen_Level': 40, 'Phosphorus_Level': 30,
            'Potassium_Level': 70, 'Temperature': 22.0,
            'Humidity': 70.0, 'Rainfall': 1200.0,
            'Crop_Type': 'Maize', 'Crop_Growth_Stage': 'Sowing',
            'Season': 'Kharif', 'Irrigation_Type': 'Drip',
            'Previous_Crop': 'Rice', 'Region': 'North',
            'Fertilizer_Used_Last_Season': 180.0,
            'Yield_Last_Season': 5.0
        }
    },
    {
        'label': 'Wheat | Low rainfall | Clay | Region_B',
        'yield_input': {
            'Crop': 'Wheat', 'Region': 'Region_B',
            'Soil_Type': 'Clay', 'Soil_pH': 7.0,
            'Rainfall_mm': 400.0, 'Temperature_C': 18.0,
            'Humidity_pct': 45.0, 'Fertilizer_Used_kg': 100.0,
            'Irrigation': 'Flood', 'Pesticides_Used_kg': 10.0,
            'Planting_Density': 10.0, 'Previous_Crop': 'Barley'
        },
        'fert_input': {
            'Soil_Type': 'Clay', 'Soil_pH': 7.0,
            'Soil_Moisture': 25.0, 'Organic_Carbon': 0.5,
            'Electrical_Conductivity': 1.5,
            'Nitrogen_Level': 70, 'Phosphorus_Level': 55,
            'Potassium_Level': 40, 'Temperature': 18.0,
            'Humidity': 45.0, 'Rainfall': 400.0,
            'Crop_Type': 'Wheat', 'Crop_Growth_Stage': 'Flowering',
            'Season': 'Rabi', 'Irrigation_Type': 'Flood',
            'Previous_Crop': 'Barley', 'Region': 'Central',
            'Fertilizer_Used_Last_Season': 90.0,
            'Yield_Last_Season': 3.2
        }
    },
    {
        'label': 'Rice | Medium conditions | Loam | Region_C',
        'yield_input': {
            'Crop': 'Rice', 'Region': 'Region_C',
            'Soil_Type': 'Loam', 'Soil_pH': 6.5,
            'Rainfall_mm': 900.0, 'Temperature_C': 28.0,
            'Humidity_pct': 75.0, 'Fertilizer_Used_kg': 175.0,
            'Irrigation': 'Sprinkler', 'Pesticides_Used_kg': 22.0,
            'Planting_Density': 15.0, 'Previous_Crop': 'Wheat'
        },
        'fert_input': {
            'Soil_Type': 'Loam', 'Soil_pH': 6.5,
            'Soil_Moisture': 45.0, 'Organic_Carbon': 1.1,
            'Electrical_Conductivity': 1.1,
            'Nitrogen_Level': 55, 'Phosphorus_Level': 45,
            'Potassium_Level': 60, 'Temperature': 28.0,
            'Humidity': 75.0, 'Rainfall': 900.0,
            'Crop_Type': 'Rice', 'Crop_Growth_Stage': 'Vegetative',
            'Season': 'Kharif', 'Irrigation_Type': 'Sprinkler',
            'Previous_Crop': 'Wheat', 'Region': 'East',
            'Fertilizer_Used_Last_Season': 160.0,
            'Yield_Last_Season': 4.2
        }
    },
    {
        'label': 'Barley | High temp | Sandy | Region_D',
        'yield_input': {
            'Crop': 'Barley', 'Region': 'Region_D',
            'Soil_Type': 'Sandy', 'Soil_pH': 6.2,
            'Rainfall_mm': 600.0, 'Temperature_C': 38.0,
            'Humidity_pct': 35.0, 'Fertilizer_Used_kg': 130.0,
            'Irrigation': 'Drip', 'Pesticides_Used_kg': 12.0,
            'Planting_Density': 12.0, 'Previous_Crop': 'Maize'
        },
        'fert_input': {
            'Soil_Type': 'Sandy', 'Soil_pH': 6.2,
            'Soil_Moisture': 20.0, 'Organic_Carbon': 0.4,
            'Electrical_Conductivity': 0.8,
            'Nitrogen_Level': 45, 'Phosphorus_Level': 35,
            'Potassium_Level': 50, 'Temperature': 38.0,
            'Humidity': 35.0, 'Rainfall': 600.0,
            'Crop_Type': 'Maize', 'Crop_Growth_Stage': 'Harvest',
            'Season': 'Zaid', 'Irrigation_Type': 'Drip',
            'Previous_Crop': 'Maize', 'Region': 'West',
            'Fertilizer_Used_Last_Season': 110.0,
            'Yield_Last_Season': 3.5
        }
    }
]

# Run all scenarios
print("\n" + "=" * 65)
print("  COMBINED MODEL TEST — ALL SCENARIOS")
print("=" * 65)

for i, sc in enumerate(scenarios, 1):
    # Yield prediction
    y_in   = pd.DataFrame([sc['yield_input']])
    y_proc = yield_prep.transform(y_in)
    y_pred = yield_model.predict(y_proc)[0]
    y_cat  = categorize_yield(y_pred)

    # Fertilizer prediction
    f_in    = pd.DataFrame([sc['fert_input']])
    f_proc  = fert_prep.transform(f_in)
    f_pred  = fert_model.predict(f_proc)[0]
    if hasattr(fert_model, 'predict_proba'):
        f_proba = fert_model.predict_proba(f_proc)[0]
    else:
        f_proba = np.zeros(len(fert_names))
        f_proba[f_pred] = 1.0

    top3    = np.argsort(f_proba)[::-1][:3]

    print(f"\n SCENARIO {i}: {sc['label']}")
    print(f" {'-'*55}")
    print(f"  [YIELD] Predicted Yield     : {y_pred:.2f} tons/ha [{y_cat}]")
    print(f"  [FERT] Top Fertilizer      : {fert_names[f_pred]} ({f_proba[f_pred]*100:.1f}%)")
    print(f"  [LIST] Top 3 Fertilizers   :")
    for rank, idx in enumerate(top3, 1):
        bar = '#' * int(f_proba[idx] * 20)
        print(f"       #{rank} {fert_names[idx]:<18} {bar:<20} {f_proba[idx]*100:.1f}%")

print("\n" + "=" * 65)
print("  ALL SCENARIOS COMPLETE")
print("=" * 65)
