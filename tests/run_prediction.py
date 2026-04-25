import sys
import os

# Fix path — must be before any src imports
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'src'))

import joblib
import pandas as pd
import numpy as np

# ── LOAD MODELS ────────────────────────────────────────────
try:
    yield_model = joblib.load(os.path.join(BASE, 'models',
                              'yield_best_model.pkl'))
    yield_prep  = joblib.load(os.path.join(BASE, 'models',
                              'yield_preprocessor.pkl'))
    fert_model  = joblib.load(os.path.join(BASE, 'models',
                              'fertilizer_best_model.pkl'))
    fert_prep   = joblib.load(os.path.join(BASE, 'models',
                              'fertilizer_preprocessor.pkl'))
    fert_names  = fert_prep.label_encoders[
                  'Recommended_Fertilizer'].classes_
    print("✅ All models loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    sys.exit(1)

# ── HELPERS ────────────────────────────────────────────────
def categorize_yield(val):
    if val < 87:    return 'Low'
    elif val < 149: return 'Medium'
    else:           return 'High'

def transform_yield(df):
    """Transform yield input using YieldPreprocessor."""
    df = df.copy()
    scaler = (
        getattr(yield_prep, 'scaler',  None) or
        getattr(yield_prep, 'scalers', None)
    )
    num_cols = (
        getattr(yield_prep, 'num_cols',       None) or
        getattr(yield_prep, 'numerical_cols', None)
    )
    cat_cols = (
        getattr(yield_prep, 'cat_cols',        None) or
        getattr(yield_prep, 'categorical_cols',None)
    )
    label_encoders = yield_prep.label_encoders
    iqr_bounds     = getattr(yield_prep, 'iqr_bounds', {})

    # IQR capping
    for col in num_cols:
        if col in iqr_bounds and col in df.columns:
            lower, upper = iqr_bounds[col]
            df[col] = np.clip(df[col], lower, upper)

    # Encode categoricals safely
    for col in cat_cols:
        if col in label_encoders and col in df.columns:
            le = label_encoders[col]
            df[col] = df[col].map(
                lambda s, le=le: s if s in le.classes_
                else le.classes_[0]
            )
            df[col] = le.transform(df[col].astype(str))

    # Scale numericals
    df[num_cols] = scaler.transform(df[num_cols])

    # Return in exact training column order
    train_cols = pd.read_csv(
        os.path.join(BASE, 'data', 'processed', 'yield_train.csv'),
        nrows=1
    ).drop(columns=['Yield_ton_per_ha'],
           errors='ignore').columns.tolist()
    return df[train_cols]

def transform_fertilizer(df):
    """Transform fertilizer input using FertilizerPreprocessor."""
    df = df.copy()
    label_encoders = fert_prep.label_encoders
    iqr_bounds     = getattr(fert_prep, 'iqr_bounds', {})

    # IQR capping
    for col in fert_prep.num_cols:
        if col in iqr_bounds and col in df.columns:
            lower, upper = iqr_bounds[col]
            df[col] = np.clip(df[col], lower, upper)

    # Encode categoricals safely
    for col in fert_prep.cat_cols:
        if col in label_encoders and col in df.columns:
            le = label_encoders[col]
            df[col] = df[col].map(
                lambda s, le=le: s if s in le.classes_
                else le.classes_[0]
            )
            df[col] = le.transform(df[col].astype(str))

    # Scale numericals
    df[fert_prep.num_cols] = fert_prep.scalers.transform(
                              df[fert_prep.num_cols])

    # Return in exact training column order
    feature_columns = [
        'Soil_Type', 'Soil_pH', 'Soil_Moisture', 'Organic_Carbon',
        'Electrical_Conductivity', 'Nitrogen_Level', 'Phosphorus_Level',
        'Potassium_Level', 'Temperature', 'Humidity', 'Rainfall',
        'Crop_Type', 'Crop_Growth_Stage', 'Season', 'Irrigation_Type',
        'Previous_Crop', 'Region', 'Fertilizer_Used_Last_Season',
        'Yield_Last_Season'
    ]
    return df[feature_columns]

# ── YIELD PREDICTION ───────────────────────────────────────
def predict_yield():
    print("\n── YIELD PREDICTION ──────────────────────────────")
    print("Enter values (press Enter to use default):\n")
    crop    = input("  Crop [Maize/Barley/Rice/Wheat]: ") or "Maize"
    region  = input("  Region [Region_A/B/C/D]: ") or "Region_A"
    soil    = input("  Soil_Type [Sandy/Loam/Clay]: ") or "Loam"
    ph      = float(input("  Soil_pH (5.5-7.5) [6.5]: ") or 6.5)
    rain    = float(input("  Rainfall_mm (200-1500) [900]: ") or 900)
    temp    = float(input("  Temperature_C (10-45) [25]: ") or 25)
    humid   = float(input("  Humidity_pct (14-100) [60]: ") or 60)
    fert    = float(input("  Fertilizer_Used_kg (50-300) [175]: ") or 175)
    irr     = input("  Irrigation [Drip/Sprinkler/Flood]: ") or "Drip"
    pest    = float(input("  Pesticides_Used_kg [20]: ") or 20)
    density = float(input("  Planting_Density (5-25) [15]: ") or 15)
    prev    = input("  Previous_Crop [Wheat/Rice/Maize/Barley]: ") or "Wheat"

    try:
        sample = pd.DataFrame([{
            'Crop': crop, 'Region': region, 'Soil_Type': soil,
            'Soil_pH': ph, 'Rainfall_mm': rain,
            'Temperature_C': temp, 'Humidity_pct': humid,
            'Fertilizer_Used_kg': fert, 'Irrigation': irr,
            'Pesticides_Used_kg': pest, 'Planting_Density': density,
            'Previous_Crop': prev
        }])
        processed = transform_yield(sample)
        pred      = yield_model.predict(processed)[0]
        cat       = categorize_yield(pred)
        print(f"\n  ✅ Predicted Yield : {pred:.2f} tons/ha")
        print(f"  📊 Category       : {cat}")
    except Exception as e:
        print(f"\n  ❌ Prediction failed: {e}")

# ── FERTILIZER RECOMMENDATION ──────────────────────────────
def predict_fertilizer():
    print("\n── FERTILIZER RECOMMENDATION ─────────────────────")
    print("Enter values (press Enter to use default):\n")
    soil   = input("  Soil_Type [Clay/Silt/Sandy/Loamy]: ") or "Clay"
    ph     = float(input("  Soil_pH (4.5-8.5) [6.5]: ") or 6.5)
    moist  = float(input("  Soil_Moisture [35]: ") or 35)
    carbon = float(input("  Organic_Carbon [0.8]: ") or 0.8)
    ec     = float(input("  Electrical_Conductivity [1.2]: ") or 1.2)
    n      = int(input("  Nitrogen_Level [50]: ") or 50)
    p      = int(input("  Phosphorus_Level [40]: ") or 40)
    k      = int(input("  Potassium_Level [80]: ") or 80)
    temp   = float(input("  Temperature [25]: ") or 25)
    humid  = float(input("  Humidity [65]: ") or 65)
    rain   = float(input("  Rainfall [900]: ") or 900)
    crop   = input("  Crop_Type [Cotton/Maize/Wheat/Potato/Rice/Sugarcane/Tomato]: ") or "Maize"
    stage  = input("  Crop_Growth_Stage [Sowing/Vegetative/Flowering/Harvest]: ") or "Vegetative"
    season = input("  Season [Kharif/Rabi/Zaid]: ") or "Kharif"
    irr    = input("  Irrigation_Type [Canal/Sprinkler/Rainfed/Drip]: ") or "Drip"
    prev   = input("  Previous_Crop [Wheat/Potato/Tomato/Maize/Sugarcane/Cotton/Rice]: ") or "Wheat"
    region = input("  Region [North/South/East/West/Central]: ") or "North"
    f_last = float(input("  Fertilizer_Used_Last_Season [150]: ") or 150)
    y_last = float(input("  Yield_Last_Season [4.5]: ") or 4.5)

    try:
        sample = pd.DataFrame([{
            'Soil_Type': soil, 'Soil_pH': ph,
            'Soil_Moisture': moist, 'Organic_Carbon': carbon,
            'Electrical_Conductivity': ec,
            'Nitrogen_Level': n, 'Phosphorus_Level': p,
            'Potassium_Level': k, 'Temperature': temp,
            'Humidity': humid, 'Rainfall': rain,
            'Crop_Type': crop, 'Crop_Growth_Stage': stage,
            'Season': season, 'Irrigation_Type': irr,
            'Previous_Crop': prev, 'Region': region,
            'Fertilizer_Used_Last_Season': f_last,
            'Yield_Last_Season': y_last
        }])
        processed = transform_fertilizer(sample)
        pred      = fert_model.predict(processed)[0]
        proba     = (fert_model.predict_proba(processed)[0]
                     if hasattr(fert_model, 'predict_proba')
                     else np.zeros(len(fert_names)))
        top3      = np.argsort(proba)[::-1][:3]

        print(f"\n  ✅ Top Recommendation : "
              f"{fert_names[pred]} "
              f"({proba[pred]*100:.1f}% confidence)")
        print(f"\n  📋 Top 3 Fertilizers  :")
        for rank, idx in enumerate(top3, 1):
            bar = '█' * int(proba[idx] * 20)
            print(f"     #{rank} {fert_names[idx]:<18}"
                  f" {bar:<20} {proba[idx]*100:.1f}%")
    except Exception as e:
        print(f"\n  ❌ Recommendation failed: {e}")

# ── MAIN MENU ──────────────────────────────────────────────
while True:
    print("\n" + "=" * 50)
    print("  SmartFarm — INTERACTIVE TESTER")
    print("=" * 50)
    print("  [1] Predict Crop Yield")
    print("  [2] Recommend Fertilizer")
    print("  [3] Exit")
    print("=" * 50)
    choice = input("  Select option (1/2/3): ").strip()
    if   choice == "1": predict_yield()
    elif choice == "2": predict_fertilizer()
    elif choice == "3": print("\n  Goodbye!\n"); sys.exit()
    else: print("  Invalid choice. Enter 1, 2, or 3.")