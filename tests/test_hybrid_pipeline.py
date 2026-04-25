import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE, 'src'))

from hybrid_pipeline import HybridPipeline

def test_hybrid_pipeline():
    pipeline = HybridPipeline()

    test_cases = [
        {
            'label': 'Scenario 1: Full Combined Inputs (Maize/Loam/Medium conditions)',
            'input': {
                'Crop': 'Maize',
                'Region': 'Region_A',
                'Soil_Type': 'Loam',
                'Soil_pH': 6.5,
                'Rainfall_mm': 900.0,
                'Temperature_C': 25.0,
                'Humidity_pct': 60.0,
                'Fertilizer_Used_kg': 175.0,
                'Irrigation': 'Drip',
                'Pesticides_Used_kg': 20.0,
                'Planting_Density': 15.0,
                'Previous_Crop': 'Wheat',
                'Soil_Moisture': 35.0,
                'Organic_Carbon': 0.8,
                'Electrical_Conductivity': 1.2,
                'Nitrogen_Level': 50,
                'Phosphorus_Level': 40,
                'Potassium_Level': 80,
                'Temperature': 25.0,
                'Humidity': 65.0,
                'Rainfall': 900.0,
                'Crop_Type': 'Maize',
                'Crop_Growth_Stage': 'Vegetative',
                'Season': 'Kharif',
                'Irrigation_Type': 'Drip',
                'Region': 'North',
                'Fertilizer_Used_Last_Season': 150.0,
                'Yield_Last_Season': 4.5
            }
        },
        {
            'label': 'Scenario 2: Minimal Inputs (relies on fallbacks/defaults)',
            'input': {
                'Crop': 'Wheat',
                'Soil_Type': 'Clay',
                'Temperature_C': 15.0,
                'Rainfall_mm': 300.0,
                'Nitrogen_Level': 100,
                'Region': 'Region_B'
            }
        },
        {
            'label': 'Scenario 3: Conflicting alias keys (uses explicit keys first)',
            'input': {
                'Crop_Type': 'Rice',
                'Crop': 'Wheat', # Should use 'Rice' for fertilizer, 'Wheat' for yield
                'Region': 'East',
                'Soil_Type': 'Sandy',
                'Nitrogen_Level': 10,
                'Phosphorus_Level': 90,
                'Fertilizer_Used_kg': 250.0
            }
        }
    ]

    for tc in test_cases:
        print(f"\n{'='*50}")
        print(f"TEST: {tc['label']}")
        print(f"{'='*50}")
        
        # 1. Yield Only
        print("\n--- predict_yield_only ---")
        yield_res = pipeline.predict_yield_only(tc['input'])
        print(yield_res)
        
        # 2. Fertilizer Only
        print("\n--- predict_fertilizer_only ---")
        fert_res = pipeline.predict_fertilizer_only(tc['input'])
        print(fert_res)
        
        # 3. Combined
        print("\n--- predict (combined) ---")
        comb_res = pipeline.predict(tc['input'])
        print(f"Yield Category: {comb_res['yield']['category']}")
        print(f"Fertilizer: {comb_res['fertilizer']['recommended']}")
        print(f"Advice: {comb_res['combined_advice']}")

if __name__ == '__main__':
    test_hybrid_pipeline()
