import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE, 'src'))

from hybrid_pipeline import HybridPipeline
from explainer import Explainer

def test_explainer():
    pipeline = HybridPipeline()
    explainer = Explainer(pipeline)
    
    test_input = {
        'Crop': 'Wheat',
        'Region': 'Region_A',
        'Soil_Type': 'Loam',
        'Soil_pH': 6.5,
        'Rainfall_mm': 600.0,
        'Temperature_C': 20.0,
        'Humidity_pct': 55.0,
        'Fertilizer_Used_kg': 150.0,
        'Irrigation': 'Sprinkler',
        'Pesticides_Used_kg': 15.0,
        'Planting_Density': 20.0,
        'Previous_Crop': 'Maize',
        'Soil_Moisture': 40.0,
        'Organic_Carbon': 1.0,
        'Electrical_Conductivity': 1.1,
        'Nitrogen_Level': 80,
        'Phosphorus_Level': 45,
        'Potassium_Level': 60,
        'Crop_Type': 'Wheat',
        'Crop_Growth_Stage': 'Vegetative',
        'Season': 'Rabi',
        'Irrigation_Type': 'Sprinkler',
        'Region_Fertilizer': 'North', # Just to separate alias mapping logic test if needed
        'Fertilizer_Used_Last_Season': 130.0,
        'Yield_Last_Season': 4.0
    }

    methods = ['shap', 'lime']
    
    for method in methods:
        print(f"\n==============================================")
        print(f"TESTING EXPLAINER: {method.upper()}")
        print(f"==============================================")
        
        print("\n--- explain_yield ---")
        yield_exp = explainer.explain_yield(test_input, method=method)
        print(f"Method: {yield_exp['method']}")
        print(f"Summary: {yield_exp['summary']}")
        print("Top 5 Features:")
        for feat in yield_exp['feature_importances'][:5]:
            print(f"  - {feat['feature']}: {feat['importance']:.4f} ({feat['direction']})")
            
        print("\n--- explain_fertilizer ---")
        fert_exp = explainer.explain_fertilizer(test_input, method=method)
        print(f"Method: {fert_exp['method']}")
        print(f"Summary: {fert_exp['summary']}")
        print("Top 5 Features:")
        for feat in fert_exp['feature_importances'][:5]:
            print(f"  - {feat['feature']}: {feat['importance']:.4f} ({feat['direction']})")

if __name__ == '__main__':
    test_explainer()
