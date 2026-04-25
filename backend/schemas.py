from pydantic import BaseModel, Field
from typing import List, Optional

class YieldPredictionRequest(BaseModel):
    Crop: str = Field(..., description="Crop type (e.g., Maize, Barley, Rice, Wheat)")
    Region: str = Field(..., description="Region category")
    Soil_Type: str = Field(..., description="Soil type category")
    Soil_pH: float
    Rainfall_mm: float
    Temperature_C: float
    Humidity_pct: float
    Fertilizer_Used_kg: float
    Irrigation: str = Field(..., description="Irrigation method (e.g., Drip, Sprinkler, Flood)")
    Pesticides_Used_kg: float
    Planting_Density: float
    Previous_Crop: str

class FeatureImportance(BaseModel):
    feature: str
    importance: float
    direction: str

class Explanation(BaseModel):
    method: str
    feature_importances: List[FeatureImportance]
    summary: str

class YieldPredictionResponse(BaseModel):
    predicted_value: float
    category: str
    explanation: Optional[Explanation] = None

class FertilizerRecommendationRequest(BaseModel):
    Soil_Type: str
    Soil_pH: float
    Soil_Moisture: float
    Organic_Carbon: float
    Electrical_Conductivity: float
    Nitrogen_Level: float
    Phosphorus_Level: float
    Potassium_Level: float
    Temperature: float
    Humidity: float
    Rainfall: float
    Crop_Type: str
    Crop_Growth_Stage: str
    Season: str
    Irrigation_Type: str
    Previous_Crop: str
    Region: str
    Fertilizer_Used_Last_Season: float
    Yield_Last_Season: float

class TopFertilizer(BaseModel):
    name: str
    probability: float

class FertilizerRecommendationResponse(BaseModel):
    recommended: str
    confidence: float
    top3: List[TopFertilizer]
    explanation: Optional[Explanation] = None

class CombinedPredictionRequest(BaseModel):
    # Base fields representing a merge of the two schemas to supply all required inputs
    Crop: str
    Region: str
    Soil_Type: str
    Soil_pH: float
    Rainfall_mm: float
    Temperature_C: float
    Humidity_pct: float
    Fertilizer_Used_kg: float
    Irrigation: str
    Pesticides_Used_kg: float
    Planting_Density: float
    Previous_Crop: str
    
    # Fertilizer specific fields
    Soil_Moisture: float
    Organic_Carbon: float
    Electrical_Conductivity: float
    Nitrogen_Level: float
    Phosphorus_Level: float
    Potassium_Level: float
    Crop_Growth_Stage: str
    Season: str
    Yield_Last_Season: float

class CombinedPredictionResponse(BaseModel):
    yield_result: YieldPredictionResponse
    fertilizer_result: FertilizerRecommendationResponse
    combined_advice: str
