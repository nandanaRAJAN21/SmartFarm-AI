from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import sys
import uvicorn
import logging

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, 'src'))
sys.path.insert(0, os.path.join(BASE, 'backend'))

from schemas import (
    YieldPredictionRequest, YieldPredictionResponse,
    FertilizerRecommendationRequest, FertilizerRecommendationResponse,
    CombinedPredictionRequest, CombinedPredictionResponse,
    Explanation, FeatureImportance, TopFertilizer
)
from hybrid_pipeline import HybridPipeline
from explainer import Explainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SmartFarm Engine")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = None
explainer = None

@app.on_event("startup")
def load_models():
    global pipeline, explainer
    try:
        pipeline = HybridPipeline()
        explainer = Explainer(pipeline)
        logger.info("Models and Explainer loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": pipeline is not None}

@app.get("/model/info")
def model_info():
    return {
        "models": {
            "yield": "Linear Regression Phase 1 (R2: 0.9821, RMSE: 5.07 tons/ha)",
            "fertilizer": "Random Forest Phase 1 (Acc: 94.98%, F1: 95.05%)"
        }
    }

@app.post("/predict/yield", response_model=YieldPredictionResponse)
def predict_yield_endpoint(req: YieldPredictionRequest):
    try:
        data = req.dict()
        res = pipeline.predict_yield_only(data)
        exp = explainer.explain_yield(data, method='shap')
        
        return YieldPredictionResponse(
            predicted_value=res['predicted_value'],
            category=res['category'],
            explanation=Explanation(**exp)
        )
    except Exception as e:
        logger.error(f"Yield prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/fertilizer", response_model=FertilizerRecommendationResponse)
def predict_fertilizer_endpoint(req: FertilizerRecommendationRequest):
    try:
        data = req.dict()
        res = pipeline.predict_fertilizer_only(data)
        exp = explainer.explain_fertilizer(data, method='shap')
        
        return FertilizerRecommendationResponse(
            recommended=res['recommended'],
            confidence=res['confidence'],
            top3=[TopFertilizer(**t) for t in res['top3']],
            explanation=Explanation(**exp)
        )
    except Exception as e:
        logger.error(f"Fertilizer prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/combined", response_model=CombinedPredictionResponse)
def predict_combined_endpoint(req: CombinedPredictionRequest):
    try:
        data = req.dict()
        
        comb_res = pipeline.predict(data)
        exp_yield = explainer.explain_yield(data, method='shap')
        exp_fert = explainer.explain_fertilizer(data, method='shap')
        
        y_resp = YieldPredictionResponse(
            predicted_value=comb_res['yield']['predicted_value'],
            category=comb_res['yield']['category'],
            explanation=Explanation(**exp_yield)
        )
        
        f_resp = FertilizerRecommendationResponse(
            recommended=comb_res['fertilizer']['recommended'],
            confidence=comb_res['fertilizer']['confidence'],
            top3=[TopFertilizer(**t) for t in comb_res['fertilizer']['top3']],
            explanation=Explanation(**exp_fert)
        )
        
        return CombinedPredictionResponse(
            yield_result=y_resp,
            fertilizer_result=f_resp,
            combined_advice=comb_res['combined_advice']
        )
    except Exception as e:
        logger.error(f"Combined prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)