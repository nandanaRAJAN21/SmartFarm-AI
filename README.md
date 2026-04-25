# Crop Yield Prediction & Fertilizer Recommendation

A comprehensive, end-to-end Machine Learning ecosystem utilizing a Hybrid Pipeline to predict crop yields and recommend optimal fertilizers. The system is enhanced with Explainable AI (XAI) using SHAP and LIME, accessible via a robust FastAPI backend and an interactive React frontend.

## Features Completed (Phases 1 & 2)

- **Data Processing**: Specialized preprocessing pipelines with intelligent handling for missing values, outliers, and categorical encoding.
- **Yield Prediction (Regression)**: Predicts expected yield (tons/ha). Best model: *Linear Regression* (R²: 0.9821, RMSE: 5.07 tons/ha).
- **Fertilizer Recommendation (Classification)**: Recommends optimal fertilizer types. Best model: *Random Forest* (Accuracy: 94.98%, F1: 95.05%).
- **Hybrid Pipeline**: A unified pipeline (`src/hybrid_pipeline.py`) that handles inference seamlessly across varied inputs.
- **Explainable AI**: SHAP and LIME integrations (`src/explainer.py`) detail *why* models make specific predictions.
- **FastAPI Backend**: A resilient API serving the models and XAI features.
- **React Frontend**: A modern, interactive application using TailwindCSS, Framer Motion, and Recharts.

---

## Directory Structure

```plaintext
CROP_YIELD_PREDICTION/
├── backend/                  # FastAPI Application
│   ├── main.py               # API endpoints
│   └── schemas.py            # Pydantic schemas
├── data/                     # Raw & Processed Datasets
├── frontend/                 # React Application (Vite, Tailwind, Recharts)
├── models/                   # Pickled ML models and preprocessors
├── notebooks/                # Jupyter Notebooks for EDA and Training
├── src/                      # Core Logic
│   ├── data_loader.py
│   ├── explainer.py          # SHAP & LIME 
│   ├── fertilizer_model.py
│   ├── fertilizer_preprocessor.py
│   ├── hybrid_pipeline.py    # Merged prediction logic
│   ├── yield_model.py
│   └── yield_preprocessor.py
├── tests/                    # Testing Suite
├── requirements.txt          # Python dependencies
├── run_all.ps1               # Automation Script
└── README.md                 # Project Documentation
```

## Setup Instructions

### 1. Prerequisites
- **Python 3.10+**
- **Node.js** (v18+ recommended)
- **npm** or **yarn**

### 2. Backend & ML Environment Setup

Open PowerShell or your terminal and clone/navigate to the project format:

```bash
# 1. Install required python packages
pip install -r requirements.txt

# 2. To completely train models and run the automated Python tests
.\run_all.ps1

# 3. Start the FastAPI backend server
python -m uvicorn backend.main:app --reload --port 8000
```
> The API will be available at `http://localhost:8000`. You can access auto-generated documentation at `http://localhost:8000/docs`.

### 3. Frontend Setup

In a **separate terminal window**, navigate to the `frontend` folder:

```bash
# 1. Navigate to frontend directory
cd frontend

# 2. Install Node modules
npm install

# 3. Start the development server
npm run dev
```
> The UI will be available at `http://localhost:5173`.

---

## API Endpoints Overview

- `GET /health` - Check if engines and API are running.
- `GET /model/info` - Get current best model stats.
- `POST /predict/yield` - Predicts yield and returns XAI summary/chart.
- `POST /predict/fertilizer` - Recommends fertilizer + top 3 alternatives + XAI breakdown.
- `POST /predict/combined` - Runs both models concurrently for holistic advice.

## Project Structure
```text
├── datasets/                          
├── data/
│   └── processed/
├── notebooks/
├── src/
├── models/
├── outputs/
│   └── plots/
├── tests/
├── requirements.txt
└── README.md
```

## Installation
```bash
pip install -r requirements.txt
```

## How to Run Python files
Note: We have provided python scripts that execute identically to notebooks.
```bash
python notebooks/01_Yield_EDA.py
python notebooks/02_Yield_Preprocessing.py
python notebooks/03_Yield_Prediction.py
python notebooks/04_Fertilizer_EDA.py
python notebooks/05_Fertilizer_Preprocessing.py
python notebooks/06_Fertilizer_Recommendation.py
```

## How to Run Tests
```bash
python tests/test_yield_model.py
python tests/test_fertilizer_model.py
python tests/test_models.py
python tests/run_prediction.py
```
