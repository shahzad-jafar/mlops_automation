from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import joblib
import pandas as pd
import uvicorn
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "data", "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "scaler.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "data", "preprocessed", "encoders.pkl")

# Load production-ready artifacts with error handling
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    logger.info("✓ Model, scaler, and encoders loaded successfully")
except FileNotFoundError as e:
    logger.error(f"Error: Model files not found - {e}")
    raise
except Exception as e:
    logger.error(f"Unexpected error loading models: {e}")
    raise

# Features required for prediction (must match training)
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float
    
    @field_validator('tenure')
    @classmethod
    def validate_tenure(cls, v):
        if v < 0:
            raise ValueError('tenure must be >= 0')
        return v
    
    @field_validator('MonthlyCharges', 'TotalCharges')
    @classmethod
    def validate_charges(cls, v):
        if v < 0:
            raise ValueError('charges must be >= 0')
        return v

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict if a telecom customer will churn",
    version="1.0"
)

def preprocess_input(data: CustomerData):
    try:
        # Convert to DataFrame
        df = pd.DataFrame([data.model_dump()])
        logger.debug(f"Input DataFrame shape: {df.shape}")

        # Encode categorical columns with error handling
        for col, le in encoders.items():
            if col in df.columns:
                try:
                    df[col] = le.transform(df[col])
                except ValueError as e:
                    # Handle unknown categories
                    raise ValueError(f"Invalid value for {col}. {str(e)}")

        # Scale features
        df_scaled = scaler.transform(df)
        logger.debug(f"Scaled features shape: {df_scaled.shape}")
        return df_scaled
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        raise ValueError(f"Preprocessing error: {str(e)}")

@app.post("/predict")
def predict_churn(customer: CustomerData):
    try:
        logger.info("Prediction request received")
        X = preprocess_input(customer)
        pred = model.predict(X)[0]  # 0 or 1
        result = "Yes" if pred == 1 else "No"
        logger.info(f"Prediction successful: {result}")
        return {"Churn Prediction": result, "status": "success"}
    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API is running!", "status": "healthy"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": "loaded"}

@app.get("/categories")
def get_valid_categories():
    """Get all valid categories for categorical fields"""
    try:
        categories_info = {}
        for col, le in encoders.items():
            categories_info[col] = list(le.classes_)
        return {
            "categorical_fields": categories_info,
            "message": "Use these exact values for categorical fields in /predict endpoint"
        }
    except Exception as e:
        logger.error(f"Error retrieving categories: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info(f"Starting API on http://127.0.0.1:8003")
    logger.info(f"API Docs: http://127.0.0.1:8003/docs")
    uvicorn.run(app, host="127.0.0.1", port=8003, log_level="info")