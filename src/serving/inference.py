import os
import pandas as pd
import mlflow
import json

# === 1. DYNAMIC PATH CONFIGURATION ===
# Docker ke andar ye path '/app' tak point karega
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Bundled model directory (Isolated from mlruns)
MODEL_DIR = os.path.join(
    BASE_DIR, 
    "src", "serving", "model", "m-72f7ce14b76541ceb0f9197f081daa5c", "artifacts"
)

# Feature file path (Ensure it exists in the artifacts folder)
FEATURE_FILE_PATH = os.path.join(MODEL_DIR, "feature_columns.json")

# === 2. MODEL LOADING ===
try:
    # Mlflow load_model automatically looks for MLmodel file in the directory
    model = mlflow.pyfunc.load_model(MODEL_DIR)
    print(f"✅ Model loaded successfully from: {MODEL_DIR}")
except Exception as e:
    # Docker log mein ye error boht kaam ayega agar path galat hua
    raise Exception(f"❌ Critical Error: Could not load model. Ensure folder exists. Error: {e}")

# === 3. FEATURE SCHEMA LOADING ===
try:
    with open(FEATURE_FILE_PATH, 'r') as f:
        FEATURE_COLS = json.load(f)
    print(f"✅ Loaded {len(FEATURE_COLS)} feature columns.")
except Exception as e:
    # Agar model artifacts mein JSON missing hai
    raise Exception(f"❌ Failed to load feature columns from {FEATURE_FILE_PATH}: {e}")

# === 4. PRE-PROCESSING LOGIC ===
BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms raw input dictionary into model-compatible features."""
    df = df.copy()
    
    # Clean whitespace from inputs
    df.columns = df.columns.str.strip()
    
    # 1. Handle Numeric Columns
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    
    # 2. Map Binary Categoricals
    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c].astype(str).str.strip().map(mapping)
                .astype("Int64").fillna(0).astype(int)
            )
    
    # 3. One-Hot Encode (handling unseen categories safely)
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns]
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)
    
    # 4. Final Alignment: Ensure exact same columns as training
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    
    # 5. Convert any remaining Booleans to Integers
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)
        
    return df

def predict(input_dict: dict) -> str:
    """Main interface for prediction."""
    df = pd.DataFrame([input_dict])
    df_enc = _serve_transform(df)
    
    try:
        preds = model.predict(df_enc)
        # Convert prediction to standard Python type
        if hasattr(preds, "tolist"):
            preds = preds.tolist()
        
        result = preds[0] if isinstance(preds, list) else preds
        
    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")
    
    return "Likely to churn" if result == 1 else "Not likely to churn"

# === 5. LOCAL TEST ===
if __name__ == "__main__":
    test_customer = {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
        "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85, "TotalCharges": 29.85
    }
    
    print("\n" + "="*40)
    print(f"DEBUG: Input Data Sample: {test_customer}")
    print(f"PREDICTION: {predict(test_customer)}")
    print("="*40 + "\n")