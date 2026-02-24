import sys
import os
from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr

# --- 1. PATH CONFIGURATION ---
# Ensure project root is in path before importing local modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.serving.inference import predict 

# --- 2. FASTAPI SETUP ---
app = FastAPI(title="Telco Churn API + Gradio", version="1.0")

class CustomerData(BaseModel):
    gender: str
    Partner: str
    Dependents: str
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
    tenure: int
    MonthlyCharges: float
    TotalCharges: float

@app.post("/predict")
def api_predict(data: CustomerData):
    # model_dump() is preferred in newer Pydantic versions
    return {"prediction": predict(data.model_dump())}

@app.get("/")
def root():
    return {"status": "ok"}

# --- 3. GRADIO LOGIC ---
def gradio_interface(
    gender, Partner, Dependents, PhoneService, MultipleLines,
    InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
    TechSupport, StreamingTV, StreamingMovies, Contract,
    PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges
):
    data = {
        "gender": gender, "Partner": Partner, "Dependents": Dependents,
        "PhoneService": PhoneService, "MultipleLines": MultipleLines,
        "InternetService": InternetService, "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup, "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport, "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies, "Contract": Contract,
        "PaperlessBilling": PaperlessBilling, "PaymentMethod": PaymentMethod,
        "tenure": int(tenure),
        "MonthlyCharges": float(MonthlyCharges),
        "TotalCharges": float(TotalCharges),
    }
    return predict(data)

# --- 4. GRADIO INTERFACE (The missing 'demo' part) ---
demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Dropdown(["Male","Female"], label="Gender"),
        gr.Dropdown(["Yes","No"], label="Partner"),
        gr.Dropdown(["Yes","No"], label="Dependents"),
        gr.Dropdown(["Yes","No"], label="Phone Service"),
        gr.Dropdown(["Yes","No","No phone service"], label="Multiple Lines"),
        gr.Dropdown(["DSL","Fiber optic","No"], label="Internet Service"),
        gr.Dropdown(["Yes","No","No internet service"], label="Online Security"),
        gr.Dropdown(["Yes","No","No internet service"], label="Online Backup"),
        gr.Dropdown(["Yes","No","No internet service"], label="Device Protection"),
        gr.Dropdown(["Yes","No","No internet service"], label="Tech Support"),
        gr.Dropdown(["Yes","No","No internet service"], label="Streaming TV"),
        gr.Dropdown(["Yes","No","No internet service"], label="Streaming Movies"),
        gr.Dropdown(["Month-to-month","One year","Two year"], label="Contract"),
        gr.Dropdown(["Yes","No"], label="Paperless Billing"),
        gr.Dropdown([
            "Electronic check","Mailed check",
            "Bank transfer (automatic)","Credit card (automatic)"
        ], label="Payment Method"),
        gr.Number(label="Tenure"),
        gr.Number(label="Monthly Charges"),
        gr.Number(label="Total Charges"),
    ],
    outputs="text",
    title="Telco Churn Predictor"
)

# --- 5. MOUNT GRADIO ---
app = gr.mount_gradio_app(app, demo, path="/ui")