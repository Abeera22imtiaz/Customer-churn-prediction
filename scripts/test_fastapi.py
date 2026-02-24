import requests

url = "http://127.0.0.1:8000/predict"

sample_data = {
    "gender": "Male",
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "tenure": 5,
    "MonthlyCharges": 70.35,
    "TotalCharges": 350.75
}

response = requests.post(url, json=sample_data)

print("Status Code:", response.status_code)
print("Response:", response.json())