import pandas as pd

def preprocess_data(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Clean and preprocess Telco churn dataset.

    Steps:
    - Remove duplicates
    - Trim column names
    - Drop ID columns
    - Encode target column (Yes/No → 1/0)
    - Fix numeric columns
    - Handle missing values
    """
    #Remove duplicate rows
    df = df.drop_duplicates()

    #Clean column names
    df.columns = df.columns.str.strip()

    #Drop ID columns
    id_cols = ["customerID", "CustomerID", "customer_id"]
    df = df.drop(columns=[col for col in id_cols if col in df.columns])

    #Encode target column
    if target_col in df.columns and df[target_col].dtype == "object":
        df[target_col] = df[target_col].str.strip().map({"No": 0, "Yes": 1})

    #Fix TotalCharges
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    #Fix SeniorCitizen
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0).astype(int)

    #Handle missing numeric values
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)

    return df