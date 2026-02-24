import pandas as pd
import os

def load_data(file_path: str) -> pd.DataFrame:
    """
    
    Load a CSV file and return it as a pandas DataFrame.

Parameters:
    file_path (str): The full path to the CSV file.

Returns:
    pd.DataFrame: A DataFrame containing the loaded data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return pd.read_csv(file_path)

