import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
import re
from geopy.geocoders import Nominatim
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

def preprocess_startup_data(df):
    """
    Comprehensive preprocessing pipeline for startup credit risk analysis.
    Input: Raw DataFrame with startup information
    Output: Processed DataFrame ready for modeling
    """
    
    # Create a copy to avoid modifying the original data
    df_processed = df.copy()
    
    # Print column names for debugging
    print("Available columns in the dataset:")
    print(df_processed.columns.tolist())
    
    # 1. Basic Data Cleaning
    # ---------------------
    # Convert date columns to datetime (using flexible column names)
    date_columns = [col for col in df_processed.columns if 'date' in col.lower() or 'funding' in col.lower()]
    print("\nFound date-related columns:", date_columns)
    
    for date_col in date_columns:
        try:
            df_processed[date_col] = pd.to_datetime(df_processed[date_col], errors='coerce')
        except Exception as e:
            print(f"Warning: Could not convert {date_col} to datetime. Error: {str(e)}")
    
    # Clean numeric columns (looking for funding-related columns)
    funding_columns = [col for col in df_processed.columns if 'funding' in col.lower() or 'rounds' in col.lower()]
    print("\nFound funding-related columns:", funding_columns)
    
    for col in funding_columns:
        try:
            if df_processed[col].dtype == 'object':
                df_processed[col] = pd.to_numeric(df_processed[col].str.replace('$', '').str.replace(',', ''), errors='coerce')
        except Exception as e:
            print(f"Warning: Could not clean numeric column {col}. Error: {str(e)}")
    
    # 2. Feature Engineering
    # ---------------------
    # Time-based features (if date columns exist)
    if len(date_columns) >= 2:
        try:
            first_date = date_columns[0]
            last_date = date_columns[-1]
            df_processed['Funding_Duration_Days'] = (df_processed[last_date] - df_processed[first_date]).dt.days
        except Exception as e:
            print(f"Warning: Could not calculate funding duration. Error: {str(e)}")
    
    # Funding features (if funding columns exist)
    if len(funding_columns) >= 2:
        try:
            total_funding_col = [col for col in funding_columns if 'total' in col.lower()][0]
            rounds_col = [col for col in funding_columns if 'rounds' in col.lower()][0]
            df_processed['Average_Round_Size'] = df_processed[total_funding_col] / df_processed[rounds_col]
        except Exception as e:
            print(f"Warning: Could not calculate average round size. Error: {str(e)}")
    
    # 3. Categorical Feature Processing
    # -------------------------------
    # Handle category list (if exists)
    category_cols = [col for col in df_processed.columns if 'category' in col.lower()]
    if category_cols:
        try:
            category_col = category_cols[0]
            df_processed['Category_Count'] = df_processed[category_col].str.count(',') + 1
            
            # Create binary indicators for top categories
            top_categories = df_processed[category_col].str.get_dummies(sep=',').sum().nlargest(10).index
            for category in top_categories:
                df_processed[f'Category_{category}'] = df_processed[category_col].str.contains(category, case=False).astype(int)
        except Exception as e:
            print(f"Warning: Could not process categories. Error: {str(e)}")
    
    # 4. Location-based Features
    # ------------------------
    location_cols = [col for col in df_processed.columns if col.lower() in ['city', 'state', 'country', 'region']]
    if location_cols:
        try:
            df_processed['Location_Complete'] = df_processed[location_cols].fillna('').agg(', '.join, axis=1)
        except Exception as e:
            print(f"Warning: Could not create location features. Error: {str(e)}")
    
    # 5. Text-based Features
    # ---------------------
    url_cols = [col for col in df_processed.columns if 'url' in col.lower() or 'homepage' in col.lower()]
    if url_cols:
        try:
            url_col = url_cols[0]
            df_processed['Domain_Length'] = df_processed[url_col].str.len()
            df_processed['Has_HTTPS'] = df_processed[url_col].str.startswith('https').astype(int)
        except Exception as e:
            print(f"Warning: Could not process URL features. Error: {str(e)}")
    
    # 6. Target Variable Processing
    # ---------------------------
    status_cols = [col for col in df_processed.columns if 'status' in col.lower()]
    if status_cols:
        try:
            status_col = status_cols[0]
            df_processed['Target'] = (df_processed[status_col].str.lower() == 'failed').astype(int)
        except Exception as e:
            print(f"Warning: Could not create target variable. Error: {str(e)}")
    
    # 7. Final Feature Selection and Scaling
    # ------------------------------------
    # Select numeric features for scaling
    numeric_features = df_processed.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if numeric_features:
        try:
            scaler = StandardScaler()
            df_processed[numeric_features] = scaler.fit_transform(df_processed[numeric_features])
        except Exception as e:
            print(f"Warning: Could not scale numeric features. Error: {str(e)}")
    
    # Fill missing values
    df_processed = df_processed.fillna(df_processed.mean())
    
    return df_processed

# Example usage
if __name__ == "__main__":
    try:
        # Load the data
        df = pd.read_csv('Data/startup_failures.csv')
        print("\nDataset shape:", df.shape)
        
        # Apply preprocessing
        processed_df = preprocess_startup_data(df)
        print("\nProcessed dataset shape:", processed_df.shape)
        print("\nProcessed columns:", processed_df.columns.tolist())
        
    except Exception as e:
        print(f"Error processing the dataset: {str(e)}")

# Optional AutoML libraries to consider:
"""
1. PyCaret: Great for quick prototyping and model comparison
   from pycaret.classification import *
   setup(data=processed_df, target='Target')
   
2. FLAML: Fast and lightweight AutoML
   from flaml import AutoML
   automl = AutoML()
   automl.fit(processed_df, y='Target')
   
3. Auto-sklearn: More comprehensive but slower
   from autosklearn.classification import AutoSklearnClassifier
   automl = AutoSklearnClassifier()
   automl.fit(processed_df, y='Target')
""" 