# ============================================
# Data Preprocessing Module
# ============================================
# This module handles all data loading, cleaning,
# missing value imputation, feature scaling, and
# train/test splitting operations.
# ============================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------
def load_dataset(filepath=None):
    """
    Load the water potability dataset from CSV.
    
    Parameters:
        filepath (str): Path to the CSV file. If None, uses default path.
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    # Use default path if not provided
    if filepath is None:
        # Try multiple possible paths
        possible_paths = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset", "water_potability.csv"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "water_potability.csv"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                filepath = path
                break
    
    if filepath is None or not os.path.exists(filepath):
        raise FileNotFoundError("Dataset file not found. Please provide a valid path.")
    
    # Load CSV into DataFrame
    df = pd.read_csv(filepath)
    
    return df


# --------------------------------------------------
# 2. DATASET OVERVIEW
# --------------------------------------------------
def get_dataset_overview(df):
    """
    Get comprehensive overview of the dataset.
    
    Parameters:
        df (pd.DataFrame): Input dataset
    
    Returns:
        dict: Dictionary containing dataset statistics
    """
    overview = {
        "shape": df.shape,                          # (rows, columns)
        "columns": list(df.columns),                # Column names
        "dtypes": df.dtypes,                        # Data types
        "head": df.head(10),                        # First 10 rows
        "tail": df.tail(5),                         # Last 5 rows
        "describe": df.describe(),                  # Statistical summary
        "info": df.info,                            # DataFrame info
        "target_distribution": df["Potability"].value_counts(),  # Class balance
    }
    return overview


# --------------------------------------------------
# 3. MISSING VALUE ANALYSIS
# --------------------------------------------------
def analyze_missing_values(df):
    """
    Analyze missing values in the dataset.
    
    Parameters:
        df (pd.DataFrame): Input dataset
    
    Returns:
        pd.DataFrame: Missing value analysis with counts and percentages
    """
    # Count missing values per column
    missing_count = df.isnull().sum()
    
    # Calculate percentage of missing values
    missing_percentage = (missing_count / len(df)) * 100
    
    # Create analysis DataFrame
    missing_analysis = pd.DataFrame({
        "Missing Count": missing_count,
        "Missing Percentage (%)": missing_percentage.round(2),
        "Data Type": df.dtypes,
        "Unique Values": df.nunique()
    })
    
    # Sort by missing percentage (descending)
    missing_analysis = missing_analysis.sort_values(
        by="Missing Percentage (%)", ascending=False
    )
    
    return missing_analysis


# --------------------------------------------------
# 4. CLEAN DATASET (Handle Missing Values)
# --------------------------------------------------
def clean_dataset(df, strategy="median"):
    """
    Clean the dataset by handling missing values.
    
    Uses median imputation by default because:
    - It's robust to outliers (water quality data often has outliers)
    - It preserves the central tendency of the distribution
    
    Parameters:
        df (pd.DataFrame): Input dataset with missing values
        strategy (str): Imputation strategy - 'median', 'mean', or 'mode'
    
    Returns:
        pd.DataFrame: Cleaned dataset with no missing values
    """
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # Get numeric columns (exclude target if needed)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    
    # Apply imputation strategy
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            if strategy == "median":
                fill_value = df_clean[col].median()
            elif strategy == "mean":
                fill_value = df_clean[col].mean()
            elif strategy == "mode":
                fill_value = df_clean[col].mode()[0]
            else:
                fill_value = df_clean[col].median()  # Default to median
            
            df_clean[col] = df_clean[col].fillna(fill_value)
    
    return df_clean


# --------------------------------------------------
# 5. FEATURE SCALING
# --------------------------------------------------
def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler (z-score normalization).
    
    This is important because:
    - SVM and KNN are distance-based and sensitive to feature scale
    - Ensures all features contribute equally to the model
    
    Parameters:
        X_train (pd.DataFrame/np.array): Training features
        X_test (pd.DataFrame/np.array): Testing features
    
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    
    # Fit on training data, transform both train and test
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


# --------------------------------------------------
# 6. SPLIT DATA INTO TRAIN/TEST
# --------------------------------------------------
def split_data(df, target_column="Potability", test_size=0.2, random_state=42):
    """
    Split dataset into training and testing sets.
    
    Uses stratified splitting to maintain class distribution,
    since the dataset is slightly imbalanced (~61/39 split).
    
    Parameters:
        df (pd.DataFrame): Cleaned dataset
        target_column (str): Name of the target column
        test_size (float): Proportion of data for testing (default 20%)
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Separate features (X) and target (y)
    cols_to_drop = [target_column, "Organic_carbon", "Trihalomethanes", "Turbidity"]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df[target_column]
    
    # Stratified split ensures class proportions are maintained
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Maintain class balance in both sets
    )
    
    return X_train, X_test, y_train, y_test


# --------------------------------------------------
# 7. FULL PREPROCESSING PIPELINE
# --------------------------------------------------
def run_full_preprocessing(filepath=None):
    """
    Run the complete preprocessing pipeline:
    1. Load data
    2. Clean data (impute missing values)
    3. Split into train/test
    4. Scale features
    
    Parameters:
        filepath (str): Optional path to CSV file
    
    Returns:
        dict: Dictionary with all preprocessed data
    """
    # Step 1: Load
    df_raw = load_dataset(filepath)
    
    # Step 2: Clean
    df_clean = clean_dataset(df_raw, strategy="median")
    
    # Step 3: Split
    X_train, X_test, y_train, y_test = split_data(df_clean)
    
    # Step 4: Scale
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    return {
        "df_raw": df_raw,
        "df_clean": df_clean,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
        "feature_names": list(X_train.columns)
    }
