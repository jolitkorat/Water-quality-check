# ============================================
# Prediction Module
# ============================================
# Handles real-time water quality predictions
# and downloadable report generation.
# ============================================

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime


# --------------------------------------------------
# 1. PREDICT WATER QUALITY
# --------------------------------------------------
def predict_water_quality(model, scaler, input_values, feature_names):
    """
    Predict whether water is potable or not.
    
    Parameters:
        model: Trained sklearn model
        scaler: Fitted StandardScaler
        input_values (dict): Feature name -> value mapping
        feature_names (list): Ordered list of feature names
    
    Returns:
        dict: Prediction result with label, probability, and details
    """
    # Create input array in correct feature order
    input_array = np.array([[input_values[f] for f in feature_names]])
    
    # Scale the input
    input_scaled = scaler.transform(input_array)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    # Get probability if available
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_scaled)[0]
        confidence = max(probabilities) * 100
        prob_not_potable = probabilities[0] * 100
        prob_potable = probabilities[1] * 100
    else:
        confidence = None
        prob_not_potable = None
        prob_potable = None
    
    # Build result
    result = {
        "prediction": int(prediction),
        "label": "POTABLE (Safe to Drink)" if prediction == 1 else "NOT POTABLE (Unsafe)",
        "is_safe": bool(prediction == 1),
        "confidence": round(confidence, 2) if confidence else None,
        "prob_potable": round(prob_potable, 2) if prob_potable else None,
        "prob_not_potable": round(prob_not_potable, 2) if prob_not_potable else None,
        "input_values": input_values,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    return result


# --------------------------------------------------
# 2. GENERATE PREDICTION REPORT (CSV)
# --------------------------------------------------
def generate_prediction_report(result, feature_names):
    """
    Generate a downloadable prediction report as CSV string.
    
    Parameters:
        result (dict): Output from predict_water_quality()
        feature_names (list): List of feature names
    
    Returns:
        str: CSV-formatted report string
    """
    lines = []
    lines.append("=" * 50)
    lines.append("WATER QUALITY PREDICTION REPORT")
    lines.append("=" * 50)
    lines.append(f"Date: {result['timestamp']}")
    lines.append(f"Prediction: {result['label']}")
    if result['confidence']:
        lines.append(f"Confidence: {result['confidence']}%")
    lines.append("")
    lines.append("-" * 50)
    lines.append("INPUT PARAMETERS:")
    lines.append("-" * 50)
    
    for feature in feature_names:
        value = result['input_values'].get(feature, "N/A")
        lines.append(f"  {feature}: {value}")
    
    lines.append("")
    lines.append("-" * 50)
    lines.append("PROBABILITY SCORES:")
    lines.append("-" * 50)
    if result['prob_potable'] is not None:
        lines.append(f"  Potable (Safe): {result['prob_potable']}%")
        lines.append(f"  Not Potable (Unsafe): {result['prob_not_potable']}%")
    
    lines.append("")
    lines.append("=" * 50)
    lines.append("DISCLAIMER: This prediction is based on a machine")
    lines.append("learning model and should be used for reference only.")
    lines.append("Always consult certified water testing laboratories")
    lines.append("for official water quality assessments.")
    lines.append("=" * 50)
    
    return "\n".join(lines)


# --------------------------------------------------
# 3. GENERATE CSV REPORT
# --------------------------------------------------
def generate_csv_report(result, feature_names):
    """
    Generate a CSV-formatted report for download.
    
    Parameters:
        result (dict): Output from predict_water_quality()
        feature_names (list): List of feature names
    
    Returns:
        pd.DataFrame: Report as DataFrame
    """
    # Input parameters row
    data = {"Parameter": [], "Value": []}
    
    data["Parameter"].append("Prediction Date")
    data["Value"].append(result["timestamp"])
    
    data["Parameter"].append("Result")
    data["Value"].append(result["label"])
    
    if result["confidence"]:
        data["Parameter"].append("Confidence (%)")
        data["Value"].append(str(result["confidence"]))
    
    data["Parameter"].append("---")
    data["Value"].append("--- Input Features ---")
    
    for feature in feature_names:
        data["Parameter"].append(feature)
        data["Value"].append(str(result["input_values"].get(feature, "N/A")))
    
    if result["prob_potable"] is not None:
        data["Parameter"].append("---")
        data["Value"].append("--- Probabilities ---")
        data["Parameter"].append("Potable Probability (%)")
        data["Value"].append(str(result["prob_potable"]))
        data["Parameter"].append("Not Potable Probability (%)")
        data["Value"].append(str(result["prob_not_potable"]))
    
    return pd.DataFrame(data)


# --------------------------------------------------
# 4. BATCH PREDICTION
# --------------------------------------------------
def batch_predict(model, scaler, df, feature_names):
    """
    Make predictions on a batch of water samples.
    
    Parameters:
        model: Trained sklearn model
        scaler: Fitted StandardScaler
        df (pd.DataFrame): DataFrame with feature columns
        feature_names (list): Ordered feature names
    
    Returns:
        pd.DataFrame: Original data with predictions added
    """
    X = df[feature_names].values
    X_scaled = scaler.transform(X)
    
    predictions = model.predict(X_scaled)
    df_result = df.copy()
    df_result["Predicted_Potability"] = predictions
    df_result["Prediction_Label"] = df_result["Predicted_Potability"].map(
        {0: "Not Potable", 1: "Potable"}
    )
    
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)
        df_result["Confidence (%)"] = (np.max(probs, axis=1) * 100).round(2)
    
    return df_result
