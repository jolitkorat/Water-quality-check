# ============================================
# Model Training & Evaluation Module
# ============================================
# Trains 5 ML models, compares accuracy, selects
# the best model, and provides evaluation metrics.
# ============================================

import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score
)

# Colors — monochrome professional
MODEL_COLORS = ["#212529", "#495057", "#6c757d", "#868e96", "#adb5bd"]
SAFE_COLOR = "#333333"
UNSAFE_COLOR = "#999999"

def _base_layout():
    return dict(
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(color="#212529", family="Inter, sans-serif"),
    )


# --------------------------------------------------
# 1. DEFINE ALL MODELS
# --------------------------------------------------
def get_models():
    """
    Initialize all 5 ML models with optimized hyperparameters.
    
    Returns:
        dict: Model name -> sklearn model instance
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, C=1.0
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, random_state=42, min_samples_split=5
        ),
        "SVM": SVC(
            kernel="rbf", C=1.0, gamma="scale", random_state=42, probability=True
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=7, weights="distance", n_jobs=-1
        ),
    }
    return models


# --------------------------------------------------
# 2. TRAIN ALL MODELS
# --------------------------------------------------
def train_all_models(X_train, y_train, X_test, y_test):
    """
    Train all 5 models and evaluate their performance.
    
    Parameters:
        X_train, y_train: Training data
        X_test, y_test: Testing data
    
    Returns:
        dict: Results with trained models, accuracies, and predictions
    """
    models = get_models()
    results = {}
    
    for name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        
        results[name] = {
            "model": model,
            "accuracy": round(accuracy * 100, 2),
            "f1_score": round(f1 * 100, 2),
            "precision": round(precision * 100, 2),
            "recall": round(recall * 100, 2),
            "y_pred": y_pred,
            "report": classification_report(y_test, y_pred, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }
    
    return results


# --------------------------------------------------
# 3. COMPARE MODELS (Accuracy Bar Chart)
# --------------------------------------------------
def plot_model_comparison(results):
    """
    Create an interactive bar chart comparing model accuracies.
    
    Parameters:
        results (dict): Output from train_all_models()
    
    Returns:
        plotly.graph_objects.Figure
    """
    names = list(results.keys())
    accuracies = [results[n]["accuracy"] for n in names]
    f1_scores = [results[n]["f1_score"] for n in names]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Accuracy (%)", x=names, y=accuracies,
        marker_color=MODEL_COLORS, text=accuracies,
        textposition="outside", textfont_size=13
    ))
    fig.add_trace(go.Bar(
        name="F1-Score (%)", x=names, y=f1_scores,
        marker_color=["#ced4da"]*len(names), text=f1_scores,
        textposition="outside", textfont_size=13, opacity=0.8
    ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        title_font_size=18, barmode="group",
        yaxis_title="Score (%)", yaxis_range=[0, 105],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **_base_layout(),
    )
    return fig


# --------------------------------------------------
# 4. GET BEST MODEL
# --------------------------------------------------
def get_best_model(results):
    """
    Automatically select the model with the highest accuracy.
    
    Parameters:
        results (dict): Output from train_all_models()
    
    Returns:
        tuple: (best_model_name, best_model_object, best_accuracy)
    """
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_model = results[best_name]["model"]
    best_accuracy = results[best_name]["accuracy"]
    
    return best_name, best_model, best_accuracy


# --------------------------------------------------
# 5. CONFUSION MATRIX PLOT
# --------------------------------------------------
def plot_confusion_matrix(results, model_name):
    """
    Plot an interactive confusion matrix for a specific model.
    
    Parameters:
        results (dict): Output from train_all_models()
        model_name (str): Name of the model
    
    Returns:
        plotly.graph_objects.Figure
    """
    cm = results[model_name]["confusion_matrix"]
    labels = ["Not Potable (0)", "Potable (1)"]
    
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale=[[0, "#f8f9fa"], [1, "#212529"]],
        text=cm, texttemplate="%{text}",
        textfont={"size": 20},
        hoverongaps=False,
    ))
    
    fig.update_layout(
        title=f"Confusion Matrix - {model_name}",
        title_font_size=18,
        xaxis_title="Predicted", yaxis_title="Actual",
        width=500, height=450,
        **_base_layout(),
    )
    return fig


# --------------------------------------------------
# 6. CLASSIFICATION REPORT TABLE
# --------------------------------------------------
def get_classification_report_df(results, model_name):
    """
    Convert classification report to a formatted DataFrame.
    
    Parameters:
        results (dict): Output from train_all_models()
        model_name (str): Name of the model
    
    Returns:
        pd.DataFrame: Classification report as table
    """
    report = results[model_name]["report"]
    
    # Extract per-class and overall metrics
    report_data = []
    for label in ["0", "1"]:
        if label in report:
            report_data.append({
                "Class": "Not Potable" if label == "0" else "Potable",
                "Precision": round(report[label]["precision"], 3),
                "Recall": round(report[label]["recall"], 3),
                "F1-Score": round(report[label]["f1-score"], 3),
                "Support": int(report[label]["support"]),
            })
    
    # Add overall accuracy
    report_data.append({
        "Class": "Overall (Weighted Avg)",
        "Precision": round(report["weighted avg"]["precision"], 3),
        "Recall": round(report["weighted avg"]["recall"], 3),
        "F1-Score": round(report["weighted avg"]["f1-score"], 3),
        "Support": int(report["weighted avg"]["support"]),
    })
    
    return pd.DataFrame(report_data)


# --------------------------------------------------
# 7. FEATURE IMPORTANCE PLOT
# --------------------------------------------------
def plot_feature_importance(results, model_name, feature_names):
    """
    Plot feature importance for tree-based models.
    
    Only works for Random Forest and Decision Tree.
    For other models, shows coefficient magnitudes.
    
    Parameters:
        results (dict): Output from train_all_models()
        model_name (str): Name of the model
        feature_names (list): List of feature names
    
    Returns:
        plotly.graph_objects.Figure or None
    """
    model = results[model_name]["model"]
    
    # Get feature importances
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return None
    
    # Create DataFrame and sort
    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=True)
    
    fig = px.bar(
        imp_df, x="Importance", y="Feature",
        orientation="h", color="Importance",
        color_continuous_scale="Greys",
        title=f"Feature Importance - {model_name}",
    )
    
    fig.update_layout(
        title_font_size=18, height=400,
        coloraxis_showscale=False,
        **_base_layout(),
    )
    return fig


# --------------------------------------------------
# 8. SAVE / LOAD MODEL
# --------------------------------------------------
def save_model(model, scaler, filepath="models/best_model.pkl"):
    """Save trained model and scaler using Joblib."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, filepath)
    return filepath


def load_model(filepath="models/best_model.pkl"):
    """Load trained model and scaler from Joblib file."""
    if not os.path.exists(filepath):
        return None
    data = joblib.load(filepath)
    return data["model"], data["scaler"]


# --------------------------------------------------
# 9. METRICS SUMMARY TABLE
# --------------------------------------------------
def get_metrics_summary(results):
    """
    Create a summary DataFrame of all model metrics.
    
    Parameters:
        results (dict): Output from train_all_models()
    
    Returns:
        pd.DataFrame
    """
    rows = []
    for name, res in results.items():
        rows.append({
            "Model": name,
            "Accuracy (%)": res["accuracy"],
            "F1-Score (%)": res["f1_score"],
            "Precision (%)": res["precision"],
            "Recall (%)": res["recall"],
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values("Accuracy (%)", ascending=False).reset_index(drop=True)
    return df


# --------------------------------------------------
# 10. RADAR CHART FOR MODEL COMPARISON
# --------------------------------------------------
def plot_radar_comparison(results):
    """Create a radar chart comparing all models across metrics."""
    fig = go.Figure()
    
    metrics = ["Accuracy (%)", "F1-Score (%)", "Precision (%)", "Recall (%)"]
    
    for i, (name, res) in enumerate(results.items()):
        values = [res["accuracy"], res["f1_score"], res["precision"], res["recall"]]
        values.append(values[0])  # Close the polygon
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],
            fill="toself",
            name=name,
            line_color=MODEL_COLORS[i % len(MODEL_COLORS)],
            opacity=0.7,
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100]),
            bgcolor="#ffffff",
        ),
        title="Model Performance Radar Chart",
        title_font_size=18,
        height=500,
        **_base_layout(),
    )
    return fig
