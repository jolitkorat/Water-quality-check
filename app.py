# ============================================
# Water Quality Prediction System
# Single-Page Streamlit Application
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(__file__))

from src.data_preprocessing import (
    load_dataset, analyze_missing_values, clean_dataset,
    split_data, scale_features, get_dataset_overview
)
from src.eda import (
    plot_correlation_heatmap, plot_target_distribution,
    plot_feature_distributions, plot_feature_boxplots,
    plot_pairwise_scatter, get_feature_statistics
)
from src.model_training import (
    train_all_models, plot_model_comparison, get_best_model,
    plot_confusion_matrix, get_classification_report_df,
    plot_feature_importance, save_model, get_metrics_summary,
    plot_radar_comparison
)
from src.prediction import (
    predict_water_quality, generate_prediction_report
)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Water Quality Predictor",
    page_icon="W",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
/* Animated Vibrant Background */
.stApp {
    background: linear-gradient(-45deg, #ff9a9e, #fecfef, #a1c4fd, #c2e9fb);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
}
@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Hide sidebar, header, footer */
[data-testid="stSidebar"], [data-testid="collapsedControl"], #MainMenu, footer, header { display: none !important; visibility: hidden !important; }

/* Page title with Gradient Text */
.page-title {
    font-size: 2.8rem; font-weight: 900;
    background: linear-gradient(45deg, #ff0844 0%, #ffb199 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center; margin-bottom: 0.2rem; padding-top: 1rem;
    letter-spacing: -1px;
}
.page-sub {
    text-align: center; color: #495057; font-size: 1.1rem;
    margin-bottom: 2rem; font-weight: 500;
}

/* Glassmorphism Metric Cards */
.metric-card {
    background: rgba(255, 255, 255, 0.4);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.6);
    border-radius: 16px;
    padding: 20px; text-align: center;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.metric-card:hover { 
    transform: translateY(-5px);
    box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.2); 
}
.metric-value { font-size: 2.2rem; font-weight: 900; color: #212529; }
.metric-label {
    font-size: 0.85rem; color: #495057; margin-top: 4px;
    text-transform: uppercase; letter-spacing: 2px; font-weight: 700;
}

/* Section Headers */
.section-header {
    font-size: 1.6rem; font-weight: 800; color: #212529;
    border-left: 6px solid #ff0844; padding-left: 14px;
    margin: 2.5rem 0 1.5rem;
    background: linear-gradient(90deg, rgba(255,255,255,0.6) 0%, rgba(255,255,255,0) 100%);
    padding: 10px 14px;
    border-radius: 0 8px 8px 0;
}

/* Result boxes */
.result-safe {
    background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
    border: none; border-radius: 16px; padding: 30px; text-align: center;
    box-shadow: 0 10px 25px rgba(132, 250, 176, 0.4);
}
.result-unsafe {
    background: linear-gradient(135deg, #ff0844 0%, #ffb199 100%);
    border: none; border-radius: 16px; padding: 30px; text-align: center;
    box-shadow: 0 10px 25px rgba(255, 8, 68, 0.3);
}
.result-label { font-size: 0.9rem; font-weight: 800; letter-spacing: 3px; text-transform: uppercase; color: #ffffff; }
.result-title { font-size: 2rem; font-weight: 900; margin-top: 8px; color: #ffffff; text-shadow: 1px 1px 2px rgba(0,0,0,0.1); }
.result-conf { color: rgba(255,255,255,0.9); margin-top: 8px; font-size: 1rem; font-weight: 600; }

/* Info Card */
.info-card {
    background: rgba(255, 255, 255, 0.5);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.6);
    border-radius: 16px; padding: 25px;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
}

/* Vibrant Button */
.stButton > button {
    background: linear-gradient(45deg, #4facfe 0%, #00f2fe 100%) !important;
    color: #ffffff !important; border: none !important; border-radius: 50px !important;
    padding: 16px 32px !important; font-weight: 800 !important;
    font-size: 1.2rem !important; transition: all 0.3s ease !important;
    width: 100%; letter-spacing: 1px;
    box-shadow: 0 4px 15px rgba(79, 172, 254, 0.4) !important;
}
.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 25px rgba(79, 172, 254, 0.6) !important;
    background: linear-gradient(45deg, #00f2fe 0%, #4facfe 100%) !important;
}

/* Divider */
.section-divider {
    border: none; border-top: 2px dashed rgba(33, 37, 41, 0.2); margin: 3rem 0;
}

/* Input Labels */
div[data-testid="stWidgetLabel"] p, .stNumberInput label p, .stNumberInput label span {
    color: #212529 !important; font-weight: 800 !important; font-size: 1rem !important;
}

/* Custom Tooltip Icon */
.stNumberInput label svg, div[data-testid*="Tooltip"] svg, div[data-testid*="WidgetLabel"] svg {
    fill: transparent !important; stroke: transparent !important; color: transparent !important;
    width: 20px !important; height: 20px !important;
    background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><path fill="%23ff0844" d="M256 512A256 256 0 1 0 256 0a256 256 0 1 0 0 512zM216 336h24V272H216c-13.3 0-24-10.7-24-24s10.7-24 24-24h48c13.3 0 24 10.7 24 24v88h8c13.3 0 24 10.7 24 24s-10.7 24-24 24H216c-13.3 0-24-10.7-24-24s10.7-24 24-24zm40-208a32 32 0 1 1 0 64 32 32 0 1 1 0-64z"/></svg>') !important;
    background-size: contain !important; background-repeat: no-repeat !important; background-position: center !important;
    margin-left: 6px !important; filter: drop-shadow(0 2px 4px rgba(255,8,68,0.3));
    transition: transform 0.2s ease !important;
}
.stNumberInput label svg:hover, div[data-testid*="Tooltip"] svg:hover, div[data-testid*="WidgetLabel"] svg:hover {
    transform: scale(1.2) !important;
}
.stNumberInput label svg path, div[data-testid*="Tooltip"] svg path, div[data-testid*="WidgetLabel"] svg path { display: none !important; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD DATA & TRAIN MODEL (cached)
# --------------------------------------------------
@st.cache_data
def get_data():
    df_raw = load_dataset()
    df_clean = clean_dataset(df_raw)
    return df_raw, df_clean

@st.cache_resource
def train_models_cached():
    # Cache buster to force retraining after dropping columns
    df_clean = clean_dataset(load_dataset())
    X_train, X_test, y_train, y_test = split_data(df_clean)
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)
    results = train_all_models(X_train_s, y_train, X_test_s, y_test)
    feature_names = list(X_train.columns)
    best_name, best_model, best_acc = get_best_model(results)
    model_path = os.path.join(os.path.dirname(__file__), "models", "best_model.pkl")
    save_model(best_model, scaler, model_path)
    return results, scaler, feature_names, best_name, best_model, best_acc

# Load everything on startup
df_raw, df_clean = get_data()

with st.spinner("Loading ML models..."):
    results, scaler, feature_names, best_name, best_model, best_acc = train_models_cached()


# ==================================================
# HEADER
# ==================================================
st.markdown('<p class="page-title">Water Quality Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="page-sub">Enter water sample parameters below to check if the water is safe for drinking</p>', unsafe_allow_html=True)

# Quick stats row
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{df_raw.shape[0]:,}</div><div class="metric-label">Samples in Dataset</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{len(feature_names)}</div><div class="metric-label">Input Features</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{best_acc}%</div><div class="metric-label">Model Accuracy</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{best_name}</div><div class="metric-label">Best Model</div></div>', unsafe_allow_html=True)


# ==================================================
# SECTION 1: INPUT FORM + PREDICTION
# ==================================================
st.markdown('<div class="section-header">Enter Water Sample Parameters</div>', unsafe_allow_html=True)

ranges = {
    "ph": (0.0, 14.0, 0.0, "pH: Acid/base balance. Safe range is typically 6.5 to 8.5 for drinking water."),
    "Hardness": (0.0, 350.0, 0.0, "Hardness: Calcium and magnesium salts in mg/L. High hardness can cause scale buildup."),
    "Solids": (0.0, 60000.0, 0.0, "Solids (TDS): Total Dissolved Solids in ppm. Desirable limit is usually around 500-1000 mg/L."),
    "Chloramines": (0.0, 14.0, 0.0, "Chloramines: Disinfectants (chlorine + ammonia). Safe levels are typically up to 4 mg/L."),
    "Sulfate": (0.0, 500.0, 0.0, "Sulfate: Natural mineral concentration in mg/L. High levels (>250 mg/L) can cause a bitter taste."),
    "Conductivity": (0.0, 800.0, 0.0, "Conductivity: Electrical conductivity in μS/cm. Relates to total dissolved salts. WHO limit is often 400 μS/cm."),
}

# Define WHO scientific limits for override
WHO_LIMITS = {
    "ph": (6.5, 8.5, "pH must be between 6.5 and 8.5"),
    "Hardness": (0.0, 300.0, "Hardness should be below 300 mg/L"),
    "Solids": (0.0, 1000.0, "Solids (TDS) should be below 1000 ppm"),
    "Chloramines": (0.0, 4.0, "Chloramines should be below 4.0 ppm"),
    "Sulfate": (0.0, 250.0, "Sulfate should be below 250 mg/L"),
    "Conductivity": (0.0, 400.0, "Conductivity should be below 400 μS/cm"),
}

col1, col2, col3 = st.columns(3)
input_values = {}

for i, feat in enumerate(feature_names):
    mn, mx, default, desc = ranges.get(feat, (0.0, 100.0, 50.0, ""))
    target_col = [col1, col2, col3][i % 3]
    with target_col:
        input_values[feat] = st.number_input(
            f"{feat}", min_value=mn, max_value=mx, value=default,
            step=0.1, key=f"input_{feat}", help=desc
        )

st.markdown("")

if st.button("Predict Water Quality", use_container_width=True):
    result = predict_water_quality(best_model, scaler, input_values, feature_names)
    
    # Apply Scientific Rule-Based Override
    failed_rules = []
    for feat in feature_names:
        if feat in WHO_LIMITS:
            min_val, max_val, msg = WHO_LIMITS[feat]
            val = input_values[feat]
            if not (min_val <= val <= max_val):
                failed_rules.append(f"**{feat} ({val})**: {msg}")
    
    if len(failed_rules) == 0:
        result["prediction"] = 1
        result["label"] = "POTABLE (Safe to Drink)"
        result["is_safe"] = True
        result["override"] = True
        result["override_msg"] = "All entered parameters fall strictly within safe scientific (WHO) limits!"
    else:
        result["prediction"] = 0
        result["label"] = "NOT POTABLE (Unsafe to Drink)"
        result["is_safe"] = False
        result["override"] = True
        result["override_msg"] = "The water failed the following strict scientific safety checks:\\n\\n" + "\\n".join(["- " + r for r in failed_rules])
        
    st.session_state["last_result"] = result

# --------------------------------------------------
# SHOW PREDICTION RESULT
# --------------------------------------------------
if "last_result" in st.session_state:
    result = st.session_state["last_result"]
    
    st.markdown("")
    
    if result["is_safe"]:
        conf_text = "🔬 Scientific Override Applied" if result.get("override") else "Model Confidence: " + str(result.get("confidence", "")) + "%"
        st.markdown(
            '<div class="result-safe">'
            '<div class="result-label">Result</div>'
            '<div class="result-title">WATER IS POTABLE — Safe to Drink</div>'
            f'<div class="result-conf">{conf_text}</div>'
            '</div>', unsafe_allow_html=True)
        if result.get("override"):
            st.success(result["override_msg"])
    else:
        conf_text = "🔬 Scientific Override Applied" if result.get("override") else "Model Confidence: " + str(result.get("confidence", "")) + "%"
        st.markdown(
            '<div class="result-unsafe">'
            '<div class="result-label">Result</div>'
            '<div class="result-title">WATER IS NOT POTABLE — Unsafe to Drink</div>'
            f'<div class="result-conf">{conf_text}</div>'
            '</div>', unsafe_allow_html=True)
        if result.get("override"):
            st.error(result["override_msg"])
    
    # Probability breakdown
    st.markdown("")
    if result["prob_potable"] is not None:
        p1, p2 = st.columns(2)
        p1.metric("Potable Probability", f"{result['prob_potable']}%")
        p2.metric("Not Potable Probability", f"{result['prob_not_potable']}%")
    
    # Download report
    report_text = generate_prediction_report(result, feature_names)
    st.download_button(
        "Download Prediction Report",
        data=report_text, file_name="water_quality_report.txt",
        mime="text/plain", use_container_width=True,
    )

    # --------------------------------------------------
    # DETAILED ANALYSIS (shown after prediction)
    # --------------------------------------------------
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Detailed Analysis</div>', unsafe_allow_html=True)
    
    # --- Input vs Dataset Comparison ---
    st.markdown("**Your Input vs Dataset Averages**")
    comparison_data = []
    for feat in feature_names:
        avg_val = df_clean[feat].mean()
        user_val = result["input_values"][feat]
        diff_pct = ((user_val - avg_val) / avg_val * 100) if avg_val != 0 else 0
        comparison_data.append({
            "Parameter": feat,
            "Your Value": round(user_val, 2),
            "Dataset Average": round(avg_val, 2),
            "Difference (%)": round(diff_pct, 1),
        })
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
    
    # --- Model Performance ---
    st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Accuracy Comparison", "Confusion Matrix", "Classification Report", "Feature Importance"])
    
    with tab1:
        st.plotly_chart(plot_model_comparison(results), use_container_width=True)
        st.dataframe(get_metrics_summary(results).style.background_gradient(
            cmap="Greys", subset=["Accuracy (%)"]), use_container_width=True)
    
    with tab2:
        st.plotly_chart(plot_confusion_matrix(results, best_name), use_container_width=True)
    
    with tab3:
        st.dataframe(get_classification_report_df(results, best_name), use_container_width=True)
    
    with tab4:
        fig = plot_feature_importance(results, best_name, feature_names)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")
    
    # --- EDA Section ---
    st.markdown('<div class="section-header">Dataset Exploration</div>', unsafe_allow_html=True)
    
    tab_a, tab_b, tab_c, tab_d = st.tabs(
        ["Target Distribution", "Correlation Heatmap", "Feature Distributions", "Box Plots"])
    
    with tab_a:
        st.plotly_chart(plot_target_distribution(df_clean), use_container_width=True)
    
    with tab_b:
        st.plotly_chart(plot_correlation_heatmap(df_clean), use_container_width=True)
    
    with tab_c:
        st.plotly_chart(plot_feature_distributions(df_clean), use_container_width=True)
    
    with tab_d:
        st.plotly_chart(plot_feature_boxplots(df_clean), use_container_width=True)
    

