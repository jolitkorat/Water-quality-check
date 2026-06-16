# ============================================
# Exploratory Data Analysis (EDA) Module
# ============================================
# Visualization functions for the water quality dataset.
# Uses Plotly for interactive Streamlit charts.
# Black & white professional theme.
# ============================================

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

# Color palette — monochrome professional
SAFE_COLOR = "#333333"
UNSAFE_COLOR = "#999999"
CHART_COLORS = ["#212529", "#495057", "#6c757d", "#adb5bd", "#ced4da",
                "#868e96", "#343a40", "#dee2e6", "#e9ecef"]

sns.set_style("whitegrid")


def _base_layout():
    """Common layout settings for all charts."""
    return dict(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#212529", family="Inter, sans-serif"),
    )


def plot_correlation_heatmap(df):
    """Create interactive correlation heatmap."""
    corr = df.corr(numeric_only=True)
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="Greys", zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate="%{text}",
        textfont={"size": 11, "color": "#212529"}, hoverongaps=False,
    ))
    fig.update_layout(
        title="Feature Correlation Heatmap", title_font_size=18,
        width=700, height=600, **_base_layout(),
    )
    return fig


def plot_target_distribution(df):
    """Plot Potability class distribution."""
    counts = df["Potability"].value_counts().reset_index()
    counts.columns = ["Potability", "Count"]
    counts["Label"] = counts["Potability"].map({0: "Not Potable", 1: "Potable"})
    fig = px.bar(counts, x="Label", y="Count", color="Label",
                 color_discrete_map={"Not Potable": "#868e96", "Potable": "#212529"},
                 text="Count", title="Target Variable Distribution")
    fig.update_traces(textposition="outside", textfont_size=14)
    fig.update_layout(showlegend=False, title_font_size=18,
                      xaxis_title="Water Quality", yaxis_title="Samples",
                      **_base_layout())
    return fig


def plot_feature_distributions(df):
    """Histograms for all features, colored by Potability."""
    feature_cols = [c for c in df.columns if c != "Potability"]
    n_cols, n_rows = 3, (len(feature_cols) + 2) // 3
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=feature_cols,
                        vertical_spacing=0.08, horizontal_spacing=0.08)
    for i, col in enumerate(feature_cols):
        r, c = i // n_cols + 1, i % n_cols + 1
        fig.add_trace(go.Histogram(x=df[df["Potability"]==0][col], name="Not Potable",
                                   marker_color="#adb5bd", opacity=0.7, showlegend=(i==0)), row=r, col=c)
        fig.add_trace(go.Histogram(x=df[df["Potability"]==1][col], name="Potable",
                                   marker_color="#212529", opacity=0.7, showlegend=(i==0)), row=r, col=c)
    fig.update_layout(title="Feature Distributions by Potability", title_font_size=18,
                      height=300*n_rows, barmode="overlay", **_base_layout())
    return fig


def plot_feature_boxplots(df):
    """Box plots for outlier detection, grouped by Potability."""
    feature_cols = [c for c in df.columns if c != "Potability"]
    n_cols, n_rows = 3, (len(feature_cols) + 2) // 3
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=feature_cols,
                        vertical_spacing=0.08, horizontal_spacing=0.08)
    for i, col in enumerate(feature_cols):
        r, c = i // n_cols + 1, i % n_cols + 1
        fig.add_trace(go.Box(y=df[df["Potability"]==0][col], name="Not Potable",
                             marker_color="#adb5bd", showlegend=(i==0)), row=r, col=c)
        fig.add_trace(go.Box(y=df[df["Potability"]==1][col], name="Potable",
                             marker_color="#212529", showlegend=(i==0)), row=r, col=c)
    fig.update_layout(title="Feature Box Plots by Potability", title_font_size=18,
                      height=350*n_rows, **_base_layout())
    return fig


def plot_pairwise_scatter(df, features=None):
    """Scatter matrix for feature pairs."""
    if features is None:
        features = ["ph", "Hardness", "Solids", "Chloramines"]
    df_plot = df.copy()
    df_plot["Quality"] = df_plot["Potability"].map({0: "Not Potable", 1: "Potable"})
    fig = px.scatter_matrix(df_plot, dimensions=features, color="Quality",
                            color_discrete_map={"Not Potable": "#adb5bd", "Potable": "#212529"},
                            title="Pairwise Feature Relationships", opacity=0.5)
    fig.update_layout(height=700, title_font_size=18, **_base_layout())
    fig.update_traces(diagonal_visible=False)
    return fig


def get_feature_statistics(df):
    """Descriptive statistics with skewness and kurtosis."""
    stats = df.describe().T
    stats["skewness"] = df.skew(numeric_only=True)
    stats["kurtosis"] = df.kurtosis(numeric_only=True)
    return stats.round(3)
