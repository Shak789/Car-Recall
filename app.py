import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="NHTSA Recall Risk Predictor",
    page_icon="🚗",
    layout="wide"
)

# ── load data and models ──────────────────────────────────────────
@st.cache_data
def load_data():
    final_df = pd.read_csv("result.csv")
    raw_df = pd.read_csv("complaints_raw.csv")
    raw_df["FAILDATE"] = pd.to_datetime(raw_df["FAILDATE"], errors="coerce")
    return final_df, raw_df

@st.cache_resource
def load_model():
    xgb = joblib.load("xgb_model.pkl")
    explainer = shap.TreeExplainer(xgb)
    return xgb, explainer

final_df, raw_df = load_data()
xgb, explainer = load_model()

FEATURES = [
    "complaints_per_day",
    "crash_ratio",
    "injured_ratio",
    "keybert_safety_score",
    "median_mileage",
]


CAR_LOGOS = {
    "ACURA": "https://www.carlogos.org/car-logos/acura-logo.png",
    "AUDI": "https://www.carlogos.org/car-logos/audi-logo.png",
    "BMW": "https://www.carlogos.org/car-logos/bmw-logo.png",
    "BUICK": "https://www.carlogos.org/car-logos/buick-logo.png",
    "CADILLAC": "https://www.carlogos.org/car-logos/cadillac-logo.png",
    "CHEVROLET": "https://www.carlogos.org/car-logos/chevrolet-logo.png",
    "CHRYSLER": "https://www.carlogos.org/car-logos/chrysler-logo.png",
    "DODGE": "https://www.carlogos.org/car-logos/dodge-logo.png",
    "FERRARI": "https://www.carlogos.org/car-logos/ferrari-logo.png",
    "FIAT": "https://www.carlogos.org/car-logos/fiat-logo.png",
    "FORD": "https://www.carlogos.org/car-logos/ford-logo.png",
    "GMC": "https://www.carlogos.org/car-logos/gmc-logo.png",
    "HONDA": "https://www.carlogos.org/car-logos/honda-logo.png",
    "HYUNDAI": "https://www.carlogos.org/car-logos/hyundai-logo.png",
    "INFINITI": "https://www.carlogos.org/car-logos/infiniti-logo.png",
    "JAGUAR": "https://www.carlogos.org/car-logos/jaguar-logo.png",
    "JEEP": "https://www.carlogos.org/car-logos/jeep-logo.png",
    "KIA": "https://www.carlogos.org/car-logos/kia-logo.png",
    "LAND ROVER": "https://www.carlogos.org/car-logos/land-rover-logo.png",
    "LEXUS": "https://www.carlogos.org/car-logos/lexus-logo.png",
    "LINCOLN": "https://www.carlogos.org/car-logos/lincoln-logo.png",
    "MASERATI": "https://www.carlogos.org/car-logos/maserati-logo.png",
    "MAZDA": "https://www.carlogos.org/car-logos/mazda-logo.png",
    "MERCEDES-BENZ": "https://www.carlogos.org/car-logos/mercedes-benz-logo.png",
    "MINI": "https://www.carlogos.org/car-logos/mini-logo.png",
    "MITSUBISHI": "https://www.carlogos.org/car-logos/mitsubishi-logo.png",
    "NISSAN": "https://www.carlogos.org/car-logos/nissan-logo.png",
    "PONTIAC": "https://www.carlogos.org/car-logos/pontiac-logo.png",
    "PORSCHE": "https://www.carlogos.org/car-logos/porsche-logo.png",
    "RAM": "https://www.carlogos.org/car-logos/ram-logo.png",
    "ROLLS-ROYCE": "https://www.carlogos.org/car-logos/rolls-royce-logo.png",
    "SAAB": "https://www.carlogos.org/car-logos/saab-logo.png",
    "SUBARU": "https://www.carlogos.org/car-logos/subaru-logo.png",
    "TESLA": "https://www.carlogos.org/car-logos/tesla-logo.png",
    "TOYOTA": "https://www.carlogos.org/car-logos/toyota-logo.png",
    "VOLKSWAGEN": "https://www.carlogos.org/car-logos/volkswagen-logo.png",
    "VOLVO": "https://www.carlogos.org/car-logos/volvo-logo.png",
}

# ── styling ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .metric-card {
        background: #1c1f26;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2d3139;
    }
    .metric-label {
        color: #8b8fa8;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: white;
    }
    .risk-high { color: #ff4b4b; }
    .risk-medium { color: #ffa500; }
    .risk-low { color: #00cc88; }
    .section-header {
        font-size: 16px;
        font-weight: 600;
        color: #8b8fa8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid #2d3139;
    }
</style>
""", unsafe_allow_html=True)

# ── header ────────────────────────────────────────────────────────
st.title("🚗 NHTSA Vehicle Recall Risk Predictor")
st.markdown("Predict the likelihood of a vehicle recall based on consumer complaint patterns and NLP analysis of complaint descriptions.")

st.divider()

# ── dropdowns ─────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    makes = sorted(final_df["MAKE"].unique())
    selected_make = st.selectbox("Make", makes)

with col2:
    models = sorted(final_df[final_df["MAKE"] == selected_make]["MODEL"].unique())
    selected_model = st.selectbox("Model", models)

with col3:
    years = sorted(final_df[
        (final_df["MAKE"] == selected_make) &
        (final_df["MODEL"] == selected_model)
    ]["YEAR"].unique(), reverse=True)
    selected_year = st.selectbox("Year", years)

# ── get vehicle data ──────────────────────────────────────────────
vehicle = final_df[
    (final_df["MAKE"] == selected_make) &
    (final_df["MODEL"] == selected_model) &
    (final_df["YEAR"] == selected_year)
]

raw_vehicle = raw_df[
    (raw_df["MAKE"] == selected_make) &
    (raw_df["MODEL"] == selected_model) &
    (raw_df["YEAR"] == selected_year)
]

if vehicle.empty:
    st.warning("No data found for this vehicle combination.")
    st.stop()

# ── prediction ────────────────────────────────────────────────────
#X_vehicle = vehicle[FEATURES]
probability = vehicle["Probability_Recall"].values[0]
prediction = vehicle["Predict_Recall"].values[0]
#shap_values = explainer(X_vehicle)

if "Recall" in vehicle.columns:
    actual_recall = int(vehicle["Recall"].values[0])
else:
    actual_recall = None

# ── top metrics row ───────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
logo_col, prob_col, pred_col, recall_col, complaints_col = st.columns([1, 1.5, 1.5, 1.5, 1.5])

with logo_col:
    logo_url = CAR_LOGOS.get(selected_make.upper())
    if logo_url:
        st.image(logo_url, width=80)
    else:
        st.markdown(f"### {selected_make}")

with prob_col:
    if probability >= 0.7:
        risk_class = "risk-high"
        risk_label = "HIGH RISK"
    elif probability >= 0.4:
        risk_class = "risk-medium"
        risk_label = "MEDIUM RISK"
    else:
        risk_class = "risk-low"
        risk_label = "LOW RISK"

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Recall Probability</div>
        <div class="metric-value {risk_class}">{probability:.1%}</div>
        <div style="color: #8b8fa8; font-size: 12px; margin-top: 4px;">{risk_label}</div>
    </div>
    """, unsafe_allow_html=True)

with pred_col:
    pred_text = "RECALL" if prediction == 1 else "NO RECALL"
    pred_color = "#ff4b4b" if prediction == 1 else "#00cc88"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Model Prediction</div>
        <div class="metric-value" style="color: {pred_color};">{pred_text}</div>
    </div>
    """, unsafe_allow_html=True)

with recall_col:
    if actual_recall is not None:
        recall_text = "YES" if actual_recall == 1 else "NO"
        recall_color = "#ff4b4b" if actual_recall == 1 else "#00cc88"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Actually Recalled</div>
            <div class="metric-value" style="color: {recall_color};">{recall_text}</div>
        </div>
        """, unsafe_allow_html=True)

with complaints_col:
    complaint_count = int(vehicle["complaint_count"].values[0])
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Total Complaints</div>
        <div class="metric-value">{complaint_count:,}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── shap + complaints over time ───────────────────────────────────
col_shap, col_time = st.columns(2)


with col_shap:
    st.markdown('<div class="section-header">What drives this prediction</div>', unsafe_allow_html=True)

    shap_cols = [c for c in vehicle.columns if c.startswith('shap_')]
    shap_row = vehicle[shap_cols].iloc[0]
    feature_impacts = shap_row.abs().sort_values(ascending=False)
    feature_impacts.index = feature_impacts.index.str.replace('shap_', '')
    
    feature_names = {
        'complaints_per_day': 'Complaints Per Day',
        'crash_ratio': 'Crash Ratio',
        'injured_ratio': 'Injured Ratio',
        'keybert_safety_score': 'Complaint Description',
        'median_mileage': 'Median Mileage Before Failure'
    }
    feature_impacts.index = feature_impacts.index.map(lambda x: feature_names.get(x, x))
    
    fig_bar = px.bar(x=feature_impacts.values, y=feature_impacts.index, 
                    orientation='h')
    
    feature_impacts_pct = (feature_impacts / feature_impacts.sum()) * 100
    fig_bar = px.bar(x=feature_impacts_pct.values, y=feature_impacts_pct.index, 
                orientation='h')
    fig_bar.update_traces(
        marker_color='#4A90E2',
        hovertemplate='<b>%{y}</b><br>Impact: %{x:.2f}%<extra></extra>'
    )
    fig_bar.update_layout(
        height=250,
        margin=dict(t=10, b=10, l=10, r=10),
        showlegend=False,
        xaxis_title="Relative Importance (%)",
        yaxis_title="Metric",
    )
    st.plotly_chart(fig_bar)

with col_time:
    st.markdown('<div class="section-header">Complaints over time</div>', unsafe_allow_html=True)

    if not raw_vehicle.empty and "FAILDATE" in raw_vehicle.columns:
        monthly = (
            raw_vehicle.dropna(subset=["FAILDATE"])
            .groupby(raw_vehicle["FAILDATE"].dt.to_period("M"))
            .size()
            .reset_index(name="count")
        )
        monthly["FAILDATE"] = monthly["FAILDATE"].astype(str)

        fig_time = px.line(
            monthly,
            x="FAILDATE",
            y="count",
            labels={"FAILDATE": "Month", "count": "Complaints"},
        )
        fig_time.update_traces(line_color="#4b9fff", line_width=2)
        fig_time.update_layout(
            plot_bgcolor="#1c1f26",
            paper_bgcolor="#1c1f26",
            font_color="white",
            height=350,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor="#2d3139", tickangle=45),
            yaxis=dict(gridcolor="#2d3139"),
        )
        st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.info("No complaint timeline data available for this vehicle.")

# ── map ───────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Complaint distribution by state</div>', unsafe_allow_html=True)

if not raw_vehicle.empty and "STATE" in raw_vehicle.columns:
    state_counts = (
        raw_vehicle.dropna(subset=["STATE"])
        .groupby("STATE")
        .size()
        .reset_index(name="complaint_count")
    )

    fig_map = px.choropleth(
        state_counts,
        locations="STATE",
        locationmode="USA-states",
        color="complaint_count",
        scope="usa",
        color_continuous_scale="Reds",
        labels={"complaint_count": "Complaints"},
    )
    fig_map.update_layout(
        paper_bgcolor="#1c1f26",
        geo_bgcolor="#1c1f26",
        font_color="white",
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(tickfont=dict(color="white")),
    )
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.info("No state data available for this vehicle.")

# ── footer ────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='color: #8b8fa8; font-size: 12px; text-align: center;'>"
    "Data sourced from NHTSA complaints and recalls database. "
    "Model trained on 2010–2020 vehicles. "
    "Predictions are probabilistic and should not be used as definitive safety assessments."
    "</div>",
    unsafe_allow_html=True
)