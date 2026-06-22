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
    page_title="Vehicle Recall Risk Predictor",
    layout="wide"
)

# ── load data and models ──────────────────────────────────────────
@st.cache_data
def load_data():
    final_df = pd.read_csv("result.csv")
    raw_df = pd.read_csv("complaints_raw.csv")
    raw_df["FAILDATE"] = pd.to_datetime(raw_df["FAILDATE"], errors="coerce")
    return final_df, raw_df


#@st.cache_resource
#def load_model():
    lr = joblib.load("lr_model.pkl")
    explainer = shap.LinearExplainer(lr)
    return lr, explainer


final_df, raw_df = load_data()
#lr, explainer = load_model()

FEATURES = [
    "complaints_per_day",
    "crash_ratio",
    "injured_ratio",
    "keybert_safety_score",
    "median_mileage",
]


CAR_LOGOS = {
    "ACURA": "https://www.carlogos.org/car-logos/acura-logo.png",
    "AUDI": "https://www.carlogos.org/car-logos/audi-logo-2009-download.png",
    "BMW": "https://www.carlogos.org/car-logos/bmw-logo-2020-gray-download.png",
    "BUICK": "https://www.carlogos.org/car-logos/buick-logo.png",
    "CADILLAC": "https://www.supercars.net/blog/wp-content/uploads/2019/12/Cadillac-Logo-Big.png",
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
    "JAGUAR": "https://www.carlogos.org/car-logos/jaguar-logo-2012-download.png",
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
    "NISSAN": "https://www.carlogos.org/car-logos/nissan-logo-2013-700x700.png",
    "PONTIAC": "https://www.carlogos.org/car-logos/pontiac-logo.png",
    "PORSCHE": "https://www.carlogos.org/car-logos/porsche-logo.png",
    "RAM": "https://www.carlogos.org/car-logos/ram-logo.png",
    "ROLLS-ROYCE": "https://www.carlogos.org/car-logos/rolls-royce-logo.png",
    "SAAB": "https://www.carlogos.org/car-logos/saab-logo.png",
    "SUBARU": "https://www.carlogos.org/car-logos/subaru-logo.png",
    "TESLA": "https://www.carlogos.org/car-logos/tesla-logo.png",
    "TOYOTA": "https://www.carlogos.org/car-logos/toyota-logo-2005-download.png",
    "VOLKSWAGEN": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Volkswagen_-_Logo.svg/3840px-Volkswagen_-_Logo.svg.png",
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


import plotly.graph_objects as go
import streamlit as st

def render_risk_gauge(risk_score, title="Recall Risk Score"):
    """
    risk_score: float between 0-100
    """
    # determine color based on score
    if risk_score < 40:
        bar_color = "#2ECC71"  # green
    elif risk_score < 80:
        bar_color = "#F1C40F"  # yellow
    else:
        bar_color = "#E74C3C"  # red

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20, 'color': 'white'}},
        number={'font': {'size': 36, 'color': 'white'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': bar_color, 'thickness': 0.3},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 1,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': 'rgba(46, 204, 113, 0.15)'},
                {'range': [40, 80], 'color': 'rgba(241, 196, 15, 0.15)'},
                {'range': [80, 100], 'color': 'rgba(231, 76, 60, 0.15)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 3},
                'thickness': 0.75,
                'value': risk_score
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "white"},
        height=320,
        margin=dict(l=20, r=20, t=0, b=20)
    )
    with st.container(key="gauge_container"):
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <style>
    div.st-key-gauge_container {
        margin-top: -40px;
    }
    </style>
    """, unsafe_allow_html=True)



makes = sorted(final_df["MAKE"].unique())
header_placeholder = st.empty()

def render_header(logo_url):
    header_html = f"""
    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: nowrap;">
        <h1 style="margin: 0; white-space: nowrap; font-size: clamp(1.5rem, 4vw, 2.5rem);">
            Vehicle Recall Risk Predictor
        </h1>
        <img src="{logo_url}" width="95" style="flex-shrink: 0; margin-left: 16px;">
    </div>
    """
    header_placeholder.markdown(header_html, unsafe_allow_html=True)

# initial render before make is known
render_header(CAR_LOGOS.get("ACURA"))  # or any sensible default

st.markdown("Predict a vehicle's recall risk based on National Highway Traffic Safety Administration (NHTSA) consumer complaint data.")
st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    selected_make = st.selectbox("Make", makes)

logo_url = CAR_LOGOS.get(selected_make.upper())
render_header(logo_url)  





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
probability = round(vehicle["Probability_Recall"].values[0] * 100, 0)
prediction = vehicle["Predict_Recall"].values[0]
#shap_values = explainer(X_vehicle)

if "Recall" in vehicle.columns:
    actual_recall = int(vehicle["Recall"].values[0])
else:
    actual_recall = None

# ── top metrics row ───────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
prob_col, recall_col, complaints_col = st.columns([1.5, 1.5, 1.5])



# usage example



with prob_col:
    render_risk_gauge(probability)  

with recall_col:
    if actual_recall is not None:
        recall_text = "Recall" if actual_recall == 1 else "No Recall"
        recall_color = "#ff4b4b" if actual_recall == 1 else "#00cc88"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Current Recall Status</div>
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
col_shap, col = st.columns([1, 1])


with col_shap:
    st.markdown(
        '<div class="section-header" style="margin-top: -100px;">What drives this prediction</div>', 
        unsafe_allow_html=True
    )
    shap_cols = [c for c in vehicle.columns if c.startswith('shap_')]
    shap_row = vehicle[shap_cols].iloc[0]
    feature_impacts = shap_row.abs().sort_values(ascending=False)
    feature_impacts.index = feature_impacts.index.str.replace('shap_', '')
    
    feature_names = {
        'complaints_first_12m': 'First Year Complaint Volume',
        'crash_ratio': 'Crash Ratio',
        'keybert_safety_score': 'Complaint Description',
        'median_mileage': 'Median Mileage Before Failure'
    }
    feature_impacts.index = feature_impacts.index.map(lambda x: feature_names.get(x, x))
    

    feature_impacts_pct = (feature_impacts / feature_impacts.sum()) * 100
    
    plot_df = feature_impacts_pct.reset_index()
    plot_df.columns = ["Feature", "Impact"]
    
    fig_bar = px.bar(
    plot_df,
    x="Impact",
    y="Feature",
    orientation="h"
    )
    
    fig_bar.update_traces(
        marker_color='#4A90E2',
        hovertemplate='<b>%{y}</b><br>Impact: %{x:.2f}%<extra></extra>'
    )
    fig_bar.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=20, b=0),
        showlegend=False,
        xaxis_title="Relative Importance (%)",
        yaxis_title="Metric",
    )

    fig_bar.update_layout(yaxis=dict(autorange="reversed"))
    
    st.plotly_chart(fig_bar)

with col:
    # ── map ───────────────────────────────────────────────────────────
    st.markdown(
        '<div class="section-header" style="margin-top: -100px;">Complaint Distribution by State</div>', 
        unsafe_allow_html=True
    )

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