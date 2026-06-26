

import streamlit as st
import pandas as pd
import numpy as np
import shap
import plotly.express as px
import plotly.graph_objects as go

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

st.markdown("""
<style>
@media only screen and (max-width: 768px) {
    .mobile-notice {
        display: block !important;
    }
}
@media only screen and (min-width: 769px) {
    .mobile-notice {
        display: none !important;
    }
}
</style>

<div class="mobile-notice">
    <p style="
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    ">
        📱 Rotate to landscape for best experience
    </p>
</div>
""", unsafe_allow_html=True)




import plotly.graph_objects as go
import streamlit as st

def render_risk_gauge(risk_score, title="Recall Risk Score"):
    """
    risk_score: float between 0-100
    """
    # determine color based on score
    if risk_score < 50:
        bar_color = "#2ECC71"  # green
    elif risk_score < 70:
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
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white", 'tickvals': [0, 20, 40, 60, 80, 100], 
            'ticktext': ["0", "20", "40", "60", "80", "100"]},
            'bar': {'color': bar_color, 'thickness': 0.3},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 1,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 49], 'color': 'rgba(17, 51, 28, 1.0)'},
                {'range': [50, 69], 'color': 'rgba(241, 196, 15, 0.15)'},
                {'range': [70, 100], 'color': 'rgba(231, 76, 60, 0.15)'}
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
        margin=dict(l=30, r=30, t=10, b=30),  
    )
    with st.container(key="gauge_container"):
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <style>
    div.st-key-gauge_container {
        margin-top: -30px;
    }
    </style>
    """, unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Vehicle Profile", "Full Rankings", "Non-Recalled High Risk Vehicles", "Manufacturer Data", "Insights", "Methodology"])

with tab1:

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
    
    col_info, metric = st.columns([1.5,1])  # Make col_info wider
    with col_info:
        with st.expander("ℹ️ About this Dashboard"):
            st.write("""
            Every year, tens of thousands of consumers file complaints with NHTSA about vehicle defects. This dashboard shows which vehicles (2022-2026 models) have the highest recall risk based on their complaint data. It provides a **risk score** on a scale from 0 to 100. These scores can be segmented into three specific tiers:

            - Low Risk (0 - 49): Continue Routine Monitoring
            - Medium Risk (50 - 69): Monitor Vehicle Closely
            - High Risk (70 - 100): Investigate Immediately
                     
            **Note:** To ensure statistical accuracy, only vehicles with **at least 10 complaints** are included in this dashboard.
                     
            A machine learning model (logistic regression) calculates the risk score by evaluating specific data points extracted from each complaint:
                     
            - Complaint Description: Identifies keywords within descriptions that strongly correlate with historical recalls.
            - First Year Complaint Proportion: Measures the vehicle's first-year complaints against the manufacturer’s total first-year complaints. This metric is normalized to isolate early defect patterns regardless of overall sales volume.
            - Median Mileage: Analyzes the median mileage of vehicles at the time of complaints to identify premature component defects.
                     
            Since complaints often accumulate before vehicles are officially recalled, the scoring metrics provide manufacturers and regulators with an early warning system. Select a vehicle to explore its risk score, what's driving it, and if it shows signs of a safety issue that haven't yet received official attention.

            For details on the model, see the *Methodology* tab above

            """)

    



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
    recall_count = vehicle["recall_count"].values[0]
    #shap_values = explainer(X_vehicle)

    if "Recall" in vehicle.columns:
        actual_recall = int(vehicle["Recall"].values[0])
    else:
        actual_recall = None

    # ── top metrics row ───────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    prob_col, tier_col, recall_col = st.columns([1.5, 1.5, 1.5])



    # usage example



    with prob_col:
        render_risk_gauge(probability)  
    
    with tier_col:
        if probability < 50:
            risk_tier = "Low Risk"
            risk_action = "Continue Routine Monitoring"
            risk_colour = "#2ECC71"
        elif probability < 70:
            risk_tier = "Medium Risk"
            risk_action = "Monitor Vehicle Closely"
            risk_colour = "#f1c40f"
        else:
            risk_tier = "High Risk"
            risk_action = "Investigate Immediately"
            risk_colour = "#ff4b4b"

        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">RISK TIER</div>
                <div class="metric-value" style="color: {risk_colour}; font-size: 1.8rem;">
                    {risk_tier}
                </div>
                <div style="color: {risk_colour}; font-size: 0.85rem; opacity: 0.85; margin-top: 4px;">
                    {risk_action}
                </div>
            </div>
            """, unsafe_allow_html=True)

    with recall_col:
        if actual_recall is not None:
            recall_text = "Recall" if actual_recall == 1 else "No Recall"
            recall_color = "#ff4b4b" if actual_recall == 1 else "#00cc88"
            recall_count_text = f"{int(recall_count)} recall{'s' if recall_count != 1 else ''} on record" if actual_recall == 1 else "No recalls on record"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">NHTSA RECALL STATUS</div>
                <div class="metric-value" style="color: {recall_color};">{recall_text}</div>
                <div style="color: {recall_color}; font-size: 0.85rem; opacity: 0.85; margin-top: 4px;">
                    {recall_count_text}
                </div>
            </div>
            """, unsafe_allow_html=True)

    spacer_col, reliability_card_col = st.columns([1.5, 3.0])

    # Leave spacer_col empty! This keeps the area under the gauge clear.

        # 2. CREATE THE ROW AND SHIFT IT UPWARD WITH NEGATIVE MARGIN
    spacer_col, reliability_card_col = st.columns([1.5, 3.0])

    # Leave spacer_col empty to keep the space under the gauge clear

    with reliability_card_col:
        st.markdown("""
        <div style="
        margin-top: -160px; 
        position: relative;  
        z-index: 10;        
        padding: 16px 20px;
        border-radius: 8px;
        background-color: #111827;
        border: 1px solid rgba(255,255,255,0.08);
        ">
            <div style="font-size: 0.85rem; color: rgba(255,255,255,0.6); margin-bottom: 10px; font-weight: 500; letter-spacing: 0.05em;">
                    MODEL RELIABILITY
            </div>
            <ul style="margin: 0; padding-left: 0px; color: white; font-size: 0.9rem; font-weight: 500; line-height: 1.5;">
            <li style="margin-bottom: 8px; padding-left: 0px;">
                <span style="margin-left: -1px; display: inline-block;">Recall rates consistently increase across risk tiers, showing the thresholds separate high-risk and low-risk vehicles.</span>
            </li>
            <li style="margin-bottom: 8px; padding-left: 0px;">
                <span style="margin-left: -1px; display: inline-block;"><strong style="color: #FF4B4B;">~96%</strong> of highest-risk vehicles (score 90–100) were recalled. Elevated scores correspond to higher real-world recall rates.</span>
            </li>
            <li style="margin-bottom: 8px; padding-left: 0px;">
                <span style="margin-left: -1px; display: inline-block;"><strong style="color: #1976D2;">10%</strong> of unrecalled vehicles score over 70, highlighting potential early-stage risks before official recalls. </span>
            </li>
            <li style="margin-bottom: 8px; padding-left: 0px;">
                <span style="margin-left: -1px; display: inline-block;"><strong style="color: #00FF00;">70%</strong> of recalled vehicles are identified in medium or high-risk tiers. </span>
            </li>
                    


        </ul>
        <!-- Subtle Divider Line inside the card -->
        <hr style="border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 14px 0 8px 0;">
        
        <!-- Footprint text at the bottom -->
        <div style="color: #6b7280; font-size: 13px; opacity: 0.95; line-height: 1.3;">
            Designed to prioritize risk; effectiveness may vary across vehicle segments. Dashboard only includes vehicles with 10+ complaints to prevent low sample sizes.
        </div>

                
                
                
                    


        </div>
        """, unsafe_allow_html=True)


    

   



    st.markdown("<br>", unsafe_allow_html=True)

    # ── shap + complaints over time ───────────────────────────────────
    col_shap, col = st.columns([1, 1])


    with col_shap:
        st.markdown(
            '<div class="section-header" style="margin-top: 0px;">What drives this prediction</div>', 
            unsafe_allow_html=True
        )
        shap_cols = [c for c in vehicle.columns if c.startswith('shap_')]
        shap_row = vehicle[shap_cols].iloc[0]
        feature_impacts = shap_row.abs().sort_values(ascending=False)
        feature_impacts.index = feature_impacts.index.str.replace('shap_', '')
        
        feature_names = {
            'complaints_first_12m_ratio': 'First Year Complaint Proportion',
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
            height=400,
            margin=dict(l=20, r=20, t=0, b=0),
            showlegend=False,
            xaxis_title="Relative Importance (%)",
            yaxis_title="Metric",
        )

        fig_bar.update_layout(yaxis=dict(autorange="reversed"))

        with st.container(key="shap_chart"):
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("""
        <style>
        div.st-key-shap_chart {
            margin-top: -10px;
        }
        </style>
        """, unsafe_allow_html=True)

    with col:
        # ── map ───────────────────────────────────────────────────────────
        st.markdown(
            '<div class="section-header" style="margin-top: 0px;">Complaint Distribution by State</div>', 
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
            with st.container(key="map"):
                st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("No state data available for this vehicle.")

        st.markdown("""
        <style>
        div.st-key-map {
            margin-top: -10px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        
    # ── footer ────────────────────────────────────────────────────────
    st.markdown("""<div style="margin-top: -40px; text-align: center; width: 100%;">
<hr style="border: none; border-top: 1px solid rgba(255,255,255,0.1); margin-bottom: 12px;">
<div style="color: #8b8fa8; font-size: 12px; line-height: 1.4; padding: 0 10px;">
Data from NHTSA complaints and recalls database. Risk scores are generated by a machine learning model that is trained on NHTSA complaint data and are intended for exploratory analysis only. They should not be used as definitive safety assessments or substitutes for official NHTSA recall determinations.
</div>
</div>""", unsafe_allow_html=True)


with tab2:

    col1, col2, col3, col4, col5, col6 = st.columns([1.2, 1.2, 1.0, 1.3, 0.7, 0.7])
    with col1:
        selected_makes = st.multiselect("Make", options=sorted(makes), key="make_table")

    with col2:
        if selected_makes:
            filtered_for_models = final_df[final_df["MAKE"].isin(selected_makes)]
        else:
            filtered_for_models = final_df
            
        models_list = sorted(filtered_for_models["MODEL"].unique())
        selected_models = st.multiselect("Model", options=models_list, key="model_table")

    with col3:
        filtered_for_years = final_df
        
        if selected_makes:
            filtered_for_years = filtered_for_years[filtered_for_years["MAKE"].isin(selected_makes)]
            
        if selected_models:
            filtered_for_years = filtered_for_years[filtered_for_years["MODEL"].isin(selected_models)]
            
        years_list = sorted(filtered_for_years["YEAR"].unique(), reverse=True)
        selected_years = st.multiselect("Year", options=years_list, key="year")
    
    with col4:
        filtered_for_recall = final_df
        filtered_for_recall['NHTSA Recall Status'] = np.where(filtered_for_recall['Recall'] == 1, "Recall", "No Recall")

        if selected_makes:
            filtered_for_recall = filtered_for_recall[filtered_for_recall["MAKE"].isin(selected_makes)]
            
        if selected_models:
            filtered_for_recall = filtered_for_recall[filtered_for_recall["MODEL"].isin(selected_models)]
        
        if selected_years:
            filtered_for_recall = filtered_for_recall[filtered_for_recall["YEAR"].isin(selected_years)]
            
        recall_list = sorted(filtered_for_recall["NHTSA Recall Status"].unique(), reverse=True)
        selected_recall = st.multiselect("NHTSA Recall Status", options=recall_list, key="recall")

    final_df['Risk Score'] = round(final_df['Probability_Recall'] * 100,0)
    
    abs_min_risk = int(final_df["Risk Score"].min())
    abs_max_risk = int(final_df["Risk Score"].max())

    with col5:
        min_risk_input = st.number_input(
            "Min Risk Score", 
            value=abs_min_risk, 
            step=1,
            key="min_risk"
        )
        
    with col6:
        max_risk_input = st.number_input(
            "Max Risk Score", 
            value=abs_max_risk, 
            step = 1, 
            key="max_risk"
        )




    display_df = final_df.copy()

    if selected_makes:
        display_df = display_df[display_df["MAKE"].isin(selected_makes)]

    if selected_models:
        display_df = display_df[display_df["MODEL"].isin(selected_models)]

    if selected_years:
        display_df = display_df[display_df["YEAR"].isin(selected_years)]
    
    if selected_recall:
        display_df = display_df[display_df["NHTSA Recall Status"].isin(selected_recall)]

    display_df['Risk Score'] = round(display_df['Probability_Recall'] * 100,0)
    display_df = display_df[
        (display_df["Risk Score"] >= min_risk_input) & 
        (display_df["Risk Score"] <= max_risk_input)
    ]



    display_df.rename(columns = {"complaint_count": "Complaints"}, inplace = True)

    shap_cols = [c for c in display_df.columns if c.startswith('shap_')]
    shap_display_cols = {}

    feature_names = {
            'shap_complaints_first_12m_ratio': 'First Year Complaint Contribution (%)',
            'shap_crash_ratio': 'Crash Contribution (%)',
            'shap_keybert_safety_score': 'Complaint Description Contribution (%)',
            'shap_median_mileage': 'Mileage Contribution (%)'
    }
    
    for old_name, new_name in feature_names.items():
        if old_name in display_df.columns:
            total_shap_abs = display_df[shap_cols].abs().sum(axis=1)
            display_df[f'{new_name}'] = (display_df[old_name].abs() / total_shap_abs).fillna(0) * 100

    display_df['NHTSA Recall Status'] = np.where(display_df['Recall'] == 1, "Recall", "No Recall")

    def assign_risk_category(score):
        if score < 50:
            return "Low: Continue Routine Monitoring"
        elif score < 70:
            return "Medium: Monitor Vehicle Closely"
        else:
            return "High: Investigate Immediately"

    display_df["Risk Category"] = display_df["Risk Score"].apply(assign_risk_category)
    
    all_possible_cols = [
        'MAKE', 'MODEL', 'YEAR', 'NHTSA Recall Status', 'Risk Score', 'Risk Category', 'Complaints',
        'First Year Complaint Contribution (%)',
        'Complaint Description Contribution (%)', 'Mileage Contribution (%)'
    ]

    col_config_dict = {
        "Risk Score": st.column_config.NumberColumn("Risk Score", format="%d")
    }

    for col in display_df.columns:
        if col in feature_names.values():
            col_config_dict[col] = st.column_config.NumberColumn(col, format="%.2f%%")


    default_visible_cols = ['MAKE', 'MODEL', 'YEAR', 'Risk Score', 'Risk Category', 'Recall Status']

    existing_all_cols = [c for c in all_possible_cols if c in display_df.columns]
    existing_default_cols = [c for c in default_visible_cols if c in display_df.columns]


    with st.expander("Customize Visible Table Columns"):
        selected_visible_cols = st.multiselect(
            "Select columns to display:",
            options=existing_all_cols,
            default=existing_default_cols
        )

    mandatory_cols = ['MAKE', 'MODEL', 'YEAR']
    for col in mandatory_cols:
        if col in existing_all_cols and col not in selected_visible_cols:
            selected_visible_cols.insert(0, col)

    filtered_table_df = display_df[selected_visible_cols]    

    st.dataframe(
        filtered_table_df, 
        column_config=col_config_dict,  
        use_container_width=True, 
        hide_index=True
    )
with tab3:


    display_df = final_df.copy()

    display_df['Risk Score'] = round(display_df['Probability_Recall'] * 100,0)

    display_df = display_df.loc[(display_df['Risk Score'] > 70) & (display_df['Recall'] == 0)]

    count = len(display_df)

    st.metric(label="⚠️ High-Risk Vehicles Without Recalls Identified", value=count)

    st.markdown(
        f"<div style='color: #8b8fa8; font-size: 14px;'>"
        f"The following vehicles have a score above 70 but no active NHTSA recalls. High scores can indicate a pattern of problematic complaints and strong potential for recall.\
            Manufacturers should prioritize these vehicles as investigation candidates. Even if NHTSA does not issue a recall, the high risk scores show that the vehicles have many problematic complaints that can affect consumer trust and sales."
        f"</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"<div style='color: #FF0000; font-size: 14px; margin-top: 15px;'>"
        f"🚨 Alert: During the development of this dashboard in June 2026, the 2022 Chrysler Pacifica Hybrid (Risk Score: 84) received an NHTSA recall document due to failing battery packs. The vehicle has not yet appeared in the NHTSA database, demonstrating the model's early warning capability in real time."
        f"</div>",
        unsafe_allow_html=True
    )

    display_df["first_complaint_date"] = pd.to_datetime(display_df["first_complaint_date"], errors="coerce")

    display_df['Days Since First Complaint'] = (pd.Timestamp.today() - display_df['first_complaint_date']).dt.days


    display_df.rename(columns = {"complaint_count": "Complaints"}, inplace = True)

    shap_cols = [c for c in display_df.columns if c.startswith('shap_')]
    shap_display_cols = {}

    feature_names = {
            'shap_complaints_first_12m_ratio': 'First Year Complaint Contribution (%)',
            'shap_crash_ratio': 'Crash Contribution (%)',
            'shap_keybert_safety_score': 'Complaint Description Contribution (%)',
            'shap_median_mileage': 'Mileage Contribution (%)'
    }
    
    for old_name, new_name in feature_names.items():
        if old_name in display_df.columns:
            total_shap_abs = display_df[shap_cols].abs().sum(axis=1)
            display_df[f'{new_name}'] = (display_df[old_name].abs() / total_shap_abs).fillna(0) * 100

    display_df['NHTSA Recall Status'] = np.where(display_df['Recall'] == 1, "Recall", "No Recall")
        
    all_possible_cols = [
        'MAKE', 'MODEL', 'YEAR', 'Risk Score', 'Days Since First Complaint',
        'First Year Complaint Contribution (%)',
        'Complaint Description Contribution (%)', 'Mileage Contribution (%)'
    ]

    col_config_dict = {
        "Risk Score": st.column_config.NumberColumn("Risk Score", format="%d")
    }

    for col in display_df.columns:
        if col in feature_names.values():
            col_config_dict[col] = st.column_config.NumberColumn(col, format="%.2f%%")


    display_df = display_df[all_possible_cols]


    # 7. Render the customized table
    st.dataframe(
        display_df, 
        column_config=col_config_dict,  # Uses your existing config dictionary safely
        use_container_width=True, 
        hide_index=True
    )

with tab4:
    st.markdown(
        f"<div style='color: #8b8fa8; font-size: 14px;'>"
        f"The table below provides a risk profile summary by manufacturer, which are ranked by proportion of high-risk models in their lineup."
        f"</div>",
        unsafe_allow_html=True
    )

    display_df = final_df.copy()

    display_df['id'] = np.arange(1, len(display_df) + 1)
    display_df['Risk Score'] = round(display_df['Probability_Recall'] * 100, 2)
    display_df = display_df.reset_index(drop=True)

    display_df['is_high_risk'] = display_df['Risk Score'] >= 70

    make_portfolio_risk = display_df.groupby(['MAKE']).agg(
        total_models_produced=('id', 'count'), 
        high_risk_models=('is_high_risk', 'sum'),
        median_score=('Risk Score', 'median'),
        recall_count = ('Recall', 'sum')
    ).reset_index()

    make_portfolio_risk['median_score'] = make_portfolio_risk['median_score'].round(0).astype(int)

    make_portfolio_risk['High Risk %'] = (make_portfolio_risk['high_risk_models'] / make_portfolio_risk['total_models_produced']) * 100

    high_risk = display_df[display_df['Risk Score'] > 70].groupby('MAKE').agg(
        high_risk_nonrecalled=('Recall', lambda x: (x == 0).sum())
    ).reset_index()

    make_portfolio_risk = make_portfolio_risk.merge(high_risk[['MAKE', 'high_risk_nonrecalled']], on = 'MAKE', how = 'left')   

    make_portfolio_risk['Non-Recalled High Risk %'] = (
        make_portfolio_risk['high_risk_nonrecalled'] / make_portfolio_risk['total_models_produced']
    ) * 100

    make_portfolio_risk = make_portfolio_risk[make_portfolio_risk['total_models_produced'] >= 5].sort_values(
        by='median_score', 
        ascending=False
    )

    col_config_dict = {
        "Risk Score": st.column_config.NumberColumn("Risk Score", format="%d"),
        "High Risk %": st.column_config.NumberColumn("High Risk %", format="%.2f%%"),
        "Non-Recalled High Risk %": st.column_config.NumberColumn("Non-Recalled High Risk %", format="%.2f%%")
    }

    make_portfolio_risk = make_portfolio_risk.rename(columns={'median_score': 'Median Risk Score',
                                                              'total_models_produced': 'Total Vehicles',
                                                              'recall_count': 'Recalls',
                                                              'high_risk_models': 'High Risk Count'})


    all_possible_cols = ['MAKE', 'Median Risk Score', 'Total Vehicles', 'Recalls', 'High Risk Count', 'High Risk %', 'Non-Recalled High Risk %']


    make_portfolio_risk = make_portfolio_risk[all_possible_cols]

    st.dataframe(
        make_portfolio_risk, 
        column_config=col_config_dict,  
        use_container_width=True, 
        hide_index=True
    )

with tab5:
        st.markdown("""
        <style>
        .insight-section {
            margin-bottom: 32px;
        }
        .insight-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #ffffff;
            margin-bottom: 12px;
            padding-bottom: 6px;
            border-bottom: 1px solid #2d2f3e;
        }
        .insight-label {
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.08em;
            color: #8b8fa8;
            margin-bottom: 4px;
        }
        .insight-text {
            font-size: 0.9rem;
            color: #d0d3e8;
            margin-bottom: 12px;
            line-height: 1.6;
        }
        .insight-action {
            font-size: 0.9rem;
            color: #d0d3e8;
            line-height: 1.6;
            padding-left: 12px;
            border-left: 3px solid #ff4b4b;
        }
        </style>

        <div class="insight-section">
            <div class="insight-title">⚠️ Early Warning System</div>
            <div class="insight-label">INSIGHT</div>
            <div class="insight-text">
                Vehicles with risk scores above 70 have not been recalled by NHTSA, 
                indicating potential safety concerns that have not yet received official action.
            </div>
            <div class="insight-label">ACTION</div>
            <div class="insight-action">
                Manufacturers and Regulators should prioritize investigation of flagged vehicles to reduce liability. Even if recalls are not issued, 
                manufacturers should monitor patterns of safety complaints to protect brand trust and consumer retention.
            </div>
        </div>

        <div class="insight-section">
            <div class="insight-title">📈 First Year Complaint Proportion</div>
            <div class="insight-label">INSIGHT</div>
            <div class="insight-text">
                Vehicles with a high percentage of complaints compared to total manufacturer complaints have a high risk of recall.
            </div>
            <div class="insight-label">ACTION</div>
            <div class="insight-action">
                Monitor complaint spikes within the first 12 months of a model's release 
                rather than waiting for volume to accumulate over years.
            </div>
        </div>

        <div class="insight-section">
            <div class="insight-title">🔍 Difficult to Detect Risk Profiles</div>
            <div class="insight-label">INSIGHT</div>
            <div class="insight-text">
                The model struggles with predicting risk scores for near-luxury, luxury, or truck models which barely meet complaint volume thresholds and whose owners often bypass NHTSA reporting. 
                American and European Luxury owners (e.g. Mercedes-Benz, BMW, Audi, Cadillac, Lincoln) often visit dealers before filing formal complaints, while truck owners may route issues through maintenance channels. This suppresses complaint volume.
            </div>
            <div class="insight-label">ACTION</div>
            <div class="insight-action">
                Dealer service visit volumes and warranty claim rates should supplement NHTSA complaint data for luxury segments. Commercial maintenance records should be monitored for trucks. NHTSA data alone will not surface risks in either group.
            </div>
        </div>
    """, unsafe_allow_html=True)

with tab6:
    with open("README.md", "r", encoding="utf-8") as f:
        st.markdown(f.read())

    
