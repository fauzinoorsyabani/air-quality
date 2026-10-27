"""
Forensic Air Quality Analytics & Multi-Objective Machine Learning Pipeline
Production Streamlit Application
Author: Fauzi Noorsyabani (portfolio.web.id)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path
from PIL import Image

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Forensic Air Quality Intelligence | Fauzi Noorsyabani",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #64748b;
        font-size: 0.98rem;
        margin-bottom: 1.2rem;
    }
    .kpi-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .status-badge-good {
        background-color: #dcfce7;
        color: #15803d;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-badge-mod {
        background-color: #fef9c3;
        color: #854d0e;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-badge-bad {
        background-color: #fee2e2;
        color: #b91c1c;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Data & Model Loader
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "air_quality_dataset.csv"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "portfolio_assets"

@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        st.error(f"Dataset not found at {DATA_PATH}")
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH)
    # Lower clamp
    for col in ['PM2.5', 'PM10', 'NOx', 'NO2', 'SO2', 'VOCs', 'CO', 'CO2', 'CH4']:
        if col in df.columns:
            df[col] = df[col].clip(lower=0.0)
    
    # Feature engineering
    df['PM_Ratio'] = df['PM2.5'] / df['PM10'].replace(0, np.nan)
    df['Oxidation_Ratio'] = df['NO2'] / df['NOx'].replace(0, np.nan)
    df['Combustion_Index'] = df['CO'] / df['CO2'].replace(0, np.nan)
    df['Thermal_VOC_Interaction'] = df['Temperature'] * df['VOCs']
    df.fillna(df.median(numeric_only=True), inplace=True)
    
    def categorize(val):
        if val <= 35.0: return 'Baik'
        elif val <= 75.0: return 'Sedang'
        else: return 'Tidak Sehat'
        
    df['Air_Quality_Category'] = df['PM2.5'].apply(categorize)
    return df

@st.cache_resource
def load_ml_models():
    reg_model_path = MODELS_DIR / "pm25_xgb_regressor.joblib"
    reg_scaler_path = MODELS_DIR / "scaler_regressor.joblib"
    reg_feats_path = MODELS_DIR / "feature_cols_reg.joblib"
    
    clf_model_path = MODELS_DIR / "risk_xgb_classifier.joblib"
    clf_scaler_path = MODELS_DIR / "scaler_classifier.joblib"
    clf_feats_path = MODELS_DIR / "feature_cols_clf.joblib"
    class_names_path = MODELS_DIR / "class_names.joblib"
    
    if reg_model_path.exists() and clf_model_path.exists():
        xgb_reg = joblib.load(reg_model_path)
        scaler_reg = joblib.load(reg_scaler_path)
        reg_feats = joblib.load(reg_feats_path)
        
        xgb_clf = joblib.load(clf_model_path)
        scaler_clf = joblib.load(clf_scaler_path)
        clf_feats = joblib.load(clf_feats_path)
        class_names = joblib.load(class_names_path)
        return xgb_reg, scaler_reg, reg_feats, xgb_clf, scaler_clf, clf_feats, class_names
    else:
        st.warning("Pre-trained model artifacts not found; falling back to dynamic model fitting...")
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from xgboost import XGBRegressor, XGBClassifier
        df = load_dataset()
        
        reg_feats = ['PM10', 'NOx', 'NO2', 'SO2', 'VOCs', 'CO', 'CO2', 'CH4', 'Temperature', 'Oxidation_Ratio', 'Combustion_Index', 'Thermal_VOC_Interaction']
        clf_feats = ['NOx', 'NO2', 'SO2', 'VOCs', 'CO', 'CO2', 'CH4', 'Temperature', 'Oxidation_Ratio', 'Combustion_Index', 'Thermal_VOC_Interaction']
        
        scaler_reg = StandardScaler()
        X_reg = scaler_reg.fit_transform(df[reg_feats])
        xgb_reg = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42)
        xgb_reg.fit(X_reg, df['PM2.5'])
        
        le = LabelEncoder()
        y_clf = le.fit_transform(df['Air_Quality_Category'])
        class_names = list(le.classes_)
        scaler_clf = StandardScaler()
        X_clf = scaler_clf.fit_transform(df[clf_feats])
        xgb_clf = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42, eval_metric='mlogloss')
        xgb_clf.fit(X_clf, y_clf)
        
        return xgb_reg, scaler_reg, reg_feats, xgb_clf, scaler_clf, clf_feats, class_names

df = load_dataset()
xgb_reg, scaler_reg, reg_feats, xgb_clf, scaler_clf, clf_feats, class_names = load_ml_models()

# -----------------------------------------------------------------------------
# Top Navigation & Header
# -----------------------------------------------------------------------------
col_header, col_meta = st.columns([3, 1])
with col_header:
    st.markdown('<div class="main-title">Forensic Air Quality Analytics & Multi-Objective ML Pipeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Dual-objective machine learning system for PM2.5 concentration forecasting & zero-leakage hazard severity classification from low-cost gas telemetry.</div>', unsafe_allow_html=True)

with col_meta:
    st.markdown("""
    **Architect:** [Fauzi Noorsyabani](https://fauziasyabani.portfolio.web.id)  
    **GitHub:** [fauzinoorsyabani/air-quality](https://github.com/fauzinoorsyabani/air-quality)  
    **Models:** XGBoost & Random Forest Ensembles
    """)

st.divider()

# -----------------------------------------------------------------------------
# Tabs Interface
# -----------------------------------------------------------------------------
tab_sim, tab_forensics, tab_benchmarks, tab_methodology = st.tabs([
    "🎮 Live AI Telemetry & Predictor",
    "📊 Atmospheric Forensics & EDA",
    "📈 Empirical Benchmarks & Diagnostics",
    "📖 System Architecture & XAI Dossier",
])

# =============================================================================
# TAB 1: Live Simulator & Predictor
# =============================================================================
with tab_sim:
    st.markdown("### 🎛️ Real-Time Multi-Sensor Telemetry Simulator")
    st.markdown("Adjust chemical sensor inputs below to test dual-objective predictions on **Continuous PM2.5** and **Categorical Health Risk** in real-time.")

    # Preset Configurations
    preset_col1, preset_col2, preset_col3, preset_col4 = st.columns([1, 1, 1, 1.5])
    preset = preset_col1.selectbox(
        "Load Environment Preset:",
        ["Custom Values", "Pristine Mountain (Good)", "Moderate Urban Corridor", "Heavy Industrial Smog (Unhealthy)"]
    )

    # Defaults
    if preset == "Pristine Mountain (Good)":
        d_pm10, d_nox, d_no2, d_so2, d_vocs, d_co, d_co2, d_ch4, d_temp = 25.0, 12.0, 8.0, 4.0, 40.0, 0.4, 395.0, 1.6, 20.0
    elif preset == "Moderate Urban Corridor":
        d_pm10, d_nox, d_no2, d_so2, d_vocs, d_co, d_co2, d_ch4, d_temp = 65.0, 48.0, 32.0, 18.0, 110.0, 1.2, 420.0, 2.1, 28.5
    elif preset == "Heavy Industrial Smog (Unhealthy)":
        d_pm10, d_nox, d_no2, d_so2, d_vocs, d_co, d_co2, d_ch4, d_temp = 140.0, 115.0, 78.0, 58.0, 240.0, 2.8, 480.0, 3.4, 34.0
    else:
        d_pm10, d_nox, d_no2, d_so2, d_vocs, d_co, d_co2, d_ch4, d_temp = 72.0, 45.0, 28.0, 16.0, 95.0, 1.1, 415.0, 2.0, 27.0

    col_inputs, col_results = st.columns([1.1, 1.4], gap="large")

    with col_inputs:
        st.subheader("Sensor Telemetry Readings")
        
        c1, c2 = st.columns(2)
        in_pm10 = c1.number_input("PM10 (µg/m³)", min_value=0.0, max_value=300.0, value=float(d_pm10), step=1.0, help="Coarse particulate matter")
        in_temp = c2.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=float(d_temp), step=0.5, help="Ambient thermal state")
        
        c3, c4 = st.columns(2)
        in_nox = c3.number_input("NOx (µg/m³)", min_value=0.0, max_value=250.0, value=float(d_nox), step=1.0, help="Total nitrogen oxides")
        in_no2 = c4.number_input("NO2 (µg/m³)", min_value=0.0, max_value=150.0, value=float(d_no2), step=1.0, help="Nitrogen dioxide")
        
        c5, c6 = st.columns(2)
        in_so2 = c5.number_input("SO2 (µg/m³)", min_value=0.0, max_value=100.0, value=float(d_so2), step=0.5, help="Sulfur dioxide (industrial combustion indicator)")
        in_vocs = c6.number_input("VOCs (ppb)", min_value=0.0, max_value=400.0, value=float(d_vocs), step=2.0, help="Volatile organic compounds")
        
        c7, c8 = st.columns(2)
        in_co = c7.number_input("CO (mg/m³)", min_value=0.0, max_value=10.0, value=float(d_co), step=0.1, help="Carbon monoxide")
        in_co2 = c8.number_input("CO2 (ppm)", min_value=300.0, max_value=800.0, value=float(d_co2), step=5.0, help="Carbon dioxide background")
        
        in_ch4 = st.number_input("CH4 (ppm)", min_value=0.0, max_value=10.0, value=float(d_ch4), step=0.1, help="Methane gas")

        # Computed features
        calc_ci = in_co / in_co2 if in_co2 > 0 else 0
        calc_nor = in_no2 / (in_nox + 1e-6)
        calc_thermal_voc = in_temp * in_vocs

        st.caption(f"🧪 **Computed Atmospheric Features:** CI = `{calc_ci:.6f}` | NOR = `{calc_nor:.4f}` | Thermal×VOC = `{calc_thermal_voc:.1f}`")

    with col_results:
        st.subheader("Dual-Objective AI Inference")

        # Construct feature vector for regression
        input_reg_dict = {
            'PM10': in_pm10,
            'NOx': in_nox,
            'NO2': in_no2,
            'SO2': in_so2,
            'VOCs': in_vocs,
            'CO': in_co,
            'CO2': in_co2,
            'CH4': in_ch4,
            'Temperature': in_temp,
            'Oxidation_Ratio': calc_nor,
            'Combustion_Index': calc_ci,
            'Thermal_VOC_Interaction': calc_thermal_voc,
        }
        df_input_reg = pd.DataFrame([input_reg_dict])[reg_feats]
        scaled_reg = scaler_reg.transform(df_input_reg)
        pred_pm25 = float(xgb_reg.predict(scaled_reg)[0])
        pred_pm25 = max(0.0, pred_pm25)

        # Construct feature vector for classification (Zero-leakage: NO PM10 or PM2.5)
        input_clf_dict = {
            'NOx': in_nox,
            'NO2': in_no2,
            'SO2': in_so2,
            'VOCs': in_vocs,
            'CO': in_co,
            'CO2': in_co2,
            'CH4': in_ch4,
            'Temperature': in_temp,
            'Oxidation_Ratio': calc_nor,
            'Combustion_Index': calc_ci,
            'Thermal_VOC_Interaction': calc_thermal_voc,
        }
        df_input_clf = pd.DataFrame([input_clf_dict])[clf_feats]
        scaled_clf = scaler_clf.transform(df_input_clf)
        pred_clf_idx = int(xgb_clf.predict(scaled_clf)[0])
        pred_clf_probs = xgb_clf.predict_proba(scaled_clf)[0]
        pred_category = class_names[pred_clf_idx]

        # Results Presentation
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.markdown("##### Task 1: PM2.5 Concentration")
            st.markdown(f"<div class='kpi-card'><div style='color:#64748b; font-size:0.85rem;'>Predicted Concentration</div><div class='metric-value'>{pred_pm25:.2f} <span style='font-size:1rem; font-weight:400;'>µg/m³</span></div></div>", unsafe_allow_html=True)
            
            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred_pm25,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 150], 'tickwidth': 1, 'tickcolor': "#64748b"},
                    'bar': {'color': "#0f172a"},
                    'bgcolor': "white",
                    'borderwidth': 1,
                    'bordercolor': "#cbd5e1",
                    'steps': [
                        {'range': [0, 35], 'color': '#bbf7d0'},
                        {'range': [35, 75], 'color': '#fef08a'},
                        {'range': [75, 150], 'color': '#fecaca'},
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 3},
                        'thickness': 0.75,
                        'value': 75.0,
                    }
                }
            ))
            fig_gauge.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with res_col2:
            st.markdown("##### Task 2: Health Hazard Severity")
            
            if pred_category == "Baik":
                badge_html = "<span class='status-badge-good'>● BAIK (Optimal Air Quality)</span>"
            elif pred_category == "Sedang":
                badge_html = "<span class='status-badge-mod'>▲ SEDANG (Moderate Risk)</span>"
            else:
                badge_html = "<span class='status-badge-bad'>■ TIDAK SEHAT (Hazardous Warning)</span>"
                
            st.markdown(f"<div class='kpi-card'><div style='color:#64748b; font-size:0.85rem;'>Zero-Leakage Classification</div><div style='margin-top:6px;'>{badge_html}</div><div style='color:#64748b; font-size:0.8rem; margin-top:8px;'>Confidence: <b>{pred_clf_probs[pred_clf_idx]*100:.1f}%</b></div></div>", unsafe_allow_html=True)

            # Probabilities Bar Chart
            prob_df = pd.DataFrame({
                'Category': class_names,
                'Probability': [p * 100 for p in pred_clf_probs]
            })
            colors = ['#22c55e' if c == 'Baik' else '#eab308' if c == 'Sedang' else '#ef4444' for c in class_names]
            fig_prob = px.bar(
                prob_df, x='Probability', y='Category', orientation='h',
                text='Probability', color='Category',
                color_discrete_map={'Baik': '#22c55e', 'Sedang': '#eab308', 'Tidak Sehat': '#ef4444'}
            )
            fig_prob.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_prob.update_layout(height=200, showlegend=False, margin=dict(l=10, r=30, t=10, b=10), xaxis=dict(range=[0, 110]))
            st.plotly_chart(fig_prob, use_container_width=True)

        st.info("💡 **Architectural Note:** The health hazard classification branch was trained strictly without $PM_{2.5}$ or $PM_{10}$ features to ensure real-world utility during optical sensor outages.")

# =============================================================================
# TAB 2: Atmospheric Forensics & EDA
# =============================================================================
with tab_forensics:
    st.subheader("Atmospheric Telemetry Distributions & Cross-Sensor Correlation")
    
    eda_col1, eda_col2 = st.columns([1, 1])
    
    with eda_col1:
        st.markdown("##### Multivariate Correlation Heatmap")
        corr_matrix = df[['PM2.5', 'PM10', 'NOx', 'NO2', 'SO2', 'VOCs', 'CO', 'CO2', 'CH4', 'Temperature', 'Oxidation_Ratio', 'Combustion_Index']].corr()
        fig_corr = px.imshow(
            corr_matrix, text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Pearson Correlation Matrix of Atmospheric Matrix"
        )
        fig_corr.update_layout(height=480, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_corr, use_container_width=True)

    with eda_col2:
        st.markdown("##### Interactive Bivariate Sensor Scatter")
        x_param = st.selectbox("X-Axis Parameter:", ['Temperature', 'SO2', 'NOx', 'CO', 'PM10', 'VOCs'], index=0)
        y_param = st.selectbox("Y-Axis Target:", ['PM2.5', 'PM10', 'NO2', 'Combustion_Index'], index=0)
        
        fig_scatter = px.scatter(
            df, x=x_param, y=y_param, color='Air_Quality_Category',
            color_discrete_map={'Baik': '#22c55e', 'Sedang': '#eab308', 'Tidak Sehat': '#ef4444'},
            trendline="ols", opacity=0.75,
            title=f"{y_param} vs {x_param} grouped by Health Risk Category"
        )
        fig_scatter.update_layout(height=480, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()
    st.markdown("##### High-Resolution Scientific Diagnostic Gallery (300 DPI)")
    
    img_select = st.selectbox(
        "Select Publication Artifact to Inspect:",
        [
            "01_pollutant_distributions.png",
            "02_multivariate_correlation_matrix.png",
            "03_temperature_impact_analysis.png",
            "04_pm_ratio_analysis.png",
            "05_category_pollutant_characteristics.png",
            "06_regression_performance_diagnostics.png",
            "07_classification_confusion_matrix.png",
            "08_feature_importance_analysis.png",
        ]
    )
    img_path = ASSETS_DIR / img_select
    if img_path.exists():
        st.image(str(img_path), caption=f"Publication Artifact: {img_select} (300 DPI Rendered for Portfolio Dossier)", use_container_width=True)

# =============================================================================
# TAB 3: Empirical Benchmarks & Diagnostics
# =============================================================================
with tab_benchmarks:
    st.subheader("Quantitative Benchmarks & Evaluation Diagnostics")
    
    st.markdown("#### Task 1: PM2.5 Continuous Forecasting (Hold-Out Test Set: 100 Rows)")
    reg_benchmark_data = {
        "Model Architecture": ["Ridge Regression (Baseline)", "Random Forest Regressor", "XGBoost Regressor (Champion)"],
        "5-Fold CV R²": [0.6131, 0.6312, 0.6124],
        "5-Fold CV RMSE": [13.7616, 13.4416, 13.8186],
        "Test RMSE (µg/m³)": [14.6112, 13.8341, 13.7341],
        "Test MAE (µg/m³)": [11.0408, 10.6760, 10.8046],
        "Test R²": [0.6231, 0.6621, 0.6670]
    }
    st.table(pd.DataFrame(reg_benchmark_data))

    st.markdown("#### Task 2: Multiclass Air Hazard Classification (Strict Zero-Leakage)")
    clf_benchmark_data = {
        "Model Architecture": ["Multinomial Logistic (Baseline)", "Random Forest Classifier", "XGBoost Classifier (Champion)"],
        "5-Fold CV Acc": [0.6200, 0.6300, 0.5925],
        "5-Fold CV Macro F1": [0.4804, 0.4905, 0.5109],
        "Test Accuracy": [0.5900, 0.6300, 0.6300],
        "Test Macro Precision": [0.5013, 0.5819, 0.6198],
        "Test Macro Recall": [0.5115, 0.5694, 0.6128],
        "Test Macro F1": [0.4637, 0.4987, 0.6048]
    }
    st.table(pd.DataFrame(clf_benchmark_data))

    st.markdown("#### Detailed Classification Matrix on Hold-Out Test Split (XGBoost)")
    report_data = {
        "Risk Category": ["Baik (Good)", "Sedang (Moderate)", "Tidak Sehat (Unhealthy)", "Macro Average", "Weighted Average"],
        "Precision": [0.64, 0.69, 0.53, 0.62, 0.64],
        "Recall": [0.45, 0.65, 0.74, 0.61, 0.63],
        "F1-Score": [0.53, 0.67, 0.62, 0.60, 0.63],
        "Support (Observations)": [20, 57, 23, 100, 100]
    }
    st.table(pd.DataFrame(report_data))

# =============================================================================
# TAB 4: Architecture & XAI Dossier
# =============================================================================
with tab_methodology:
    st.subheader("Architectural Rationale & Engineering Standards")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("""
        ### 1. Physical Zero-Drift Correction
        Electrochemical telemetry encounters thermal drift resulting in non-physical negative values. Instead of throwing out observations or skewing variance with linear offsets, the pipeline enforces a non-negative lower bound:
        $$\\tilde{x}_i = \\max(0, x_i)$$
        
        ### 2. Domain Chemistry Formulations
        - **Combustion Index ($CI$):** $CO / CO_2$ differentiates vehicular vs industrial fuel burning.
        - **Nitrogen Oxidation Ratio ($NOR$):** $NO_2 / NO_x$ serves as a photochemical proxy for photochemical smog age.
        - **Thermal Interaction:** $Temperature \\times VOCs$ models volatile evaporation dynamics under thermal inversions.
        """)

    with col_m2:
        st.markdown("""
        ### 3. Strict Anti-Data Leakage Protocol
        - **Transform Isolation:** `StandardScaler` is fitted solely on the training splits and strictly transforms the test fold.
        - **Feature Isolation:** In the classification task, $PM_{2.5}$ and $PM_{10}$ are expunged from the predictor matrix. The model is forced to generalize hazard levels directly from gas telemetry ($SO_2, NO_x, NO_2, CO, CO_2, VOCs, CH_4$).
        
        ### 4. FAANG STAR Framework Highlights
        - **Situation:** Expensive optical PM2.5 instruments fail in dusty smart-city deployments.
        - **Task:** Build dual-objective continuous regression & risk classification using low-cost electrochemical arrays.
        - **Action:** Enforced zero-drift clipping, domain feature engineering, and anti-leakage cross-validation.
        - **Result:** $R^2 = 0.6670$ for PM2.5 regression; 0.6048 Macro F1 (+30.4%) with 74% recall on hazardous air.
        """)

st.divider()
st.markdown("<div style='text-align:center; color:#94a3b8; font-size:0.85rem;'>Forensic Air Quality Analytics Pipeline · Engineered by Fauzi Noorsyabani · MIT License</div>", unsafe_allow_html=True)
