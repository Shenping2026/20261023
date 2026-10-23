import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import shap
from PIL import Image
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------- 全局CSS --------------------------
st.markdown("""
<style>
    .stApp { background-color: #f0f8ff; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    * { font-family: 'Times New Roman', serif !important; }
    h1 { font-size: 2.2rem !important; text-align: center !important; font-weight: bold !important; }
    h3 { font-size: 1.5rem !important; text-align: center !important; font-weight: bold !important; }

    div[data-testid="stImage"] {
        display: flex !important; flex-direction: column !important;
        align-items: center !important; justify-content: center !important;
        width: 100% !important; margin: 0 auto !important; text-align: center !important;
    }
    div[data-testid="stImage"] img { margin: 0 auto !important; }
    div[data-testid="caption"] { text-align: center !important; font-size: 1.1rem !important; margin-top: 8px !important; }

    div[data-testid="stNumberInput"] label p, div[data-testid="stSelectbox"] label p {
        font-size: 1.25rem !important; font-weight: bold !important; color: #2c3e50 !important;
    }
    div[data-testid="stNumberInput"] input { font-size: 1.2rem !important; height: 2.8rem !important; }

    [data-testid="stDataFrame"] table { width: 100%; }
    [data-testid="stDataFrame"] th, [data-testid="stDataFrame"] td {
        text-align: center !important; vertical-align: middle !important;
    }
    [data-testid="stDataFrame"] th { font-weight: bold !important; }

    .stButton > button[kind="primary"] {
        background-color: #FFFFFF !important; border: 2px solid red !important; 
        border-radius: 8px !important; height: 5.5rem !important; 
    }
    .stButton > button[kind="primary"] * { color: red !important; font-weight: bold !important; font-size: 2rem !important; }
    .stButton > button[kind="primary"]:hover { background-color: #e6e6e6 !important; border-color: #000000 !important; }
    .stButton > button[kind="primary"]:hover * { color: #000000 !important; }
</style>
""", unsafe_allow_html=True)

# -------------------------- 9个特征配置 --------------------------
feature_display = {
    'SAPS_II': 'SAPS II Score',
    'RESP_RATE': 'Respiratory Rate',
    'LPR': 'LPR (Ratio)',
    'Phenotype A': 'Phenotype A (0=No, 1=Yes)',
    'CCI': 'Charlson Index (CCI)',
    'RPR': 'RPR (Ratio)',
    'WBC': 'WBC Count',
    'BUN': 'BUN',
    'BCR': 'BCR (Ratio)'
}
feature_list = list(feature_display.keys())
feature_units = [
    "score", "bpm", "ratio", "0/1", "score", "ratio", "10⁹/L", "mmol/L", "ratio"
]


# -------------------------- 模型与解释器加载 --------------------------
@st.cache_resource
def load_system_assets():
    model = joblib.load('20261021final_apaki_rf_model.joblib')
    explainer = shap.TreeExplainer(model)

    try:
        # 加载完整的阈值字典
        full_thresholds = joblib.load('20261021threshold_config.joblib')
        # 强制只提取并保留字典中的前 3 个阈值选项
        thresholds_dict = {k: full_thresholds[k] for k in list(full_thresholds.keys())[:3]}
    except Exception:
        # 文件加载失败时的 3 个兜底阈值
        thresholds_dict = {'Default_0.5': 0.5000, 'Optimal_Youden': 0.3500, 'Strict_FixedFPR': 0.1500}

    return model, explainer, thresholds_dict

try:
    model, explainer, thresholds_dict = load_system_assets()
except Exception as e:
    st.error(f"Initialization failed: {e}\nPlease ensure model and config files exist.")
    st.stop()

# -------------------------- 页面结构 --------------------------
st.set_page_config(page_title="ICU Mortality Predictor", layout="wide")
st.title("ICU Mortality Risk Predictor for Colorectal Cancer (CRC) Patients")
st.markdown("---")

col_left, col_mid, col_right = st.columns([0.8, 1, 1.2])

with col_left:
    st.subheader("System Introduction")
    st.markdown("""
    <p style="font-size: 1.3rem; line-height: 1.8; text-align: justify; margin: 0;">
     &nbsp;&nbsp;&nbsp;&nbsp;This intelligent risk prediction system is a professional clinical decision support tool specifically developed for colon cancer patients admitted to the Intensive Care Unit (ICU). Utilizing the advanced Random Forest machine learning algorithm, this system accurately predicts the ultimate clinical endpoint: <strong>Survival vs. Mortality</strong>.
    </p>
    <p style="font-size: 1.35rem; font-weight: bold; margin: 18px 0 8px 0;">Core Technical Features:</p>
    <ul style="font-size: 1.3rem; line-height: 1.8; padding-left: 20px; margin: 0;">
    <li><strong>9 Multidimensional Predictors:</strong> Incorporates core clinical indicators including SAPS II, vital signs, and critical biomarkers.</li>
    <li><strong>Dynamic Thresholds:</strong> Allows selection among multiple operational decision cutoffs to balance sensitivity and specificity.</li>
    <li><strong>Explainable AI via SHAP:</strong> Visually unpacks the "black box" of the model.</li>
    </ul>
    """, unsafe_allow_html=True)
    try:
        local_img = Image.open("1.png")
        st.image(local_img, caption="Clinical Decision Support System", width=500)
    except FileNotFoundError:
        st.info("Local image missing (1.png).")

with col_mid:
    st.subheader("Parameter Input")
    input_dict = {}

    input_dict["SAPS_II"] = st.number_input("SAPS II Score", min_value=0.0, max_value=200.0, value=30.0, step=1.0)
    input_dict["RESP_RATE"] = st.number_input("Respiratory Rate (bpm)", min_value=0.0, max_value=80.0, value=20.0,
                                              step=1.0)
    input_dict["LPR"] = st.number_input("LPR", min_value=0.0, max_value=100.0, value=1.5, step=0.1)
    input_dict["Phenotype A"] = st.selectbox("Phenotype A", options=[0.0, 1.0],
                                             format_func=lambda x: "0" if x == 0 else "1")
    input_dict["CCI"] = st.number_input("Charlson Index (CCI)", min_value=0.0, max_value=30.0, value=2.0, step=1.0)
    input_dict["RPR"] = st.number_input("RPR", min_value=0.0, max_value=100.0, value=0.15, step=0.01)
    input_dict["WBC"] = st.number_input("WBC (10⁹/L)", min_value=0.0, max_value=200.0, value=7.5, step=0.5)
    input_dict["BUN"] = st.number_input("BUN (mmol/L)", min_value=0.0, max_value=150.0, value=6.5, step=0.5)
    input_dict["BCR"] = st.number_input("BCR", min_value=0.0, max_value=200.0, value=15.0, step=1.0)

    st.markdown("---")
    st.subheader("Decision Threshold Configuration")
    selected_threshold_key = st.selectbox(
        "Select Operating Threshold:",
        options=list(thresholds_dict.keys()),
        help="Different thresholds adjust the strictness of the mortality risk alarm."
    )
    current_threshold = thresholds_dict[selected_threshold_key]
    st.caption(f"Current risk cutoff: **{current_threshold:.4f}**")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Make Risk Prediction", type="primary", use_container_width=True)

with col_right:
    st.subheader("Input Preview & Result")

    display_df = pd.DataFrame({
        "Indicator": [feature_display[feat] for feat in feature_list],
        "Value": [input_dict[feat] for feat in feature_list],
        "Unit": feature_units
    })

    html_table = (
        display_df.style.format({'Value': '{:.2f}'})
        .set_properties(**{'text-align': 'center', 'font-size': '16px', 'border': '1px solid #e0e0e0'})
        .set_table_styles([{'selector': 'th',
                            'props': [('text-align', 'center'), ('font-weight', 'bold'), ('font-size', '16px'),
                                      ('background-color', '#e6f2ff')]}])
        .hide(axis="index").to_html(escape=False)
    ).replace('<table', '<table style="width: 100% !important; min-width:100%; border-collapse: collapse; margin:0;"')

    st.markdown(f'<div style="width: 100%; max-height: 480px; overflow-y: auto;">{html_table}</div>',
                unsafe_allow_html=True)
    st.markdown("---")

    if predict_btn:
        feature_values = [input_dict[feat] for feat in feature_list]
        features = np.array([feature_values], dtype=float)

        predicted_proba = model.predict_proba(features)[0]
        prob_1 = predicted_proba[1]
        prob_1_pct = prob_1 * 100

        predicted_class = 1 if prob_1 >= current_threshold else 0

        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            if predicted_class == 1:
                st.markdown('<h4 style="text-align:center; color:#c00000; margin:10px 0 0 0;">High Mortality Risk</h4>',
                            unsafe_allow_html=True)
                st.markdown(
                    '<div style="height:200px; display:flex; justify-content:center; align-items:center; text-align:center; padding:16px; background:#ffe6e6; border:2px solid #d32f2f; border-radius:8px; margin-top:10px;">'
                    '<p style="font-size:1.3rem; color:#222; margin:0;">Threshold Exceeded. Immediate critical care intervention recommended.</p></div>',
                    unsafe_allow_html=True)
                chart_color = '#E53935'
            else:
                st.markdown('<h4 style="text-align:center; color:#1b5e20; margin:10px 0 0 0;">Low Risk Predicted</h4>',
                            unsafe_allow_html=True)
                st.markdown(
                    '<div style="height:200px; display:flex; justify-content:center; align-items:center; text-align:center; padding:16px; background:#e6f9e6; border:2px solid #2e7d32; border-radius:8px; margin-top:10px;">'
                    '<p style="font-size:1.3rem; color:#222; margin:0;">Below selected risk threshold. Continue routine ICU monitoring.</p></div>',
                    unsafe_allow_html=True)
                chart_color = '#4CAF50'

        with res_col2:
            st.markdown('<h4 style="text-align:center; color:#333; margin:10px 0 0 0;">Mortality Probability</h4>',
                        unsafe_allow_html=True)
            fig_donut = go.Figure(data=[go.Pie(
                labels=['Mortality Prob', 'Survival Prob'],
                values=[prob_1_pct, 100 - prob_1_pct],
                hole=0.75, marker_colors=[chart_color, '#E5E5E5'], textinfo='none', hoverinfo='label+percent'
            )])
            fig_donut.update_layout(
                showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=200,
                annotations=[dict(text=f"{prob_1_pct:.1f}%", x=0.5, y=0.5, font_size=26, showarrow=False,
                                  font=dict(color='#333'))]
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        st.markdown("---")
        st.markdown('<h4 style="text-align:center; font-size:1.6rem; margin:10px 0;">SHAP Feature Contribution</h4>',
                    unsafe_allow_html=True)

        shap_values = explainer.shap_values(features)
        if isinstance(shap_values, list):
            shap_class1 = shap_values[1][0]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_class1 = shap_values[0, :, 1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
            shap_class1 = shap_values[0] if shap_values.shape[0] == 1 else shap_values[:, 1]
        else:
            shap_class1 = np.array(shap_values).flatten()

        shap_class1 = np.array(shap_class1).flatten()[:len(feature_list)]

        shap_df = pd.DataFrame({
            'Feature': [feature_display[feat] for feat in feature_list],
            'Value': feature_values,
            'SHAP Contribution': shap_class1
        })
        shap_sorted_bar = shap_df.reindex(shap_df['SHAP Contribution'].abs().sort_values(ascending=True).index)

        fig_shap = px.bar(shap_sorted_bar, x='SHAP Contribution', y='Feature', orientation='h')
        fig_shap.update_traces(
            marker_color=['#E53935' if val > 0 else '#1f77b4' for val in shap_sorted_bar['SHAP Contribution']])
        fig_shap.update_layout(
            margin=dict(t=10, b=20, l=0, r=0), height=500, xaxis_title="SHAP Value (Impact on Prediction)",
            yaxis_title="",
            plot_bgcolor='rgba(0,0,0,0)'
        )
        fig_shap.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        st.plotly_chart(fig_shap, use_container_width=True)