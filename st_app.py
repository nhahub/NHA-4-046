import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
from datetime import datetime

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="Rossmann Sales Dashboard",
    page_icon="📈",
    layout="wide"
)

# =========================================
# CUSTOM CSS
# =========================================
st.markdown("""
<style>

/* =========================
   MAIN APP
========================= */
.stApp {
    background-color: #f8fafc;
    color: #1e293b;
    font-family: 'Segoe UI', sans-serif;
}

header {
    visibility: hidden;
}

/* =========================
   SIDEBAR
========================= */
[data-testid="stSidebar"] {
    background-color: #004d40 !important;
    border-right: 1px solid rgba(0,0,0,0.05);
    padding-top: 5px;
}

/* =========================
   SIDEBAR LOGO
========================= */
.logo-title {
    font-size: 34px;
    font-weight: 900;
    font-style: italic;
    color: #ffffff;
    letter-spacing: 1px;
    margin-top: 10px;
    margin-bottom: 25px;

    text-shadow:
        0px 0px 6px rgba(255,255,255,0.3),
        0px 0px 12px rgba(255,255,255,0.15);
}

/* Sidebar Text Elements */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] p {
    color: #ffffff !important;
}

h2, h3 {
    color: #004d40 !important;
}

/* =========================
   INPUTS & SELECTBOXES
========================= */
.stNumberInput input,
.stDateInput input,
.stTextInput input,
textarea,
div[data-baseweb="select"] {
    background-color: rgba(255,255,255,1) !important;
    border-radius: 14px !important;
    border: 1px solid #cbd5e1 !important;
    font-weight: 400 !important;
    color: black !important;
    font-size: 18px !important;
}

.stNumberInput input:focus,
.stDateInput input:focus,
div[data-baseweb="select"]:focus-within {
    border: 1px solid #81c784 !important;
    box-shadow: 0px 0px 12px rgba(129, 199, 132, 0.55) !important;
}

div[data-baseweb="select"] {
    color: black !important;
}

div[data-baseweb="select"] span[data-testid="stWidgetLabel"] + div,
div[data-baseweb="select"] [data-baseweb="select"] div {
    color: black !important;
    font-weight: 400 !important;
    font-size: 18px !important;
}

/* =========================
   NUMBER INPUT BUTTONS
========================= */
div[data-testid="stNumberInputStepUp"], 
div[data-testid="stNumberInputStepDown"],
div[data-testid="stNumberInput"] button {
    background-color: #81c784 !important;
    border-color: #81c784 !important;
    color: white !important;
}

div[data-testid="stNumberInput"] button svg {
    fill: white !important;
    color: white !important;
}

div[data-testid="stNumberInputStepUp"]:hover, 
div[data-testid="stNumberInputStepDown"]:hover,
div[data-testid="stNumberInput"] button:hover {
    background-color: #66bb6a !important;
    border-color: #66bb6a !important;
}

/* =========================
   PLACEHOLDERS STYLE
========================= */
div[data-baseweb="select"] div[aria-live="polite"] {
    color: #6b7280 !important;
    font-weight: 400 !important;
    opacity: 0.85;
}

/* =========================
   DATE PICKER SELECTED DAY
========================= */
button[aria-selected="true"] {
    background-color: #004d40 !important;
    color: white !important;
    border-radius: 50% !important;
}

/* =========================
   MULTISELECT TAG
========================= */
span[data-baseweb="tag"] {
    background: linear-gradient(
        90deg,
        #004d40,
        #125C36
    ) !important;

    color: white !important;
    border-radius: 10px !important;
    border: none !important;
    padding: 4px 8px !important;
    font-weight: 400 !important;
}

span[data-baseweb="tag"] span {
    color: white !important;
    font-weight: 400 !important;
}

div[role="listbox"] * {
    color: black !important;
    font-weight: 400 !important;
}

label {
    color: #1e293b !important;
    font-weight: 600 !important;
}

/* =========================
   BUTTON
========================= */
.stButton {
    margin-top: 25px !important;
}

.stButton > button {
    width: 100%;
    height: 55px;
    border: none;
    border-radius: 16px;

    background: linear-gradient(
        90deg,
        #125C36,
        #1E824C
    ) !important;

    color: white !important;
    font-size: 18px;
    font-weight: 700;
    transition: 0.3s ease;

    box-shadow: 0px 5px 20px rgba(18, 92, 54, 0.25);
}

.stButton > button:hover {
    transform: translateY(-3px);

    background: linear-gradient(
        90deg,
        #1E824C,
        #4caf50
    ) !important;
}

/* =========================
   METRIC CARDS
========================= */
div[data-testid="metric-container"] {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    padding: 25px;
    border-radius: 24px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
}

div[data-testid="stMetricValue"] {
    color: #004d40 !important;
    font-size: 40px !important;
    font-weight: 800 !important;
}

div[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-size: 16px !important;
}

/* =========================
   CUSTOM CARDS
========================= */
.main-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    padding: 30px;
    border-radius: 24px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
    margin-top: 10px;
    margin-bottom: 30px;
}

/* =========================
   INFO ALERT
========================= */
.stAlert {
    border-radius: 18px !important;
    background: #e0f2f1 !important;
    border: 1px solid rgba(0, 77, 64, 0.2) !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    box-shadow: 0px 4px 12px rgba(0, 77, 64, 0.05);
}

.stAlert p {
    color: #004d40 !important;
}

/* =========================
   SCROLLBAR
========================= */
::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-thumb {
    background: #81c784;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# LOAD MODELS
# =========================================
@st.cache_resource
def load_models():
    try:
        xgb_model = joblib.load("models/rossmann_xgboost_best.pkl")
        rf_model = joblib.load("models/rossmann_rf_best.pkl")
        scaler_y = joblib.load("models/scaler_y.pkl")

        return xgb_model, rf_model, scaler_y

    except Exception as e:
        st.error(f"Error Loading Models: {e}")
        return None, None, None

xgb_model, rf_model, scaler_y = load_models()

# =========================================
# SIDEBAR
# =========================================
st.sidebar.markdown(
    '<div class="logo-title">Rossmann store sales</div>',
    unsafe_allow_html=True
)

store = st.sidebar.number_input(
    "Store ID",
    min_value=1,
    max_value=1115,
    value=None,
    placeholder="Enter Store ID..."
)

prediction_date = st.sidebar.date_input(
    "Prediction Date",
    value=None
)

open_store = st.sidebar.selectbox(
    "Store Open?",
    [1, 0],
    index=None,
    placeholder="Select option...",
    format_func=lambda x: "Yes" if x == 1 else "No"
)

promo = st.sidebar.selectbox(
    "Promo Active?",
    [1, 0],
    index=None,
    placeholder="Select option...",
    format_func=lambda x: "Yes" if x == 1 else "No"
)

state_holiday = st.sidebar.selectbox(
    "State Holiday",
    ['0', 'a', 'b', 'c'],
    index=None,
    placeholder="Select holiday..."
)

school_holiday = st.sidebar.selectbox(
    "School Holiday?",
    [1, 0],
    index=None,
    placeholder="Select option...",
    format_func=lambda x: "Yes" if x == 1 else "No"
)

store_type = st.sidebar.selectbox(
    "Store Type",
    ['a', 'b', 'c', 'd'],
    index=None,
    placeholder="Select type..."
)

assortment = st.sidebar.selectbox(
    "Assortment",
    ['a', 'b', 'c'],
    index=None,
    placeholder="Select assortment..."
)

competition_distance = st.sidebar.number_input(
    "Competition Distance",
    min_value=0.0,
    value=500.0
)

promo2 = st.sidebar.selectbox(
    "Promo2 Active?",
    [1, 0],
    index=None,
    placeholder="Select option...",
    format_func=lambda x: "Yes" if x == 1 else "No"
)

selected_models = st.sidebar.multiselect(
    "Select Models",
    ['XGBoost', 'Random Forest'],
    default=['XGBoost']
)

predict_btn = st.sidebar.button("🚀 Predict Sales")

# =========================================
# MAIN PAGE
# =========================================
st.markdown("""
<div class="main-card">

<h2>🤖 AI Sales Forecasting System</h2>

<p style="font-size:18px;">
Predict future Rossmann sales using Machine Learning models.
</p>

<p style="font-size:18px;">
Compare forecasting performance between:
</p>

<ul style="font-size:18px;">
<li>XGBoost</li>
<li>Random Forest</li>
</ul>

<p style="font-size:18px;">
Adjust store parameters from the sidebar and generate predictions instantly.
</p>

</div>
""", unsafe_allow_html=True)

# =========================================
# PREDICTION
# =========================================
if predict_btn:

    if None in [
        store,
        prediction_date,
        open_store,
        promo,
        state_holiday,
        school_holiday,
        store_type,
        assortment,
        promo2
    ]:

        st.warning("⚠️ Please fill all fields first.")

    elif xgb_model is not None:

        year = prediction_date.year
        month = prediction_date.month
        day = prediction_date.day
        day_of_week = prediction_date.weekday() + 1
        week_of_year = prediction_date.isocalendar()[1]

        if month in [12, 1, 2]:
            season = 0
        elif month in [3, 4, 5]:
            season = 1
        elif month in [6, 7, 8]:
            season = 2
        else:
            season = 3

        data_input = np.zeros((1, 27))

        data_input[0, 0] = store
        data_input[0, 1] = day_of_week
        data_input[0, 2] = open_store
        data_input[0, 3] = promo

        state_holiday_map = {
            '0': 0,
            'a': 1,
            'b': 2,
            'c': 3
        }

        data_input[0, 4] = state_holiday_map[state_holiday]
        data_input[0, 5] = school_holiday

        store_type_map = {
            'a': 0,
            'b': 1,
            'c': 2,
            'd': 3
        }

        data_input[0, 6] = store_type_map[store_type]

        assortment_map = {
            'a': 0,
            'b': 1,
            'c': 2
        }

        data_input[0, 7] = assortment_map[assortment]

        data_input[0, 8] = competition_distance
        data_input[0, 9] = month
        data_input[0, 10] = year
        data_input[0, 11] = promo2
        data_input[0, 12] = week_of_year
        data_input[0, 13] = year
        data_input[0, 14] = 1
        data_input[0, 15] = year
        data_input[0, 16] = month
        data_input[0, 17] = day
        data_input[0, 18] = week_of_year
        data_input[0, 19] = 0
        data_input[0, 20] = 12
        data_input[0, 21] = 6
        data_input[0, 22] = season
        data_input[0, 23] = 0
        data_input[0, 24] = 0
        data_input[0, 25] = 0
        data_input[0, 26] = 0

        st.markdown('<div class="main-card">', unsafe_allow_html=True)

        st.subheader("📊 Prediction Results")

        cols = st.columns(len(selected_models))
        predictions = {}

        for i, model_name in enumerate(selected_models):

            with cols[i]:

                current_model = (
                    xgb_model
                    if model_name == 'XGBoost'
                    else rf_model
                )

                try:
                    prediction = current_model.predict(data_input)

                    final_sales = scaler_y.inverse_transform(
                        prediction.reshape(-1, 1)
                    )[0][0]

                    predictions[model_name] = final_sales

                    st.metric(
                        label=f"{model_name} Forecast",
                        value=f"€ {final_sales:,.2f}"
                    )

                except Exception as e:
                    st.error(f"Prediction Error: {e}")

        if len(selected_models) > 1:

            st.markdown("---")
            st.subheader("📉 Models Comparison")

            comparison_df = pd.DataFrame({
                "Model": list(predictions.keys()),
                "Sales": list(predictions.values())
            })

            fig = px.bar(
                comparison_df,
                x="Model",
                y="Sales",
                color="Model",
                color_discrete_sequence=["#1E824C", "#004d40"]
            )

            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font_color="#004d40",
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("👉 Adjust parameters from the sidebar then click Predict Sales.")