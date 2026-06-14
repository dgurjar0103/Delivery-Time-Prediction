import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Food Delivery Time Predictor",
    page_icon="🛵",
    layout="wide",
)

# ── Load model & encoders ─────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("best_random_forest_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model, encoders

@st.cache_data
def load_data():
    return pd.read_csv("Food_Delivery_Times.csv")

model, label_encoders = load_artifacts()
df = load_data()

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f0f1a; color: #f0f0f0; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #1a1a2e; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e1e3f, #2a2a5a);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #3a3a6e;
    }
    [data-testid="stMetricValue"] { color: #f97316 !important; font-size: 2rem !important; }
    [data-testid="stMetricLabel"] { color: #a0a0c0 !important; }

    /* Predict button */
    .stButton > button {
        background: linear-gradient(135deg, #f97316, #ea580c);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 14px 32px;
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        width: 100%;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #ea580c, #c2410c);
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(249,115,22,0.4);
    }

    /* Selectboxes & sliders label */
    label { color: #c0c0e0 !important; font-weight: 500; }

    /* Section headers */
    h2 { color: #f97316; }
    h3 { color: #e0e0ff; }

    /* Result box */
    .result-box {
        background: linear-gradient(135deg, #1e3a2f, #14532d);
        border: 2px solid #22c55e;
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        margin-top: 12px;
    }
    .result-time {
        font-size: 3.5rem;
        font-weight: 800;
        color: #4ade80;
    }
    .result-label {
        font-size: 1rem;
        color: #86efac;
        margin-top: 4px;
    }

    /* Info box */
    .info-chip {
        display: inline-block;
        background: #1e1e3f;
        border: 1px solid #3a3a6e;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.82rem;
        color: #a0a0d0;
        margin: 4px 2px;
    }
    
    /* Divider */
    hr { border-color: #2a2a4a; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛵 About this App")
    st.markdown("""
    This app uses a **Random Forest Regressor** trained on food delivery data
    to predict estimated delivery times.

    **Model details:**
    """)
    st.markdown('<span class="info-chip">🌳 311 estimators</span>', unsafe_allow_html=True)
    st.markdown('<span class="info-chip">📊 7 features</span>', unsafe_allow_html=True)
    st.markdown('<span class="info-chip">🎯 Tuned via RandomizedSearchCV</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Feature Importance:**")

    importances = model.feature_importances_
    feature_names = model.feature_names_in_
    imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    imp_df = imp_df.sort_values("Importance", ascending=False)

    for _, row in imp_df.iterrows():
        bar_pct = int(row["Importance"] * 100)
        st.markdown(f"""
        <div style="margin-bottom:8px">
            <div style="display:flex;justify-content:space-between;font-size:0.82rem;color:#c0c0e0">
                <span>{row['Feature']}</span><span>{row['Importance']:.3f}</span>
            </div>
            <div style="background:#1e1e3f;border-radius:4px;height:6px;margin-top:3px">
                <div style="background:#f97316;width:{bar_pct}%;height:6px;border-radius:4px"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="color:#606080;font-size:0.78rem;text-align:center">
        Built with Streamlit · Random Forest
    </div>
    """, unsafe_allow_html=True)

# ── Main content ──────────────────────────────────────────────────────────────
st.markdown("# 🛵 Food Delivery Time Predictor")
st.markdown("Enter the order details below to get an estimated delivery time.")
st.markdown("---")

# ── Dataset stats row ─────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("📦 Total Orders", f"{len(df):,}")
with c2:
    st.metric("⏱️ Avg Delivery", f"{df['Delivery_Time_min'].mean():.0f} min")
with c3:
    st.metric("📍 Max Distance", f"{df['Distance_km'].max():.1f} km")
with c4:
    st.metric("🌤️ Weather Types", len(label_encoders['Weather'].classes_))

st.markdown("---")

# ── Input form ────────────────────────────────────────────────────────────────
st.markdown("## 📋 Order Details")

col_left, col_right = st.columns(2, gap="large")

with col_left:
    st.markdown("### 📍 Delivery Info")

    distance = st.slider(
        "Distance (km)",
        min_value=float(df["Distance_km"].min()),
        max_value=float(df["Distance_km"].max()),
        value=float(df["Distance_km"].median()),
        step=0.1,
        help="Distance from restaurant to delivery location"
    )

    weather = st.selectbox(
        "🌦️ Weather Condition",
        options=list(label_encoders["Weather"].classes_),
        help="Current weather at time of order"
    )

    traffic = st.selectbox(
        "🚦 Traffic Level",
        options=list(label_encoders["Traffic_Level"].classes_),
        help="Current traffic conditions"
    )

    time_of_day = st.selectbox(
        "🕐 Time of Day",
        options=list(label_encoders["Time_of_Day"].classes_),
        help="Time slot when the order is placed"
    )

with col_right:
    st.markdown("### 🏍️ Courier & Restaurant")

    vehicle = st.selectbox(
        "🚗 Vehicle Type",
        options=list(label_encoders["Vehicle_Type"].classes_),
        help="Type of vehicle used by the courier"
    )

    prep_time = st.slider(
        "🍳 Preparation Time (min)",
        min_value=int(df["Preparation_Time_min"].min()),
        max_value=int(df["Preparation_Time_min"].max()),
        value=int(df["Preparation_Time_min"].median()),
        step=1,
        help="Time taken by the restaurant to prepare the order"
    )

    courier_exp = st.slider(
        "⭐ Courier Experience (years)",
        min_value=float(df["Courier_Experience_yrs"].min()),
        max_value=float(df["Courier_Experience_yrs"].max()),
        value=float(df["Courier_Experience_yrs"].median()),
        step=0.5,
        help="Years of experience of the delivery courier"
    )

    # Live summary card
    st.markdown("#### 📝 Order Summary")
    st.markdown(f"""
    <div style="background:#1a1a2e;border:1px solid #2a2a4a;border-radius:10px;padding:14px;font-size:0.88rem;color:#c0c0e0;line-height:2">
        📍 <b>Distance:</b> {distance} km<br>
        🌦️ <b>Weather:</b> {weather}<br>
        🚦 <b>Traffic:</b> {traffic}<br>
        🕐 <b>Time:</b> {time_of_day}<br>
        🚗 <b>Vehicle:</b> {vehicle}<br>
        🍳 <b>Prep time:</b> {prep_time} min<br>
        ⭐ <b>Courier exp:</b> {courier_exp} yrs
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Predict button ────────────────────────────────────────────────────────────
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict_clicked = st.button("🔮 Predict Delivery Time")

if predict_clicked:
    # Encode categoricals
    weather_enc   = label_encoders["Weather"].transform([weather])[0]
    traffic_enc   = label_encoders["Traffic_Level"].transform([traffic])[0]
    tod_enc       = label_encoders["Time_of_Day"].transform([time_of_day])[0]
    vehicle_enc   = label_encoders["Vehicle_Type"].transform([vehicle])[0]

    input_data = pd.DataFrame([[
        distance,
        weather_enc,
        traffic_enc,
        tod_enc,
        vehicle_enc,
        prep_time,
        courier_exp
    ]], columns=model.feature_names_in_)

    prediction = model.predict(input_data)[0]

    # Get prediction range from individual trees
    tree_preds = np.array([tree.predict(input_data)[0] for tree in model.estimators_])
    low, high = np.percentile(tree_preds, [10, 90])

    st.markdown("---")
    st.markdown("## 🎯 Prediction Result")

    r1, r2, r3 = st.columns(3)
    with r1:
        st.metric("⬇️ Optimistic (10th pct)", f"{low:.0f} min")
    with r2:
        st.metric("🎯 Predicted Time", f"{prediction:.0f} min")
    with r3:
        st.metric("⬆️ Pessimistic (90th pct)", f"{high:.0f} min")

    # Big result box
    st.markdown(f"""
    <div class="result-box">
        <div class="result-time">🕒 {prediction:.0f} minutes</div>
        <div class="result-label">Estimated delivery time (range: {low:.0f}–{high:.0f} min)</div>
    </div>
    """, unsafe_allow_html=True)

    # Contextual message
    avg = df["Delivery_Time_min"].mean()
    if prediction < avg - 10:
        msg = "🚀 **Faster than average!** Great conditions for a quick delivery."
        color = "#22c55e"
    elif prediction > avg + 10:
        msg = "⚠️ **Longer than usual.** Consider weather/traffic conditions."
        color = "#f97316"
    else:
        msg = "✅ **About average.** Typical delivery conditions."
        color = "#60a5fa"

    st.markdown(f"""
    <div style="background:#1a1a2e;border-left:4px solid {color};border-radius:8px;
                padding:14px 18px;margin-top:16px;color:#e0e0f0">
        {msg} &nbsp;|&nbsp; Average in dataset: <b>{avg:.0f} min</b>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#404060;font-size:0.8rem;padding-bottom:12px">
    Food Delivery Time Predictor · Random Forest Model · Built with Streamlit
</div>
""", unsafe_allow_html=True)