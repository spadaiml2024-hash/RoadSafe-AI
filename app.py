import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="RoadSafe AI",
    page_icon="🚗",
    layout="wide"
)

# Load model
model = joblib.load("roadsafe_model.pkl")

st.title("🚗 RoadSafe AI")
st.write("Context-Aware Driving Risk Prediction System")

st.divider()

# INPUTS
col1, col2, col3 = st.columns(3)

with col1:
    driver = st.selectbox(
        "Driver State",
        ["Alert", "Drowsy", "Distracted"]
    )

    speed = st.number_input(
        "Vehicle Speed (km/h)",
        min_value=0,
        max_value=150,
        value=50
    )

with col2:
    road = st.selectbox(
        "Road Condition",
        ["Dry", "Wet", "Poor"]
    )

    weather = st.selectbox(
        "Weather",
        ["Clear", "Rain", "Heavy Rain"]
    )

with col3:
    traffic = st.selectbox(
        "Traffic Level",
        ["Low", "Medium", "High"]
    )

    distance = st.number_input(
        "Nearby Vehicle Distance (m)",
        min_value=1,
        max_value=100,
        value=30
    )

st.divider()

# PREDICTION
if st.button("🚀 Predict Risk", use_container_width=True):

    driver_map = {
        "Alert": 0,
        "Distracted": 1,
        "Drowsy": 2
    }

    road_map = {
        "Dry": 0,
        "Poor": 1,
        "Wet": 2
    }

    weather_map = {
        "Clear": 0,
        "Heavy Rain": 1,
        "Rain": 2
    }

    traffic_map = {
        "High": 0,
        "Low": 1,
        "Medium": 2
    }

    input_data = pd.DataFrame([[
        driver_map[driver],
        speed,
        road_map[road],
        weather_map[weather],
        traffic_map[traffic],
        distance
    ]], columns=[
        "driver_state",
        "speed",
        "road_condition",
        "weather",
        "traffic",
        "distance"
    ])

    # AI prediction
    risk = round(model.predict(input_data)[0], 2)

    # Risk level
    if risk <= 30:
        level = "LOW"
    elif risk <= 60:
        level = "MEDIUM"
    elif risk <= 80:
        level = "HIGH"
    else:
        level = "CRITICAL"

    # SAFE SPEED
    safe_speed = speed

    if road == "Wet":
        safe_speed -= 10
    elif road == "Poor":
        safe_speed -= 20

    if weather == "Rain":
        safe_speed -= 10
    elif weather == "Heavy Rain":
        safe_speed -= 20

    if driver == "Drowsy":
        safe_speed -= 15
    elif driver == "Distracted":
        safe_speed -= 10

    if traffic == "High":
        safe_speed -= 10

    if distance < 15:
        safe_speed -= 10

    safe_speed = max(20, safe_speed)

    # RESULTS
    st.subheader("📊 AI Risk Analysis")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("AI Risk Score", risk)

    with c2:
        st.metric("Risk Level", level)

    with c3:
        st.metric(
            "Recommended Safe Speed",
            f"{safe_speed} km/h"
        )

    st.progress(min(risk / 100, 1.0))

    # WARNING
    st.subheader("⚠️ Adaptive Warning")

    if level == "LOW":
        st.success("✅ Drive Normally")
        action = "Maintain your current speed and continue safely."

    elif level == "MEDIUM":
        st.warning("⚠️ Caution: Reduce Speed")
        action = "Reduce speed and maintain a safe distance."

    elif level == "HIGH":
        st.warning("🚨 Warning: Slow Down Immediately")
        action = "Reduce speed immediately and increase following distance."

    else:
        st.error("🚨 CRITICAL: Take Immediate Action")
        action = "Slow down, maintain a safe distance, and avoid risky driving."

    st.subheader("💡 Recommended Action")
    st.info(action)


# SCENARIO GRAPH
st.divider()

st.subheader("📈 Scenario Risk Comparison")

scenario_data = pd.DataFrame({
    "Scenario": [
        "Normal",
        "Moderate",
        "Dangerous"
    ],
    "Risk Score": [
        25.75,
        38.0,
        131.1
    ]
})

fig, ax = plt.subplots()

ax.bar(
    scenario_data["Scenario"],
    scenario_data["Risk Score"]
)

ax.set_xlabel("Driving Scenario")
ax.set_ylabel("Risk Score")
ax.set_title("RoadSafe AI Risk Comparison")

st.pyplot(fig)
