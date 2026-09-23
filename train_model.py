import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
data = pd.read_csv("roadsafe_data.csv")

# Convert text columns into numbers
le_driver = LabelEncoder()
le_road = LabelEncoder()
le_weather = LabelEncoder()
le_traffic = LabelEncoder()

data["driver_state"] = le_driver.fit_transform(data["driver_state"])
data["road_condition"] = le_road.fit_transform(data["road_condition"])
data["weather"] = le_weather.fit_transform(data["weather"])
data["traffic"] = le_traffic.fit_transform(data["traffic"])

# Features and target
X = data[
    [
        "driver_state",
        "speed",
        "road_condition",
        "weather",
        "traffic",
        "distance"
    ]
]

y = data["risk_score"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Save model
joblib.dump(model, "roadsafe_model.pkl")

print("Model trained successfully!")
print("Model saved as roadsafe_model.pkl")