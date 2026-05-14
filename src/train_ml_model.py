import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

# Load dataset
df = pd.read_csv(
    "data/ml_dataset.csv",
    parse_dates=["date"]
)

# Sort chronologically
df = df.sort_values("date")

# Features
FEATURES = [
    "momentum_12m",
    "short_term_momentum",
    "volatility_1m"
]

TARGET = "target_return_1m"

# Train/Test split
split_date = "2024-01-01"

train_df = df[df["date"] < split_date]
test_df = df[df["date"] >= split_date]

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)

print("Training model...")

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, predictions)


rank_ic, _ = spearmanr(y_test, predictions)

print()
print("ML Model Evaluation")
print()

print("Mean Squared Error:", mse)
print("Rank IC:", rank_ic)

# Feature importance
importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

print()
print("Feature Importances:")
print()
print(importance_df)

# Save predictions
test_df = test_df.copy()

test_df["predicted_return"] = predictions

test_df.to_csv(
    "results/ml_predictions.csv",
    index=False
)

print()
print("Predictions saved to results/ml_predictions.csv")
