import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

data_path = os.path.join(os.path.dirname(__file__), "..", "data", "house_data.csv")
model_path = os.path.join(os.path.dirname(__file__), "..", "house_price_model.pkl")

df = pd.read_csv(data_path)

X = df.drop(columns=["price"])
y = df["price"]

categorical_cols = ["location"]
numeric_cols = [c for c in X.columns if c not in categorical_cols]

pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
], remainder="passthrough")

model = Pipeline([
    ("pre", pre),
    ("rf", RandomForestRegressor(n_estimators=200, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

pred = model.predict(X_test)

mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("RMSE:", rmse)
print("MAE:", mae)
print("R2:", r2)

joblib.dump(model, model_path)

sample = X_test.iloc[:20].copy()
sample["actual_price"] = y_test.iloc[:20].values
sample["predicted_price"] = np.round(model.predict(X_test.iloc[:20])).astype(int)
sample.to_csv(os.path.join(os.path.dirname(__file__), "..", "data", "sample_predictions.csv"), index=False)
