import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset (the UCI CSV uses semicolons)
df = pd.read_csv("student-mat.csv", sep=";")

# Convert original school grades (0-20) to pointer scale (0-10)
df["sem1_pointer"] = df["G1"] / 2
df["sem2_pointer"] = df["G2"] / 2

# Convert final grade (0-20) to percentage (0-100)
df["final_percentage"] = df["G3"] * 5

features = [
    "age", "studytime", "failures", "absences",
    "sem1_pointer", "sem2_pointer", "activities"
]
target = "final_percentage"

X = df[features]
y = df[target]

numerical_features = [
    "age", "studytime", "failures", "absences",
    "sem1_pointer", "sem2_pointer"
]
categorical_features = ["activities"]

numerical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_transformer, numerical_features),
    ("cat", categorical_transformer, categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=42
    )
}

results = []
trained_models = {}

for name, algorithm in models.items():
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", algorithm)
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE (%)": round(mae, 2),
        "RMSE (%)": round(rmse, 2),
        "R2 Score": round(r2, 3)
    })
    trained_models[name] = model

results_df = pd.DataFrame(results).sort_values("RMSE (%)")
results_df.to_csv("model_results.csv", index=False)

best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]

joblib.dump({
    "model": best_model,
    "features": features,
    "best_name": best_name,
    "results": results_df
}, "student_model.joblib")

plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["RMSE (%)"])
plt.title("Model Comparison - RMSE")
plt.ylabel("RMSE (percentage points; lower is better)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.close()

print("\nModel Evaluation Results:")
print(results_df.to_string(index=False))
print("\nBest Model:", best_name)
print("Model saved successfully!")