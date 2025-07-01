import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# === Load recorded 2-hand gesture data (safe read) ===
try:
    df = pd.read_csv("gesture_data.csv", on_bad_lines='skip')
except Exception as e:
    print("❌ Failed to load CSV:", e)
    exit()

# === Verify correct number of columns ===
expected_columns = 126 + 1  # 126 features + 1 label
if df.shape[1] != expected_columns:
    print(f"⚠️ Warning: Expected {expected_columns} columns, but got {df.shape[1]}")
    df = df.iloc[:, :expected_columns]  # Truncate extra columns
    df = df.dropna()  # Drop rows with missing values

# Optional: Save clean version
df.to_csv("gesture_data_clean.csv", index=False)
print("✅ Cleaned data saved as 'gesture_data_clean.csv'")

# === Separate features and label ===
X = df.drop(columns=["label"])
y = df["label"]

# === Split dataset ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train classifier ===
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# === Evaluate performance ===
y_pred = model.predict(X_test)
print("\n📊 Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# === Save trained model ===
joblib.dump(model, "gesture_knn_model.pkl")
print("✅ Model saved as 'gesture_knn_model.pkl'")
