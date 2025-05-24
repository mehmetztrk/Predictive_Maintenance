import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# Veriyi oku
df = pd.read_csv("data/processed_data.csv")

# Özellikler ve hedef değişkeni ayır
X = df.drop("Machine failure", axis=1)
y = df["Machine failure"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model 1: Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Model 2: Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Değerlendirme
print("📊 Decision Tree Performance:")
print(classification_report(y_test, dt_model.predict(X_test)))

print("\n📊 Random Forest Performance:")
print(classification_report(y_test, rf_model.predict(X_test)))

# Confusion matrix (ekstra görsel olarak analiz edilebilir)
print("\nDecision Tree Confusion Matrix:")
print(confusion_matrix(y_test, dt_model.predict(X_test)))

print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_model.predict(X_test)))

# Modelleri kaydet
os.makedirs("models", exist_ok=True)
joblib.dump(dt_model, "models/decision_tree_model.pkl")
joblib.dump(rf_model, "models/random_forest_model.pkl")
print("\n✅ Models saved in 'models/' folder.")
