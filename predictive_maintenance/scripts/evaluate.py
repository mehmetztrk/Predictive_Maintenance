import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
import os

# Test setini yükle
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/processed_data.csv")
X = df.drop("Machine failure", axis=1)
y = df["Machine failure"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Modelleri yükle
dt_model = joblib.load("models/decision_tree_model.pkl")
rf_model = joblib.load("models/random_forest_model.pkl")

# Tahminleri al
dt_probs = dt_model.predict_proba(X_test)[:, 1]
rf_probs = rf_model.predict_proba(X_test)[:, 1]

# ROC ve AUC
fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_probs)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
auc_dt = auc(fpr_dt, tpr_dt)
auc_rf = auc(fpr_rf, tpr_rf)

# ROC grafiği çiz
plt.figure(figsize=(8,6))
plt.plot(fpr_dt, tpr_dt, label=f"Decision Tree (AUC = {auc_dt:.2f})")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.2f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()

os.makedirs("output", exist_ok=True)
plt.savefig("output/roc_curve.png")
plt.show()

# Feature importance (sadece Random Forest için)
importances = rf_model.feature_importances_
features = X.columns
feat_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feat_df["Feature"], feat_df["Importance"], color='steelblue')
plt.xlabel("Importance Score")
plt.title("Feature Importance - Random Forest")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("output/feature_importance.png")
plt.show()

# Tahminleri CSV olarak export et
pred_df = X_test.copy()
pred_df["Actual"] = y_test.values
pred_df["RF_Prediction"] = rf_model.predict(X_test)
pred_df["RF_Prob"] = rf_probs

pred_df.to_csv("output/predictions.csv", index=False)
print("✅ ROC curve and feature importance saved to output/.")
print("✅ Predictions saved to output/predictions.csv")
