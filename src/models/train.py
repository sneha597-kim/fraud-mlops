import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier


# =====================================================
# 1️⃣ Set Project Base Directory (Professional Way)
# =====================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

data_path = os.path.join(BASE_DIR, "data", "processed", "baseline.csv")
model_dir = os.path.join(BASE_DIR, "models")
model_path = os.path.join(model_dir, "best_model.pkl")

# Ensure models directory exists
os.makedirs(model_dir, exist_ok=True)


# =====================================================
# 2️⃣ Load Dataset
# =====================================================

print("📂 Loading dataset...")
df = pd.read_csv(data_path)

X = df.drop("Class", axis=1)
y = df["Class"]


# =====================================================
# 3️⃣ Train-Test Split
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =====================================================
# 4️⃣ Initialize XGBoost (Handles Imbalance)
# =====================================================

scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss"
)


# =====================================================
# 5️⃣ Train Model
# =====================================================

print("🚀 Training XGBoost model...")
model.fit(X_train, y_train)


# =====================================================
# 6️⃣ Evaluate Model
# =====================================================

preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

print("\n📊 Classification Report:\n")
print(classification_report(y_test, preds))

print("ROC-AUC:", roc_auc_score(y_test, probs))


# =====================================================
# 7️⃣ Save Model
# =====================================================

joblib.dump(model, model_path)

print(f"\n✅ Model saved successfully at: {model_path}")
