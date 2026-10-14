import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve, auc

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

train = pd.read_csv("train.csv")

TARGET = "satisfaction"
ID = "id"

X = train.drop(columns=[TARGET, ID])
y = (train[TARGET] == "True").astype(int)

# ------------------------------------------------------------
# 2. ENCODE CATEGORICAL FEATURES
# ------------------------------------------------------------

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].fillna("__MISSING__").astype(str))
    label_encoders[col] = le

# Fill numeric missing values
for col in num_cols:
    X[col] = X[col].fillna(X[col].median())

# ------------------------------------------------------------
# 3. TRAIN / VALIDATION SPLIT
# ------------------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# ------------------------------------------------------------
# 4. SCALE FEATURES
# ------------------------------------------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)

# ------------------------------------------------------------
# 5. BUILD ANN — 5 hidden layers × 32 nodes + Dropout
# ------------------------------------------------------------
#
# Architecture:
#   Input → [Dense(32, relu) → Dropout(0.3)] × 5 → Dense(1, sigmoid)
#
# Dropout rate of 0.3 is applied after every hidden layer to
# regularise the network and reduce over-fitting.

input_dim = X_train_scaled.shape[1]

model = Sequential([
    # Layer 1
    Dense(32, activation="relu", input_dim=input_dim),
    Dropout(0.3),

    # Layer 2
    Dense(32, activation="relu"),
    Dropout(0.3),

    # Layer 3
    Dense(32, activation="relu"),
    Dropout(0.3),

    # Layer 4
    Dense(32, activation="relu"),
    Dropout(0.3),

    # Layer 5
    Dense(32, activation="relu"),
    Dropout(0.3),

    # Output
    Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ------------------------------------------------------------
# 6. TRAIN
# ------------------------------------------------------------

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    mode="min",
    restore_best_weights=True,
    verbose=1,
)

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=100,
    batch_size=4096,
    callbacks=[early_stop],
    verbose=1,
)

# ------------------------------------------------------------
# 7. EVALUATE — AUC
# ------------------------------------------------------------

val_probs = model.predict(X_val_scaled).ravel()
# Clip to avoid log(0) / fully-saturated outputs before AUC calculation
val_probs_clipped = np.clip(val_probs, 1e-7, 1 - 1e-7)
roc_auc = roc_auc_score(y_val, val_probs_clipped)

print(f"\n--- VALIDATION RESULTS ---")
print(f"ROC-AUC: {roc_auc:.5f}")

# ------------------------------------------------------------
# 8. PLOT ROC CURVE
# ------------------------------------------------------------

fpr, tpr, _ = roc_curve(y_val, val_probs_clipped)
roc_auc_plot = auc(fpr, tpr)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color="steelblue", lw=2,
         label=f"ROC curve (AUC = {roc_auc_plot:.4f})")
plt.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ANN — Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("ann_roc_curve.png", dpi=150)
plt.show()
print("ROC curve saved to ann_roc_curve.png")
