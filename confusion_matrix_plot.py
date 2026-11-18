import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ============================
# Ground Truth & Predictions
# 1 = Aircraft, 0 = Background
# ============================

y_true = np.array([
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0
])

y_pred = np.array([
    1, 1, 1, 0, 1, 1, 0, 1, 1, 1,
    0, 1, 0, 0, 0, 0, 1, 0
])

# ============================
# Confusion Matrix Calculation
# ============================

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print("Confusion Matrix Values")
print(f"True Positive  (TP): {tp}")
print(f"False Positive (FP): {fp}")
print(f"False Negative (FN): {fn}")
print(f"True Negative  (TN): {tn}")

# ============================
# Normalized Confusion Matrix
# ============================

cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(6,5))
sns.heatmap(
    cm_normalized,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=["Background", "Aircraft"],
    yticklabels=["Background", "Aircraft"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Normalized Confusion Matrix")
plt.tight_layout()
plt.show()