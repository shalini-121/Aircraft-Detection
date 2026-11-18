import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import pandas as pd

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# =========================
# Model Performance Data
# =========================
models = ["Faster R-CNN", "RetinaNet"]

# Metrics from evaluation
precision_values = [0.763, 0.918]
recall_values = [0.937, 0.942]
map_05_values = [0.714, 0.865]
f1_scores = [0.841, 0.930]  # F1 = 2 * (precision * recall) / (precision + recall)
accuracy_values = [0.78, 0.86]

# =========================
# Confusion Matrix Data
# For binary classification: Aircraft vs Background
# =========================

# Faster R-CNN Confusion Matrix (example data - replace with actual)
# Based on TP, FP, FN from evaluation
# Assuming: TP=150, FP=47, FN=10, TN=93
faster_rcnn_cm = np.array([
    [93, 47],   # TN, FP (Background)
    [10, 150]   # FN, TP (Aircraft)
])

# RetinaNet Confusion Matrix (example data - replace with actual)
# Assuming: TP=155, FP=14, FN=10, TN=121
retinanet_cm = np.array([
    [121, 14],  # TN, FP (Background)
    [10, 155]   # FN, TP (Aircraft)
])

# =========================
# Function to Calculate Metrics from Confusion Matrix
# =========================
def calculate_metrics_from_cm(cm):
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Specificity': specificity
    }

# Calculate metrics for both models
faster_rcnn_metrics = calculate_metrics_from_cm(faster_rcnn_cm)
retinanet_metrics = calculate_metrics_from_cm(retinanet_cm)

# =========================
# 1️⃣ Confusion Matrices Side by Side
# =========================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Faster R-CNN Confusion Matrix
sns.heatmap(faster_rcnn_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Background', 'Aircraft'],
            yticklabels=['Background', 'Aircraft'],
            ax=axes[0], cbar_kws={'label': 'Count'})
axes[0].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=12, fontweight='bold')
axes[0].set_title('Faster R-CNN Confusion Matrix', fontsize=14, fontweight='bold')

# RetinaNet Confusion Matrix
sns.heatmap(retinanet_cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Background', 'Aircraft'],
            yticklabels=['Background', 'Aircraft'],
            ax=axes[1], cbar_kws={'label': 'Count'})
axes[1].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
axes[1].set_ylabel('True Label', fontsize=12, fontweight='bold')
axes[1].set_title('RetinaNet Confusion Matrix', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
# plt.show()

# =========================
# 2️⃣ Normalized Confusion Matrices
# =========================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Normalize confusion matrices
faster_rcnn_cm_norm = faster_rcnn_cm.astype('float') / faster_rcnn_cm.sum(axis=1)[:, np.newaxis]
retinanet_cm_norm = retinanet_cm.astype('float') / retinanet_cm.sum(axis=1)[:, np.newaxis]

# Faster R-CNN Normalized
sns.heatmap(faster_rcnn_cm_norm, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=['Background', 'Aircraft'],
            yticklabels=['Background', 'Aircraft'],
            ax=axes[0], cbar_kws={'label': 'Percentage'})
axes[0].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=12, fontweight='bold')
axes[0].set_title('Faster R-CNN Normalized Confusion Matrix', fontsize=14, fontweight='bold')

# RetinaNet Normalized
sns.heatmap(retinanet_cm_norm, annot=True, fmt='.2%', cmap='Greens',
            xticklabels=['Background', 'Aircraft'],
            yticklabels=['Background', 'Aircraft'],
            ax=axes[1], cbar_kws={'label': 'Percentage'})
axes[1].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
axes[1].set_ylabel('True Label', fontsize=12, fontweight='bold')
axes[1].set_title('RetinaNet Normalized Confusion Matrix', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('normalized_confusion_matrices.png', dpi=300, bbox_inches='tight')
# plt.show()

# =========================
# 3️⃣ Metrics Comparison Table
# =========================
metrics_df = pd.DataFrame({
    'Model': models,
    'Precision': precision_values,
    'Recall': recall_values,
    'F1-Score': f1_scores,
    'mAP@0.5': map_05_values,
    'Accuracy': accuracy_values
})

print("\n" + "="*70)
print("MODEL PERFORMANCE COMPARISON TABLE")
print("="*70)
print(metrics_df.to_string(index=False))
print("="*70 + "\n")

# =========================
# 4️⃣ Combined Bar Chart - All Metrics
# =========================
x = np.arange(len(models))
width = 0.15

fig, ax = plt.subplots(figsize=(12, 6))

bars1 = ax.bar(x - 2*width, precision_values, width, label='Precision', color='#3498db')
bars2 = ax.bar(x - width, recall_values, width, label='Recall', color='#2ecc71')
bars3 = ax.bar(x, map_05_values, width, label='mAP@0.5', color='#e74c3c')
bars4 = ax.bar(x + width, f1_scores, width, label='F1-Score', color='#f39c12')
bars5 = ax.bar(x + 2*width, accuracy_values, width, label='Accuracy', color='#9b59b6')

ax.set_xlabel('Models', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Comprehensive Performance Comparison of Detection Models', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(loc='lower right', fontsize=10)
ax.set_ylim(0, 1.0)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2, bars3, bars4, bars5]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('comprehensive_metrics_comparison.png', dpi=300, bbox_inches='tight')
# plt.show()

# =========================
# 5️⃣ Line Chart - Metrics Trend
# =========================
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(models, precision_values, marker='o', linewidth=2, markersize=10, label='Precision', color='#3498db')
ax.plot(models, recall_values, marker='s', linewidth=2, markersize=10, label='Recall', color='#2ecc71')
ax.plot(models, map_05_values, marker='^', linewidth=2, markersize=10, label='mAP@0.5', color='#e74c3c')
ax.plot(models, f1_scores, marker='D', linewidth=2, markersize=10, label='F1-Score', color='#f39c12')
ax.plot(models, accuracy_values, marker='*', linewidth=2, markersize=12, label='Accuracy', color='#9b59b6')

ax.set_xlabel('Models', fontsize=12, fontweight='bold')
ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Metric Trends Across Models', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.set_ylim(0.6, 1.0)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('metrics_trend_line_chart.png', dpi=300, bbox_inches='tight')
# plt.show()

# =========================
# 6️⃣ Scatter Plot - Precision vs Recall
# =========================
fig, ax = plt.subplots(figsize=(10, 8))

colors = ['#3498db', '#2ecc71']
sizes = [s * 1000 for s in map_05_values]  # Size based on mAP

for i, model in enumerate(models):
    ax.scatter(recall_values[i], precision_values[i], 
              s=sizes[i], alpha=0.6, c=colors[i], 
              edgecolors='black', linewidth=2, label=model)
    
    # Add annotations
    ax.annotate(model, 
                (recall_values[i], precision_values[i]),
                textcoords="offset points", 
                xytext=(0,10), 
                ha='center',
                fontsize=11,
                fontweight='bold')

ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
ax.set_title('Precision vs Recall (Bubble size = mAP@0.5)', fontsize=14, fontweight='bold')
ax.legend(loc='lower left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.7, 1.0)
ax.set_ylim(0.7, 1.0)

plt.tight_layout()
plt.savefig('precision_recall_scatter.png', dpi=300, bbox_inches='tight')
# plt.show()

# =========================
# 7️⃣ Radar Chart - Multi-metric Comparison
# =========================
categories = ['Precision', 'Recall', 'F1-Score', 'mAP@0.5', 'Accuracy']
N = len(categories)

# Create angles for radar chart
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

# Data for each model
faster_rcnn_values = [precision_values[0], recall_values[0], f1_scores[0], map_05_values[0], accuracy_values[0]]
retinanet_values = [precision_values[1], recall_values[1], f1_scores[1], map_05_values[1], accuracy_values[1]]

faster_rcnn_values += faster_rcnn_values[:1]
retinanet_values += retinanet_values[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

ax.plot(angles, faster_rcnn_values, 'o-', linewidth=2, label='Faster R-CNN', color='#3498db')
ax.fill(angles, faster_rcnn_values, alpha=0.25, color='#3498db')

ax.plot(angles, retinanet_values, 'o-', linewidth=2, label='RetinaNet', color='#2ecc71')
ax.fill(angles, retinanet_values, alpha=0.25, color='#2ecc71')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
ax.set_title('Multi-Metric Radar Comparison', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
ax.grid(True)

plt.tight_layout()
plt.savefig('radar_chart_comparison.png', dpi=300, bbox_inches='tight')
# plt.show()

# =========================
# 8️⃣ Grouped Bar Chart - Individual Metrics
# =========================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Individual Metric Comparisons', fontsize=16, fontweight='bold')

metrics_to_plot = [
    ('Precision', precision_values, '#3498db'),
    ('Recall', recall_values, '#2ecc71'),
    ('F1-Score', f1_scores, '#f39c12'),
    ('mAP@0.5', map_05_values, '#e74c3c'),
    ('Accuracy', accuracy_values, '#9b59b6')
]

for idx, (metric_name, values, color) in enumerate(metrics_to_plot):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    bars = ax.bar(models, values, color=color, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Score', fontsize=10, fontweight='bold')
    ax.set_title(metric_name, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# Hide the last subplot
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('individual_metrics_comparison.png', dpi=300, bbox_inches='tight')
# plt.show()

# =========================
# 9️⃣ Performance Summary Box Plot
# =========================
fig, ax = plt.subplots(figsize=(10, 6))

data_to_plot = [
    [precision_values[0], recall_values[0], f1_scores[0], map_05_values[0], accuracy_values[0]],
    [precision_values[1], recall_values[1], f1_scores[1], map_05_values[1], accuracy_values[1]]
]

bp = ax.boxplot(data_to_plot, labels=models, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2),
                whiskerprops=dict(color='black', linewidth=1.5),
                capprops=dict(color='black', linewidth=1.5))

ax.set_ylabel('Score Distribution', fontsize=12, fontweight='bold')
ax.set_title('Overall Performance Distribution', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0.6, 1.0)

plt.tight_layout()
plt.savefig('performance_boxplot.png', dpi=300, bbox_inches='tight')
# plt.show()

# =========================
# Save Metrics to CSV
# =========================
metrics_df.to_csv('model_comparison_metrics.csv', index=False)
print("\n[SUCCESS] All graphs saved successfully!")
print("[SUCCESS] Metrics saved to 'model_comparison_metrics.csv'")
print("\nGenerated Files:")
print("  1. confusion_matrices.png")
print("  2. normalized_confusion_matrices.png")
print("  3. comprehensive_metrics_comparison.png")
print("  4. metrics_trend_line_chart.png")
print("  5. precision_recall_scatter.png")
print("  6. radar_chart_comparison.png")
print("  7. individual_metrics_comparison.png")
print("  8. performance_boxplot.png")
print("  9. model_comparison_metrics.csv")