# %% [markdown]
# # Fertilizer Recommendation Models
# 
# Compare classification models, tune the best one, evaluate it, and save the result.

# %%
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

sys.path.append(os.path.abspath('src'))
from fertilizer_model import FertilizerModel
from sklearn.metrics import confusion_matrix, classification_report

base_dir = r"C:\Users\NANDANA\OneDrive\Desktop\smartfarm\crop yield prediction"
out_dir_plots = os.path.join(base_dir, r"outputs\plots\fertilizer_model")
os.makedirs(out_dir_plots, exist_ok=True)

# %% [markdown]
# ## 1. Load Processed Data
# %%
train = pd.read_csv(os.path.join(base_dir, r"data\processed\fertilizer_train.csv"))
test = pd.read_csv(os.path.join(base_dir, r"data\processed\fertilizer_test.csv"))

X_train = train.drop(columns=['Recommended_Fertilizer'])
y_train = train['Recommended_Fertilizer']
X_test = test.drop(columns=['Recommended_Fertilizer'])
y_test = test['Recommended_Fertilizer']

# Also load preprocessor to get target class names
prep = joblib.load(os.path.join(base_dir, r"models\fertilizer_preprocessor.pkl"))
target_le = prep.label_encoders['Recommended_Fertilizer']
target_names = target_le.classes_

# %% [markdown]
# ## 2. Train Models and Evaluate
# %%
trainer = FertilizerModel()
results = trainer.train_and_evaluate(X_train, y_train, X_test, y_test)

# Print comparison table
print("+---------------------+----------+----------+--------+-------+---------+--------------+")
print("| Model               | Accuracy |Precision | Recall |   F1  | ROC-AUC | CV Accuracy  |")
print("+---------------------+----------+----------+--------+-------+---------+--------------+")
for name, res in results.items():
    print(f"| {name:<19} |  {res['Accuracy']:6.2f}  |  {res['Precision']:6.2f}  | {res['Recall']:6.2f} | {res['F1']:5.2f} |  {res['ROC_AUC']:6.4f} | {res['CV_Acc_mean']:6.2f} ± {res['CV_Acc_std']:.2f} |")
print("+---------------------+----------+----------+--------+-------+---------+--------------+")

# %% [markdown]
# ## 3. Select and Tune Best Model
# %%
best_name, best_f1, best_acc = trainer.select_best_model()
print(f"Best Model: {best_name} F1={best_f1:.2f} Accuracy={best_acc:.2f}")

best_params = trainer.tune_best_model(X_train, y_train)

# Re-evaluate
best_model = trainer.best_model
y_pred_tuned = best_model.predict(X_test)
if hasattr(best_model, 'predict_proba'):
    y_proba_tuned = best_model.predict_proba(X_test)
else:
    y_proba_tuned = np.zeros((len(y_test), len(target_names)))

acc, prec, rec, f1, roc_auc = trainer.evaluate_model(y_test, y_pred_tuned, y_proba_tuned)
print(f"\nImprovement after tuning:")
print(f"Accuracy: {best_acc:.2f}% -> {acc:.2f}%")
print(f"F1: {best_f1:.2f}% -> {f1:.2f}%")

# %% [markdown]
# ## 4. Classification Report
# %%
print("\nClassification Report (Tuned Model):")
print(classification_report(y_test, y_pred_tuned, target_names=target_names))

# %% [markdown]
# ## 5. Visualizations
# Save all requested plots to `outputs/plots/fertilizer_model/`.
# %%
# Plot 1: Model Comparison
models = list(results.keys())
acc_scores = [res['Accuracy'] for res in results.values()]
f1_scores = [res['F1'] for res in results.values()]

x = np.arange(len(models))
width = 0.35

fig, ax1 = plt.subplots(figsize=(10, 5))
color = 'tab:blue'
ax1.set_xlabel('Models')
ax1.set_ylabel('Accuracy (%)', color=color)
bars1 = ax1.bar(x - width/2, acc_scores, width, color=color, label='Accuracy')
ax1.tick_params(axis='y', labelcolor=color)

best_idx = models.index(best_name)
bars1[best_idx].set_color('forestgreen')

ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('F1 Score (%)', color=color)
bars2 = ax2.bar(x + width/2, f1_scores, width, color=color, label='F1')
ax2.tick_params(axis='y', labelcolor=color)

ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=45)
fig.tight_layout()
plt.title('Model Comparison')
plt.savefig(os.path.join(out_dir_plots, 'model_comparison.png'))
plt.show()

# Plot 2: confusion_matrix
cm = confusion_matrix(y_test, y_pred_tuned)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix ({best_name})')
plt.tight_layout()
plt.savefig(os.path.join(out_dir_plots, 'confusion_matrix.png'))
plt.show()

# Plot 3: roc_auc_curves
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

if hasattr(best_model, 'predict_proba'):
    y_test_bin = label_binarize(y_test, classes=range(len(target_names)))
    n_classes = y_test_bin.shape[1]
    
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette("husl", n_classes)
    for i, color in zip(range(n_classes), colors):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba_tuned[:, i])
        roc_auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{target_names[i]} (AUC = {roc_auc_val:.2f})')
        
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC Curves (One-vs-Rest)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir_plots, 'roc_auc_curves.png'))
    plt.show()

# Plot 4: feature_importance
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    features = X_train.columns
    indices = np.argsort(importances)
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), importances[indices], color='teal', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir_plots, 'feature_importance.png'))
    plt.show()

# Plot 5: per_class_metrics
report = classification_report(y_test, y_pred_tuned, target_names=target_names, output_dict=True)
classes = target_names
precisions = [report[c]['precision'] for c in classes]
recalls = [report[c]['recall'] for c in classes]
f1s = [report[c]['f1-score'] for c in classes]

x2 = np.arange(len(classes))
w = 0.25

plt.figure(figsize=(12, 6))
plt.bar(x2 - w, precisions, w, label='Precision', color='skyblue')
plt.bar(x2, recalls, w, label='Recall', color='lightgreen')
plt.bar(x2 + w, f1s, w, label='F1', color='salmon')

plt.xlabel('Classes')
plt.ylabel('Score')
plt.title('Per-Class Metrics')
plt.xticks(x2, classes, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir_plots, 'per_class_metrics.png'))
plt.show()

# Plot 6: prediction_confidence
if hasattr(best_model, 'predict_proba'):
    max_probs = np.max(y_proba_tuned, axis=1)
    plt.figure(figsize=(8, 5))
    sns.histplot(max_probs, bins=20, kde=True, color='purple')
    plt.xlabel('Max Prediction Probability')
    plt.ylabel('Frequency')
    plt.title('Prediction Confidence Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir_plots, 'prediction_confidence.png'))
    plt.show()

# %% [markdown]
# ## 6. Save the Best Model
# %%
model_path = os.path.join(base_dir, r"models\fertilizer_best_model.pkl")
trainer.save_model(model_path)

print("===============================================")
print("FERTILIZER RECOMMENDATION — FINAL RESULTS")
print("===============================================")
print(f"Best Model  : {best_name}")
print(f"Accuracy    : {acc:.2f} %")
print(f"Precision   : {prec:.2f} %")
print(f"Recall      : {rec:.2f} %")
print(f"F1-Score    : {f1:.2f} %")
print(f"ROC-AUC     : {roc_auc:.4f}")
print(f"CV Accuracy : {results[best_name]['CV_Acc_mean']:.2f} ± {results[best_name]['CV_Acc_std']:.2f}")
print("===============================================")
