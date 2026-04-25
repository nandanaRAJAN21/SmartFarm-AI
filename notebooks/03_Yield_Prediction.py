# %% [markdown]
# # Yield Prediction Models
# 
# Compare regression models, tune the best one, evaluate it, and save the result.

# %%
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

sys.path.append(os.path.abspath('src'))
from yield_model import YieldModel

base_dir = r"C:\Users\NANDANA\OneDrive\Desktop\smartfarm\crop yield prediction"
out_dir_plots = os.path.join(base_dir, r"outputs\plots\yield_model")
os.makedirs(out_dir_plots, exist_ok=True)

# %% [markdown]
# ## 1. Load Processed Data
# %%
train = pd.read_csv(os.path.join(base_dir, r"data\processed\yield_train.csv"))
test = pd.read_csv(os.path.join(base_dir, r"data\processed\yield_test.csv"))

X_train = train.drop(columns=['Yield_ton_per_ha'])
y_train = train['Yield_ton_per_ha']
X_test = test.drop(columns=['Yield_ton_per_ha'])
y_test = test['Yield_ton_per_ha']

# %% [markdown]
# ## 2. Train Models and Evaluate
# %%
trainer = YieldModel()
results = trainer.train_and_evaluate(X_train, y_train, X_test, y_test)

# Print comparison table
print("+---------------------+-------+-------+------+-------+--------------+")
print("| Model               |  MAE  |  RMSE |  R²  |  MAPE |   CV R²      |")
print("+---------------------+-------+-------+------+-------+--------------+")
for name, res in results.items():
    print(f"| {name:<19} | {res['MAE']:5.2f} | {res['RMSE']:5.2f} | {(res['R2']):.4f} | {res['MAPE']:5.2f} | {res['CV_R2_mean']:6.4f} ± {res['CV_R2_std']:.4f} |")
print("+---------------------+-------+-------+------+-------+--------------+")

# %% [markdown]
# ## 3. Select and Tune Best Model
# %%
best_name, best_r2, best_rmse = trainer.select_best_model()
print(f"Best Model: {best_name} R²={best_r2:.4f} RMSE={best_rmse:.2f}")

best_params = trainer.tune_best_model(X_train, y_train)

# Re-evaluate
best_model = trainer.best_model
y_pred_tuned = best_model.predict(X_test)
mae, rmse, r2, mape = trainer.evaluate_model(y_test, y_pred_tuned)
print(f"\nImprovement after tuning:")
print(f"R²: {best_r2:.4f} -> {r2:.4f}")

# %% [markdown]
# ## 4. Visualizations
# Save all requested plots to `outputs/plots/yield_model/`.
# %%
# Plot 1: Model Comparison
models = list(results.keys())
r2_scores = [res['R2'] for res in results.values()]
rmse_scores = [res['RMSE'] for res in results.values()]

x = np.arange(len(models))
width = 0.35

fig, ax1 = plt.subplots(figsize=(10, 5))
color = 'tab:blue'
ax1.set_xlabel('Models')
ax1.set_ylabel('R² Score', color=color)
bars1 = ax1.bar(x - width/2, r2_scores, width, color=color, label='R²')
ax1.tick_params(axis='y', labelcolor=color)

# Highlight best
best_idx = models.index(best_name)
bars1[best_idx].set_color('forestgreen')

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('RMSE', color=color)
bars2 = ax2.bar(x + width/2, rmse_scores, width, color=color, label='RMSE')
ax2.tick_params(axis='y', labelcolor=color)

ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=45)
fig.tight_layout()
plt.title('Model Comparison')
plt.savefig(os.path.join(out_dir_plots, 'model_comparison.png'))
plt.show()

# Plot 2: actual_vs_predicted
plt.figure(figsize=(8, 8))
errors = np.abs(y_test - y_pred_tuned)
scatter = plt.scatter(y_test, y_pred_tuned, c=errors, cmap='coolwarm', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.colorbar(scatter, label='Absolute Error')
plt.xlabel('Actual Yield')
plt.ylabel('Predicted Yield')
plt.title(f'Actual vs Predicted ({best_name})')
plt.tight_layout()
plt.savefig(os.path.join(out_dir_plots, 'actual_vs_predicted.png'))
plt.show()

# Plot 3: residual_plot
plt.figure(figsize=(8, 5))
residuals = y_test - y_pred_tuned
sns.scatterplot(x=y_pred_tuned, y=residuals, alpha=0.6, color='purple')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.tight_layout()
plt.savefig(os.path.join(out_dir_plots, 'residual_plot.png'))
plt.show()

# Plot 4: feature_importance
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    features = X_train.columns
    indices = np.argsort(importances)
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], color='forestgreen', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir_plots, 'feature_importance.png'))
    plt.show()

# Plot 5: learning_curve (Train vs Val scores)
from sklearn.model_selection import learning_curve
train_sizes, train_scores, test_scores = learning_curve(
    best_model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1, 
    train_sizes=np.linspace(0.1, 1.0, 5))

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
plt.plot(train_sizes, test_mean, 'o-', color='orange', label='Validation Score')
plt.xlabel('Training Size')
plt.ylabel('R² Score')
plt.title('Learning Curve')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir_plots, 'learning_curve.png'))
plt.show()

# Plot 6: error_distribution
plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True, color='brown')
plt.xlabel('Prediction Error (Actual - Predicted)')
plt.ylabel('Frequency')
plt.title('Error Distribution')
plt.tight_layout()
plt.savefig(os.path.join(out_dir_plots, 'error_distribution.png'))
plt.show()

# %% [markdown]
# ## 5. Save the Best Model
# %%
model_path = os.path.join(base_dir, r"models\yield_best_model.pkl")
trainer.save_model(model_path)

print("===============================================")
print("YIELD PREDICTION — FINAL RESULTS")
print("===============================================")
print(f"Best Model     : {best_name}")
print(f"MAE            : {mae:.4f} tons/ha")
print(f"RMSE           : {rmse:.4f} tons/ha")
print(f"R²             : {r2:.4f}")
print(f"MAPE           : {mape:.2f} %")
print(f"CV R² (5-fold) : {results[best_name]['CV_R2_mean']:.4f} ± {results[best_name]['CV_R2_std']:.4f}")
print("===============================================")
