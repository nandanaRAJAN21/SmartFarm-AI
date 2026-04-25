# %% [markdown]
# # Fertilizer Recommendation - Exploratory Data Analysis
# 
# This notebook explores the fertilizer recommendation dataset to understand feature distributions and class relationships.

# %%
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath('src'))
from data_loader import DataLoader

# Set plot style and directory
sns.set_theme(style="whitegrid")
out_dir = r"C:\Users\NANDANA\OneDrive\Desktop\smartfarm\crop yield prediction\outputs\plots\fertilizer_eda"
os.makedirs(out_dir, exist_ok=True)

# Initialize loader
loader = DataLoader()
df = loader.load_fertilizer()

# %% [markdown]
# ## 1. Overview
# %%
loader.get_info(df)
print("Duplicate rows:", df.duplicated().sum())

# %% [markdown]
# ## 2. Target Analysis
# %%
target_col = 'Recommended_Fertilizer'

plt.figure(figsize=(10, 5))
sns.countplot(data=df, x=target_col, order=df[target_col].value_counts().index, palette='viridis')
plt.title('Recommended Fertilizer Distribution')
plt.xlabel('Fertilizer')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'target_bar_chart.png'))
plt.show()
plt.close()

plt.figure(figsize=(8, 8))
class_counts = df[target_col].value_counts()
plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('viridis', len(class_counts)))
plt.title('Class Distribution Percentages')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'target_pie_chart.png'))
plt.show()
plt.close()

print("\nClass Imbalance Ratio (Max / Min):", class_counts.max() / class_counts.min())

# %% [markdown]
# ## 3. Numerical Feature Analysis
# %%
num_cols = ['Soil_pH', 'Soil_Moisture', 'Organic_Carbon', 'Electrical_Conductivity', 
            'Nitrogen_Level', 'Phosphorus_Level', 'Potassium_Level', 'Temperature', 
            'Humidity', 'Rainfall', 'Fertilizer_Used_Last_Season', 'Yield_Last_Season']

for col in num_cols:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(df[col], kde=True, color='teal')
    plt.title(f'{col} Distribution')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[col], color='teal')
    plt.title(f'{col} Boxplot')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'num_{col}_dist.png'))
    plt.show()
    plt.close()

# Correlation
plt.figure(figsize=(12, 10))
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'correlation_heatmap.png'))
plt.show()
plt.close()

# Boxplot grouped by Target
for col in num_cols:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x=target_col, y=col, palette='Set2')
    plt.title(f'{col} by {target_col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'box_{col}_by_target.png'))
    plt.show()
    plt.close()

# %% [markdown]
# ## 4. Categorical Feature Analysis
# %%
cat_cols = ['Soil_Type', 'Crop_Type', 'Crop_Growth_Stage', 'Season', 'Irrigation_Type', 'Previous_Crop', 'Region']

for col in cat_cols:
    plt.figure(figsize=(10, 5))
    sns.countplot(data=df, x=col, palette='magma')
    plt.title(f'Count Plot of {col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'cat_{col}_count.png'))
    plt.show()
    plt.close()
    
# Heatmaps for crosstabs
crop_fert = pd.crosstab(df['Crop_Type'], df[target_col])
plt.figure(figsize=(10, 6))
sns.heatmap(crop_fert, annot=True, fmt='d', cmap='Blues')
plt.title('Crop Type vs Recommended Fertilizer')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'heatmap_crop_vs_fert.png'))
plt.show()
plt.close()

soil_fert = pd.crosstab(df['Soil_Type'], df[target_col])
plt.figure(figsize=(10, 6))
sns.heatmap(soil_fert, annot=True, fmt='d', cmap='Greens')
plt.title('Soil Type vs Recommended Fertilizer')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'heatmap_soil_vs_fert.png'))
plt.show()
plt.close()

# Stacked Bar (example with Season)
season_fert = pd.crosstab(df['Season'], df[target_col], normalize='index')
season_fert.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='viridis')
plt.title('Fertilizer Distribution across Seasons')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'stacked_season_vs_fert.png'))
plt.show()
plt.close()

# %% [markdown]
# ## 5. NPK Analysis
# %%
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Nitrogen_Level', y='Phosphorus_Level', hue=target_col, palette='tab10', alpha=0.7)
plt.title('N vs P colored by Fertilizer')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'scatter_N_vs_P_fert.png'))
plt.show()
plt.close()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Nitrogen_Level', y='Potassium_Level', hue=target_col, palette='tab10', alpha=0.7)
plt.title('N vs K colored by Fertilizer')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'scatter_N_vs_K_fert.png'))
plt.show()
plt.close()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Phosphorus_Level', y='Potassium_Level', hue=target_col, palette='tab10', alpha=0.7)
plt.title('P vs K colored by Fertilizer')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'scatter_P_vs_K_fert.png'))
plt.show()
plt.close()

# 3D scatter
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
fert_classes = df[target_col].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(fert_classes)))
color_map = dict(zip(fert_classes, colors))
ax.scatter(df['Nitrogen_Level'], df['Phosphorus_Level'], df['Potassium_Level'], 
           c=df[target_col].map(color_map), alpha=0.6)
ax.set_xlabel('Nitrogen')
ax.set_ylabel('Phosphorus')
ax.set_zlabel('Potassium')
plt.title('3D NPK Scatter by Fertilizer')
# Custom legend
import matplotlib.patches as mpatches
handles = [mpatches.Patch(color=color_map[c], label=c) for c in fert_classes]
plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig(os.path.join(out_dir, '3d_scatter_npk.png'))
plt.show()
plt.close()

# %% [markdown]
# ## 6. Pairplot
# %%
# Sample for speed as required
sample_df = df.sample(min(1000, len(df)), random_state=42)
plt.figure(figsize=(12, 12))
# sns.pairplot(sample_df[['Nitrogen_Level', 'Phosphorus_Level', 'Potassium_Level', 'Soil_pH', 'Temperature', target_col]], 
#            hue=target_col, palette='tab10', corner=True)
plt.savefig(os.path.join(out_dir, 'pairplot_fertilizer.png'), bbox_inches='tight')
plt.show()
plt.close()

print("Fertilizer EDA Complete. Plots saved to outputs/plots/fertilizer_eda/")
