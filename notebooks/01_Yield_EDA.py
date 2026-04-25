# %% [markdown]
# # Crop Yield Prediction - Exploratory Data Analysis
# 
# This notebook explores the yield prediction dataset to understand variable distributions, find anomalies, and summarize variables.

# %%
import sys
import os
sys.path.append(os.path.abspath('src'))

from data_loader import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set plot style and directory
sns.set_theme(style="whitegrid")
out_dir = r"C:\Users\NANDANA\OneDrive\Desktop\smartfarm\crop yield prediction\outputs\plots\yield_eda"

# Initialize loader
loader = DataLoader()
df = loader.load_yield()

# %% [markdown]
# ## 1. Overview
# Examine shape, types, missing values.
# %%
loader.get_info(df)

# Check for duplicates
print("Duplicate rows:", df.duplicated().sum())

# %% [markdown]
# ## 2. Target Analysis
# Our target variable is `Yield_ton_per_ha`. Let's view its distribution.
# %%
plt.figure(figsize=(8, 5))
sns.histplot(df['Yield_ton_per_ha'], kde=True, color='forestgreen')
plt.title('Distribution of Yield_ton_per_ha')
plt.xlabel('Yield (tons/ha)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'target_distribution.png'))
plt.show()
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(x=df['Yield_ton_per_ha'], color='forestgreen')
plt.title('Boxplot of Yield_ton_per_ha')
plt.xlabel('Yield (tons/ha)')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'target_boxplot.png'))
plt.show()
plt.close()

print(f"Skewness: {df['Yield_ton_per_ha'].skew():.4f}")
print(f"Kurtosis: {df['Yield_ton_per_ha'].kurt():.4f}")

# Define yield categories
def categorize_yield(val):
    if val < 87: return 'Low'
    elif val < 149: return 'Medium'
    else: return 'High'

df['Yield_Category'] = df['Yield_ton_per_ha'].apply(categorize_yield)

plt.figure(figsize=(8, 5))
sns.countplot(x=df['Yield_Category'], order=['Low', 'Medium', 'High'], palette='viridis')
plt.title('Yield Category Distribution')
plt.xlabel('Category')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'yield_category_dist.png'))
plt.show()
plt.close()

# %% [markdown]
# ## 3. Numerical Feature Analysis
# Analysis of numerical inputs
# %%
num_cols = ['Soil_pH', 'Rainfall_mm', 'Temperature_C', 'Humidity_pct', 
            'Fertilizer_Used_kg', 'Pesticides_Used_kg', 'Planting_Density']

for col in num_cols:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(df[col], kde=True, color='teal')
    plt.title(f'Histogram of {col}')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[col], color='teal')
    plt.title(f'Boxplot of {col}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'num_{col}_dist.png'))
    plt.show()
    plt.close()

# Correlation Heatmap
corr = df[num_cols + ['Yield_ton_per_ha']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'correlation_heatmap.png'))
plt.show()
plt.close()

# Scatter plots with regression line
for col in num_cols:
    plt.figure(figsize=(8, 5))
    sns.regplot(data=df, x=col, y='Yield_ton_per_ha', scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    plt.title(f'{col} vs Yield_ton_per_ha')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'scatter_{col}_vs_yield.png'))
    plt.show()
    plt.close()

# %% [markdown]
# ## 4. Categorical Feature Analysis
# Explore categorical inputs.
# %%
cat_cols = ['Crop', 'Region', 'Soil_Type', 'Irrigation', 'Previous_Crop']

for col in cat_cols:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.countplot(data=df, x=col, palette='Set2')
    plt.title(f'Count Plot of {col}')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    sns.boxplot(data=df, x=col, y='Yield_ton_per_ha', palette='Set2')
    plt.title(f'Yield by {col}')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'cat_{col}_analysis.png'))
    plt.show()
    plt.close()

# Mean yield bar chart per category
for col in cat_cols:
    mean_yield = df.groupby(col)['Yield_ton_per_ha'].mean().reset_index()
    plt.figure(figsize=(8, 5))
    sns.barplot(data=mean_yield, x=col, y='Yield_ton_per_ha', palette='magma')
    plt.title(f'Mean Yield by {col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'mean_yield_{col}.png'))
    plt.show()
    plt.close()

# %% [markdown]
# ## 5. Missing Value Analysis
# Identify patterns and impute values.
# %%
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'missing_values_heatmap.png'))
plt.show()
plt.close()

# Bar chart of missing count
missing_counts = df.isnull().sum()
missing_counts = missing_counts[missing_counts > 0]
if not missing_counts.empty:
    plt.figure(figsize=(8, 5))
    sns.barplot(x=missing_counts.index, y=missing_counts.values, palette='Reds')
    plt.title('Missing Value Counts')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'missing_values_bar.png'))
    plt.show()
    plt.close()

print("Sample rows where Irrigation is null:")
print(df[df['Irrigation'].isnull()].head(3))

print("\nSample rows where Previous_Crop is null:")
print(df[df['Previous_Crop'].isnull()].head(3))

print(f"\nMode for Irrigation: {df['Irrigation'].mode()[0]}")
print(f"Mode for Previous_Crop: {df['Previous_Crop'].mode()[0]}")

# %% [markdown]
# ## 6. Pairplot
# Pairwise relationships matrix.
# %%
sample_df = df.sample(min(1000, len(df)), random_state=42) # sampling for performance
plt.figure(figsize=(15, 15))
# For pairplot we can't easily save efficiently if it's too huge but 1000 is fine
# sns.pairplot(sample_df[num_cols + ['Yield_ton_per_ha', 'Crop']], hue='Crop', palette='tab10', corner=True)
plt.savefig(os.path.join(out_dir, 'pairplot_yield.png'), bbox_inches='tight')
plt.show()
plt.close()

print("EDA Complete. All plots saved to outputs/plots/yield_eda/")
