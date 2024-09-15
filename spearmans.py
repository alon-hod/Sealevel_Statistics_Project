# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 15:46:42 2024

@author: assaf
"""

import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
output_path = 'output'
sealevel = pd.read_csv('sealevel_df.csv')
sealevel = sealevel.drop(columns=['Unnamed: 0'])

# Display columns
print(sealevel.columns)

# Function to check normality and perform non-parametric tests if needed
def check_normality_and_test(df, alpha=0.05):
    normality_results = {}
    non_parametric_tests = {}
    
    for column in df.columns:
        # Kolmogorov-Smirnov test for normality
        stat, p_value = stats.kstest(df[column], 'norm', args=(np.mean(df[column]), np.std(df[column])))
        normality_results[column] = (stat, p_value)
        
        # If p-value < alpha, we reject the null hypothesis of normality
        if p_value < alpha:
            # Perform a non-parametric test: Wilcoxon signed-rank test against zero
            stat_non_param, p_value_non_param = stats.wilcoxon(df[column] - np.median(df[column]))
            non_parametric_tests[column] = (stat_non_param, p_value_non_param)
        else:
            non_parametric_tests[column] = None
    
    return normality_results, non_parametric_tests

# Check normality and perform non-parametric tests
normality_results, non_parametric_tests = check_normality_and_test(sealevel)

# Print results
for column in sealevel.columns:
    stat, p_value = normality_results[column]
    print(f"Column: {column}")
    print(f"  Kolmogorov-Smirnov test statistic: {stat}, p-value: {p_value}")
    if non_parametric_tests[column]:
        stat_non_param, p_value_non_param = non_parametric_tests[column]
        print(f"  Non-parametric test (Wilcoxon) statistic: {stat_non_param}, p-value: {p_value_non_param}")
    else:
        print(f"  The column is normally distributed.")
    print("\n")

# Visualize the distribution of each column
for column in sealevel.columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(sealevel[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

# Generate the Spearman correlation matrix and p-values
correlation_matrix = sealevel.corr(method='spearman')
p_values_matrix = pd.DataFrame(np.zeros_like(correlation_matrix), columns=correlation_matrix.columns, index=correlation_matrix.index)

for col in correlation_matrix.columns:
    for row in correlation_matrix.index:
        if col != row:
            corr, p_value = stats.spearmanr(sealevel[col], sealevel[row])
            p_values_matrix.loc[row, col] = p_value
        else:
            p_values_matrix.loc[row, col] = np.nan  # Not testing self-correlation

# Annotate significant correlations
significance_level = 0.05
significant = p_values_matrix < significance_level

# 1. Plot and save the full Spearman correlation matrix
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5, fmt=".2f", 
            cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Full Spearman Correlation Matrix')
plt.savefig(output_path + '/full_spearman_correlation_matrix.png', dpi=300)
plt.show()

# 2. Plot and save the Spearman correlation matrix with significance annotations
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5, fmt=".2f", 
            cbar_kws={'label': 'Correlation Coefficient'},
            mask=~significant, annot_kws={"size": 10, "color": "black"})
plt.title('Spearman Correlation Matrix with Significant Correlations')
plt.savefig(output_path + '/spearman_correlation_matrix_significant.png', dpi=300)
plt.show()

# 3. Plot and save the full Spearman correlation matrix with significance levels in parentheses
annot = correlation_matrix.round(2).astype(str) + "\n(p=" + p_values_matrix.round(2).astype(str) + ")"
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=annot, cmap='coolwarm', linewidths=.5, fmt="", 
            cbar_kws={'label': 'Correlation Coefficient'})
plt.title('Full Spearman Correlation Matrix with Significance Levels')
plt.savefig(output_path + '/full_spearman_correlation_matrix_with_significance.png', dpi=300)
plt.show()

# Save the Spearman correlation matrix and p-values as CSV files for reference
correlation_matrix.to_csv(output_path + '/spearman_correlation_matrix.csv')
p_values_matrix.to_csv(output_path + '/spearman_correlation_p_values.csv')
