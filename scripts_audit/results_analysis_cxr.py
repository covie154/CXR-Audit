#%%
# Imports
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, cohen_kappa_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

df_reports_overall = pd.read_csv('../data_audit/200_judged.csv')
# Remove rows with missing values in any of the three grading columns
df_clean = df_reports_overall.dropna(subset=['priority_manual', 'priority_algo', 'priority_llm'])

#%%
# Plot the distribution of the dataset
##################################################
#####################################
### Plot the distribution of data ###
#####################################

# Plot the distribution of IP/OP
df_clean['Admission_Type'] = df_clean['Admission_Type'].replace({'I': 1, 'O': 0})
# Plot the distribution as a pie chart
df_clean['Admission_Type'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
plt.title('Distribution of IP/OP')
plt.ylabel('')
plt.show()

# Plot the distribution of the system of findings as a bar graph
import ast
from collections import Counter

df_with_findings = pd.read_csv(r"C:\Users\covie\OneDrive\Documents\Work\Research\PRIME\data_audit\v1\100_findings_v2_classified.csv")

# Function to parse a string representation of a list of dictionaries
def parse_findings(findings_str):
    try:
        # Try literal_eval first
        return ast.literal_eval(findings_str)
    except Exception:
        try:
            # If that fails, try json.loads (after replacing single quotes)
            return json.loads(findings_str.replace("'", '"'))
        except Exception as e:
            print(f"Could not parse findings: {e}")
            return []

# Assuming df_reports_overall has a 'report_findings' column
# that contains a string representation of a list of findings.
all_systems = []
for entry in df_with_findings['report_findings'].dropna():
    findings = parse_findings(entry)
    for find in findings:
        # Expect each finding to be a dict with a "system" key.
        if isinstance(find, dict) and 'system' in find:
            all_systems.append(find['system'])

# Count frequency using Counter
systems_count = Counter(all_systems)
df_systems = pd.DataFrame(list(systems_count.items()), columns=['System', 'Frequency']).sort_values(by='Frequency', ascending=False)

# Plot the distribution as a bar graph
plt.figure(figsize=(10, 6))
sns.barplot(data=df_systems, x='System', y='Frequency', palette='viridis')
plt.title('Distribution of System of Findings')
plt.xlabel('System')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# %%
# Plot some basic stats
########################
### Basic Statistics ###
########################

# Remove rows with missing values in any of the three grading columns
df_clean = df_reports_overall.dropna(subset=['priority_manual', 'priority_algo', 'priority_llm'])

# 1. Basic descriptive statistics
print("# Basic Descriptive Statistics:")
for col in ['priority_manual', 'priority_algo', 'priority_llm', 'judge_grade']:
    print(f"\n{col}:")
    print(f"Mean: {df_clean[col].mean():.2f}")
    print(f"Median: {df_clean[col].median()}")
    print(f"Std Dev: {df_clean[col].std():.2f}")
    print(f"Distribution: {df_clean[col].value_counts().sort_index().to_dict()}")

# 2. Agreement metrics
print("\n# Agreement Metrics:")
# Exact agreement
exact_agreement_algo = (df_clean['priority_algo'] == df_clean['priority_manual']).mean() * 100
exact_agreement_llm = (df_clean['priority_llm'] == df_clean['priority_manual']).mean() * 100
exact_agreement_judge = (df_clean['judge_grade'] == df_clean['priority_manual']).mean() * 100
print(f"Exact Agreement - Algo : {exact_agreement_algo:.2f}%")
print(f"Exact Agreement - LLM  : {exact_agreement_llm:.2f}%")
print(f"Exact Agreement - Judge: {exact_agreement_judge:.2f}%")

# Within-1 agreement
within_one_algo = (abs(df_clean['priority_algo'] - df_clean['priority_manual']) <= 1).mean() * 100
within_one_llm = (abs(df_clean['priority_llm'] - df_clean['priority_manual']) <= 1).mean() * 100
within_one_judge = (abs(df_clean['judge_grade'] - df_clean['priority_manual']) <= 1).mean() * 100
print(f"Within-1 Agreement - Algo : {within_one_algo:.2f}%")
print(f"Within-1 Agreement - LLM  : {within_one_llm:.2f}%")
print(f"Within-1 Agreement - Judge: {within_one_judge:.2f}%")

# Cohen's Kappa
kappa_algo = cohen_kappa_score(df_clean['priority_manual'], df_clean['priority_algo'])
kappa_llm = cohen_kappa_score(df_clean['priority_manual'], df_clean['priority_llm'])
kappa_judge = cohen_kappa_score(df_clean['priority_manual'], df_clean['judge_grade'])
print(f"Cohen's Kappa - Algo : {kappa_algo:.3f}")
print(f"Cohen's Kappa - LLM  : {kappa_llm:.3f}")
print(f"Cohen's Kappa - Judge: {kappa_judge:.3f}")

# 3. Correlation analysis
print("\n# Correlation Analysis:")
pearson_algo, p_value_pearson_algo = pearsonr(df_clean['priority_manual'], df_clean['priority_algo'])
pearson_llm, p_value_pearson_llm = pearsonr(df_clean['priority_manual'], df_clean['priority_llm'])
pearson_judge, p_value_pearson_judge = pearsonr(df_clean['priority_manual'], df_clean['judge_grade'])
print(f"Pearson Correlation - Algo : {pearson_algo:.3f} (p-value: {p_value_pearson_algo:.4f})")
print(f"Pearson Correlation - LLM  : {pearson_llm:.3f} (p-value: {p_value_pearson_llm:.4f})")
print(f"Pearson Correlation - Judge: {pearson_judge:.3f} (p-value: {p_value_pearson_judge:.4f})")

spearman_algo, p_value_spearman_algo = spearmanr(df_clean['priority_manual'], df_clean['priority_algo'])
spearman_llm, p_value_spearman_llm = spearmanr(df_clean['priority_manual'], df_clean['priority_llm'])
spearman_judge, p_value_spearman_judge = spearmanr(df_clean['priority_manual'], df_clean['judge_grade'])
print(f"Spearman Correlation - Algo : {spearman_algo:.3f} (p-value: {p_value_spearman_algo:.4f})")
print(f"Spearman Correlation - LLM  : {spearman_llm:.3f} (p-value: {p_value_spearman_llm:.4f})")
print(f"Spearman Correlation - Judge: {spearman_judge:.3f} (p-value: {p_value_spearman_judge:.4f})")

# 4. Error analysis
print("\n# Error Analysis:")
mae_algo = mean_absolute_error(df_clean['priority_manual'], df_clean['priority_algo'])
mae_llm = mean_absolute_error(df_clean['priority_manual'], df_clean['priority_llm'])
mae_judge = mean_absolute_error(df_clean['priority_manual'], df_clean['judge_grade'])
print(f"Mean Absolute Error - Algo : {mae_algo:.3f}")
print(f"Mean Absolute Error - LLM  : {mae_llm:.3f}")
print(f"Mean Absolute Error - Judge: {mae_judge:.3f}")

rmse_algo = np.sqrt(mean_squared_error(df_clean['priority_manual'], df_clean['priority_algo']))
rmse_llm = np.sqrt(mean_squared_error(df_clean['priority_manual'], df_clean['priority_llm']))
rmse_judge = np.sqrt(mean_squared_error(df_clean['priority_manual'], df_clean['judge_grade']))
print(f"Root Mean Square Error - Algo : {rmse_algo:.3f}")
print(f"Root Mean Square Error - LLM  : {rmse_llm:.3f}")
print(f"Root Mean Square Error - Judge: {rmse_judge:.3f}")

print("\n# Grouped Results:")
print("We can group the grades into three categories:")
print("1: Normal (1-2), 2: Actionable (3-4), 3: Critical (5-6)")
# We can group grade 1 & 2 (normal), 3 & 4 (actionable), 5 & 6 (critical)
# Create a new column for the grouped grades
df_reports_overall['priority_manual_grouped'] = df_reports_overall['priority_manual'].apply(lambda x: 1 if x <= 2 else (2 if x <= 4 else 3))
df_reports_overall['priority_algo_grouped'] = df_reports_overall['priority_algo'].apply(lambda x: 1 if x <= 2 else (2 if x <= 4 else 3))
df_reports_overall['priority_llm_grouped'] = df_reports_overall['priority_llm'].apply(lambda x: 1 if x <= 2 else (2 if x <= 4 else 3))
df_reports_overall['judge_grade_grouped'] = df_reports_overall['judge_grade'].apply(lambda x: 1 if x <= 2 else (2 if x <= 4 else 3))

# Calculate the exact agreement for each approach
exact_agreement_algo_grouped = (df_reports_overall['priority_algo_grouped'] == df_reports_overall['priority_manual_grouped']).mean() * 100
exact_agreement_llm_grouped = (df_reports_overall['priority_llm_grouped'] == df_reports_overall['priority_manual_grouped']).mean() * 100
print(f"Exact Agreement - Algo (grouped): {exact_agreement_algo_grouped:.2f}%")
print(f"Exact Agreement - LLM (grouped): {exact_agreement_llm_grouped:.2f}%")
# Calculate Cohen's Kappa for grouped grades
kappa_algo_grouped = cohen_kappa_score(df_reports_overall['priority_manual_grouped'], df_reports_overall['priority_algo_grouped'])
kappa_llm_grouped = cohen_kappa_score(df_reports_overall['priority_manual_grouped'], df_reports_overall['priority_llm_grouped'])
print(f"Cohen's Kappa - Algo (grouped): {kappa_algo_grouped:.3f}")
print(f"Cohen's Kappa - LLM (grouped): {kappa_llm_grouped:.3f}")

# 5. Clinical relevance analysis
print("\n# Clinical Relevance Analysis:")
print("## Basic Undergrading:")
# Under-grading (potentially dangerous)
undergrade_algo = (df_clean['priority_algo'] < df_clean['priority_manual']).mean() * 100
undergrade_llm = (df_clean['priority_llm'] < df_clean['priority_manual']).mean() * 100
undergrade_judge = (df_clean['judge_grade'] < df_clean['priority_manual']).mean() * 100
print(f"Under-grading Rate - Algo : {undergrade_algo:.2f}%")
print(f"Under-grading Rate - LLM  : {undergrade_llm:.2f}%")
print(f"Under-grading Rate - Judge: {undergrade_judge:.2f}%")

# Critical under-grading (missing high-priority cases)
critical_undergrade_algo = ((df_clean['priority_manual'] >= 4) & 
                           (df_clean['priority_algo'] < 4)).sum()
critical_undergrade_llm = ((df_clean['priority_manual'] >= 4) & 
                          (df_clean['priority_llm'] < 4)).sum()
critical_undergrade_judge = ((df_clean['priority_manual'] >= 4) &
                          (df_clean['judge_grade'] < 4)).sum()
total_critical = (df_clean['priority_manual'] >= 4).sum()

print('\n## Critical Undergrading Rate: ')
print('Cases which the LLM ranked <4 when it should be 4 or 5')
if total_critical > 0:
    print(f"Critical Under-grading - Algo: {critical_undergrade_algo}/{total_critical} cases ({critical_undergrade_algo/total_critical*100:.2f}%)")
    print(f"Critical Under-grading - LLM: {critical_undergrade_llm}/{total_critical} cases ({critical_undergrade_llm/total_critical*100:.2f}%)")
    print(f"Critical Under-grading - Judge: {critical_undergrade_judge}/{total_critical} cases ({critical_undergrade_judge/total_critical*100:.2f}%)")
else:
    print("No critical cases (manual priority ≥ 5) in the dataset")
print("\n")

#%%
# Filter reports where the manual grade is critical (>=4) but either the algorithm or LLM grade is below 5
# critical_undergrading_df = df_reports_overall[
#    ((df_reports_overall['priority_manual'] >= 4) & (df_reports_overall['priority_algo'] < 4)) |
#    ((df_reports_overall['priority_manual'] >= 4) & (df_reports_overall['priority_llm'] < 4)) | 
#    ((df_reports_overall['priority_manual'] >= 4) & (df_reports_overall['judge_grade'] < 4))
#].sample(n=5, random_state=42)
critical_undergrading_df = df_reports_overall[((df_reports_overall['priority_manual'] >= 4) & (df_reports_overall['judge_grade'] < 4))].sample(n=5, random_state=42)

# Print key details of these reports
print("Sample of critically undergraded reports:")
for index, row in critical_undergrading_df.iterrows():
    print(f"{row['REPORT']}")
    print(f"Manual Grade: {row['priority_manual']}; Algo/LLM Grade: {row['priority_algo']}/{row['priority_llm']}")
    print(f"Judge Choice (Algo/LLM): {row['judge_choice']}")
    print("\n**************************************************************\n")

#%%
# Visualization of results
################################
### Visualization of Results ###
################################

# Plot histogram of priority_manual, priority_algo and priority_llm, plus an overlay plot
plt.figure(figsize=(16, 16))

# Histogram for priority_manual
plt.subplot(3, 2, 1)
plt.hist(df_reports_overall['priority_manual'], bins=range(1, 8), color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Histogram of priority_manual')
plt.xlabel('Grade')
plt.ylabel('Frequency')
plt.xticks(range(1, 6))

# Histogram for priority_algo
plt.subplot(3, 2, 2)
plt.hist(df_reports_overall['priority_algo'], bins=range(1, 8), color='lightgreen', edgecolor='black', alpha=0.7)
plt.title('Histogram of priority_algo')
plt.xlabel('Grade')
plt.ylabel('Frequency')
plt.xticks(range(1, 6))

# Histogram for priority_llm
plt.subplot(3, 2, 3)
plt.hist(df_reports_overall['priority_llm'], bins=range(1, 8), color='salmon', edgecolor='black', alpha=0.7)
plt.title('Histogram of priority_llm')
plt.xlabel('Grade')
plt.ylabel('Frequency')
plt.xticks(range(1, 6))

# Histogram for judge_grade
plt.subplot(3, 2, 4)
plt.hist(df_reports_overall['judge_grade'], bins=range(1, 8), color='plum', edgecolor='black', alpha=0.7, label='Judge')
plt.title('Histogram of judge_grade')
plt.xlabel('Grade')
plt.ylabel('Frequency')
plt.xticks(range(1, 6))

# Overlay histogram of all three grades
plt.subplot(3, 2, 5)
plt.hist(df_reports_overall['priority_manual'], bins=range(1, 8), color='skyblue', edgecolor='black', alpha=0.5, label='Manual')
plt.hist(df_reports_overall['priority_algo'], bins=range(1, 8), color='lightgreen', edgecolor='black', alpha=0.5, label='Algorithm')
plt.hist(df_reports_overall['priority_llm'], bins=range(1, 8), color='salmon', edgecolor='black', alpha=0.5, label='LLM')
plt.title('Overlay Histogram of All Grades')
plt.xlabel('Grade')
plt.ylabel('Frequency')
plt.xticks(range(1, 6))
plt.legend()

plt.subplot(3, 2, 6)
plt.hist(df_reports_overall['priority_manual'], bins=range(1, 8), color='skyblue', edgecolor='black', alpha=0.5, label='Manual')
plt.hist(df_reports_overall['priority_algo'], bins=range(1, 8), color='lightgreen', edgecolor='black', alpha=0.5, label='Algorithm')
plt.hist(df_reports_overall['priority_llm'], bins=range(1, 8), color='salmon', edgecolor='black', alpha=0.5, label='LLM')
plt.hist(df_reports_overall['judge_grade'], bins=range(1, 8), color='plum', edgecolor='black', alpha=0.7, label='Judge')
plt.title('Overlay Histogram of All Grades + Judged')
plt.xlabel('Grade')
plt.ylabel('Frequency')
plt.xticks(range(1, 6))
plt.legend()

plt.tight_layout()
plt.show()

# Visualization of results
plt.figure(figsize=(15, 15))

# 1. Distribution of grades
plt.subplot(3, 2, 1)
df_reports_overall[['priority_manual', 'priority_algo', 'priority_llm', 'judge_grade']].boxplot()
plt.title('Distribution of Grades')
plt.ylabel('Priority Grade')
plt.grid(False)

# 2. Confusion matrices
# Helper function to plot confusion matrix
def plot_confusion_matrix(ax, y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred, labels=range(1, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted Grade')
    ax.set_ylabel('Manual Grade')
    ax.set_xticks(np.arange(5) + 0.5, range(1, 6))
    ax.set_yticks(np.arange(5) + 0.5, range(1, 6))
    ax.set_xticklabels(range(1, 6))
    ax.set_yticklabels(range(1, 6))

plt.subplot(3, 2, 3)
plot_confusion_matrix(plt.gca(), df_reports_overall['priority_manual'], df_reports_overall['priority_algo'], 'Confusion Matrix - Algorithmic Approach')

plt.subplot(3, 2, 4)
plot_confusion_matrix(plt.gca(), df_reports_overall['priority_manual'], df_reports_overall['priority_llm'], 'Confusion Matrix - LLM Approach')

# 3. Error distribution
plt.subplot(3, 2, 2)
error_algo = df_reports_overall['priority_algo'] - df_reports_overall['priority_manual']
error_llm = df_reports_overall['priority_llm'] - df_reports_overall['priority_manual']
plt.hist(error_algo, alpha=0.5, bins=np.arange(-5.5, 6.5, 1), label='Algorithmic')
plt.hist(error_llm, alpha=0.5, bins=np.arange(-5.5, 6.5, 1), label='LLM')
plt.xlabel('Error (Predicted - Manual)')
plt.ylabel('Frequency')
plt.title('Error Distribution')
plt.legend()
plt.grid(False)

plt.subplot(3, 2, 5)
confusion_matrix_judge = confusion_matrix(df_reports_overall['priority_manual'], df_reports_overall['judge_grade'], labels=range(1, 7))
sns.heatmap(confusion_matrix_judge, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - Judge Approach')
plt.xlabel('Predicted Grade (Judge)')
plt.ylabel('Manual Grade')
plt.xticks(np.arange(5) + 0.5, range(1, 6))
plt.yticks(np.arange(5) + 0.5, range(1, 6))

plt.tight_layout()
plt.show()

# Confusion matrix for grouped grades
confusion_matrix_algo = confusion_matrix(df_reports_overall['priority_manual_grouped'], df_reports_overall['priority_algo_grouped'])
confusion_matrix_llm = confusion_matrix(df_reports_overall['priority_manual_grouped'], df_reports_overall['priority_llm_grouped'])
confusion_matrix_judge = confusion_matrix(df_reports_overall['priority_manual_grouped'], df_reports_overall['judge_grade_grouped'])

plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
sns.heatmap(confusion_matrix_algo, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Algorithm Approach')
plt.xlabel('Predicted Grade (Grouped)')
plt.ylabel('Manual Grade (Grouped)')

plt.subplot(2, 2, 2)
sns.heatmap(confusion_matrix_llm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - LLM Approach')
plt.xlabel('Predicted Grade (Grouped)')
plt.ylabel('Manual Grade (Grouped)')

plt.subplot(2, 2, 3)
sns.heatmap(confusion_matrix_judge, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Judge Approach')
plt.xlabel('Predicted Grade (Grouped)')
plt.ylabel('Manual Grade (Grouped)')

plt.tight_layout()
plt.show()

# Calculate Pearson correlation coefficients
pearson_algo, _ = pearsonr(df_reports_overall['priority_manual'], df_reports_overall['priority_algo'])
pearson_llm, _ = pearsonr(df_reports_overall['priority_manual'], df_reports_overall['priority_llm'])

# Scatter plots to visualize correlation
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.scatter(df_reports_overall['priority_manual'], df_reports_overall['priority_algo'], alpha=0.6)
plt.plot([1, 5], [1, 5], 'r--')  # Diagonal line representing perfect prediction
plt.xlabel('Manual Grade')
plt.ylabel('Algorithmic Grade')
plt.title(f'Manual vs Algorithmic (r={pearson_algo:.2f})')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.scatter(df_reports_overall['priority_manual'], df_reports_overall['priority_llm'], alpha=0.6)
plt.plot([1, 5], [1, 5], 'r--')  # Diagonal line representing perfect prediction
plt.xlabel('Manual Grade')
plt.ylabel('LLM Grade')
plt.title(f'Manual vs LLM (r={pearson_llm:.2f})')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.scatter(df_reports_overall['priority_manual'], df_reports_overall['judge_grade'], alpha=0.6, color='purple')
plt.plot([1, 5], [1, 5], 'r--')  # Diagonal line representing perfect prediction
plt.xlabel('Manual Grade')
plt.ylabel('Judge Grade')
plt.title('Manual vs Judge')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#%%
# Token counting
import tiktoken

def count_tokens(prompt, model="gpt-3.5-turbo"):
    # Retrieve the encoding for the model. Adjust the model name if needed.
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(prompt)
    return len(tokens)

print('The semi-algorithmic approach uses ~1500 tokens')
print('The LLM approach uses ~650 tokens')
# %%