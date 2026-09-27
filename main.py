#Part A
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("credit_risk_dataset.csv")

print(df.shape)
print(df.columns)
print(df.info())
print(df.describe())
print(df.isna().sum())

#Part B

#1. Target variable distributions
status_counts = df['loan_status'].value_counts()
print(status_counts)

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(status_counts.index.astype(str), status_counts.values, color=["green", "red"])
ax.set_xlabel('Loan Status')
ax.set_ylabel('Number of Applicants')
ax.set_title('Loan Status Distribution (0 = Low Risk, 1 = High Risk)')
# Interpretation: The classes are imbalanced – 25,473 applicants (78.2%) are
# low-risk vs 7,108 (21.8%) high-risk. This is exactly why the train-test split
# later uses stratify=y, and why accuracy alone is not a reliable metric here –
# recall/F1 matter more, which is why Part F tunes GridSearchCV on scoring='f1'.
plt.tight_layout()
#plt.savefig('loan_status_distribution.png', dpi=300)
plt.show()

#2. Correlation heatmap
numeric_df = df.select_dtypes(include="number")
corr_matrix = numeric_df.corr()

fig, ax = plt.subplots(figsize=(8, 6))
heatmap = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(heatmap, ax=ax, label='Correlation')
ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=90)
ax.set_yticks(range(len(corr_matrix.columns)))
ax.set_yticklabels(corr_matrix.columns)
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha='center', va='center', fontsize=8)
ax.set_title("Correlation Heatmap of Numeric Features")
# Interpretation: loan_percent_income (0.38) and loan_int_rate (0.34) have the
# strongest positive correlation with loan_status – applicants using a bigger
# share of their income for the loan, or paying a higher interest rate, are
# more likely to be high-risk. person_income has the strongest negative
# correlation (-0.14): higher earners are somewhat less likely to be high-risk.
plt.tight_layout()
#plt.savefig('correlation_heatmap.png', dpi=300)
plt.show()

#3. Boxplots for outliers
outlier_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt']

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, col in zip(axes, outlier_cols):
    ax.boxplot(df[col].dropna())
    ax.set_title(col)
plt.tight_layout()
#plt.savefig('outlier_boxplots.png', dpi=300)
plt.show()
# Interpretation: person_age and person_emp_length both show extreme outliers.
# person_age has values well above 100, and person_emp_length has values that
# exceed a realistic working lifetime (max was 123). These look like data-entry
# errors, which is exactly why Part C caps them at 100 and 50 respectively.

#4. Histograms of selected numerical features
hist_cols = ['person_income', 'loan_amnt', 'loan_int_rate']

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, hist_cols):
    ax.hist(df[col].dropna(), bins=30, color='steelblue', edgecolor='black')
    ax.set_title(col)
plt.tight_layout()
#plt.savefig('feature_histograms.png', dpi=300)
plt.show()
# Interpretation: person_income and loan_amnt are both right-skewed with a long
# tail of higher values – typical for income and loan-size data. This skew is
# part of the justification for using median (not mean) imputation and for the
# 99th-percentile cap on income in Part C.

# ============================================================
# PART C – Data Cleaning and Preprocessing
# ============================================================

# ----- C1 Missing values (justification in comments) -----
# person_emp_length : 895 missing, right-skewed, max=123 (impossible).
#   Median is robust → use median.
# loan_int_rate     : 3116 missing, mildly skewed.
#   Median keeps the fill realistic and consistent.
print("=" * 60)
print("PART C – Missing values BEFORE imputation")
print("=" * 60)
print(df.isna().sum())

imputer = SimpleImputer(strategy='median')
df[['person_emp_length', 'loan_int_rate']] = imputer.fit_transform(
    df[['person_emp_length', 'loan_int_rate']]
)

print("\nMissing values AFTER imputation")
print(df.isna().sum())

# ----- C2 Remove duplicates -----
print(f"\nShape before dropping duplicates: {df.shape}")
df = df.drop_duplicates()
print(f"Shape after dropping duplicates:  {df.shape}")
# 165 exact duplicates removed

# ----- C3 Treat outliers -----
print("\nOutlier check BEFORE treatment:")
print("person_emp_length max:", df['person_emp_length'].max())
print("person_age max       :", df['person_age'].max())
print("person_income max    :", df['person_income'].max())

# Cap impossible employment length
df['person_emp_length'] = df['person_emp_length'].clip(upper=50)

# Cap unrealistic age
df['person_age'] = df['person_age'].clip(upper=100)

# Soft 99th-percentile cap on income
income_cap = df['person_income'].quantile(0.99)
df['person_income'] = df['person_income'].clip(upper=income_cap)

print("\nAfter outlier treatment:")
print("person_emp_length max:", df['person_emp_length'].max())
print("person_age max       :", df['person_age'].max())
print("person_income max    :", df['person_income'].max())

# ----- C4 Separate features and target -----
X = df.drop('loan_status', axis=1)
y = df['loan_status']

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'string', 'str']).columns.tolist()

print("\nNumeric features  :", numeric_features)
print("Categorical features:", categorical_features)

# ----- C5 Encode categoricals (One-Hot) -----
# Using OneHotEncoder so the model never assumes order in loan_grade etc.
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_cat_encoded = encoder.fit_transform(X[categorical_features])
cat_column_names = encoder.get_feature_names_out(categorical_features)

X_cat_df = pd.DataFrame(X_cat_encoded, columns=cat_column_names, index=X.index)

# Combine numeric + encoded categorical
X_processed = pd.concat([X[numeric_features], X_cat_df], axis=1)

print(f"\nShape after encoding: {X_processed.shape}")

# ----- C6 80/20 stratified split -----
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y,
    test_size=0.20,
    random_state=42,
    stratify=y          # keeps the ~78/22 balance
)

print(f"\nTrain shape: {X_train.shape}")
print(f"Test shape : {X_test.shape}")
print("Class balance in train:\n", y_train.value_counts(normalize=True).round(3))
print("Class balance in test :\n", y_test.value_counts(normalize=True).round(3))

# ----- C7 Feature scaling (fit on TRAIN only) -----
# Why scaling is necessary:
# Logistic Regression and KNN both rely on distances or gradients that are
# sensitive to a feature’s raw numeric range. Without scaling, a feature like
# person_income (tens of thousands) would completely dominate a feature like
# loan_percent_income (a 0–1 ratio), even if the smaller-range feature is
# actually more predictive. StandardScaler puts every feature on the same
# footing (mean 0, std 1) so the model weighs them by actual predictive value,
# not by coincidence of scale. Decision trees don’t need this (they split on
# thresholds per feature), but it doesn’t hurt them either, which is why the
# same scaled data is reused for all three models.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\nScaled train shape: {X_train_scaled.shape}")
print(f"Scaled test shape : {X_test_scaled.shape}")

# ============================================================
# PART D – Principal Component Analysis (PCA)
# ============================================================

pca = PCA()
X_train_pca_full = pca.fit_transform(X_train_scaled)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("=" * 60)
print("PART D – Explained Variance Ratio (first 15 components)")
print("=" * 60)
for i, (ev, cum) in enumerate(zip(explained_variance[:15], cumulative_variance[:15]), 1):
    print(f"PC{i:2d}: {ev:.4f}   cumulative: {cum:.4f}")

n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"\nComponents for ≥90% variance: {n_components_90}")
print(f"Components for ≥95% variance: {n_components_95}")

n_components = n_components_90
print(f"\nSelected number of components: {n_components}")

pca_final = PCA(n_components=n_components)
X_train_pca = pca_final.fit_transform(X_train_scaled)
X_test_pca  = pca_final.transform(X_test_scaled)

print(f"PCA train shape: {X_train_pca.shape}")
print(f"PCA test shape : {X_test_pca.shape}")

# Cumulative variance plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o', linestyle='-')
ax.axhline(y=0.90, color='r', linestyle='--', label='90% threshold')
ax.axhline(y=0.95, color='g', linestyle='--', label='95% threshold')
ax.set_xlabel('Number of Principal Components')
ax.set_ylabel('Cumulative Explained Variance')
ax.set_title('PCA – Cumulative Explained Variance')
ax.legend()
plt.tight_layout()
#plt.savefig('pca_cumulative_variance.png', dpi=300)
plt.show()

# We selected the number of components that capture at least 90 % of the
# total variance (17 components). 90 % is a common practical threshold:
# it keeps most of the information while still reducing dimensionality
# from 26 features down to 17. The cumulative variance plot confirms the
# curve flattens after this point, so additional components add very little
# extra information. PCA is fitted only on the training data and then
# applied to the test set to avoid data leakage.

# ============================================================
# PART E – Model Development (Original features + PCA features)
# ============================================================

def evaluate_model(model, X_tr, X_te, y_tr, y_te, model_name):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    print(f"\n{'=' * 60}")
    print(f"RESULTS – {model_name}")
    print(f"{'=' * 60}")
    print(f"Accuracy : {accuracy_score(y_te, y_pred):.4f}")
    print(f"Precision: {precision_score(y_te, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_te, y_pred):.4f}")
    print(f"F1-score : {f1_score(y_te, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_te, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_te, y_pred, target_names=['Low Risk (0)', 'High Risk (1)']))
    return model

print("\n" + "#" * 70)
print("PART E – MODEL DEVELOPMENT (ORIGINAL + PCA)")
print("#" * 70)

print("\n" + "#" * 70)
print("MODELS TRAINED ON ORIGINAL SCALED FEATURES")
print("#" * 70)

lr_orig = evaluate_model(LogisticRegression(max_iter=1000, random_state=42),
                         X_train_scaled, X_test_scaled, y_train, y_test,
                         "Logistic Regression (Original)")

knn_orig = evaluate_model(KNeighborsClassifier(n_neighbors=5),
                          X_train_scaled, X_test_scaled, y_train, y_test,
                          "K-Nearest Neighbours (Original)")

dt_orig = evaluate_model(DecisionTreeClassifier(random_state=42),
                         X_train_scaled, X_test_scaled, y_train, y_test,
                         "Decision Tree (Original)")

print("\n" + "#" * 70)
print("MODELS TRAINED ON PCA FEATURES")
print("#" * 70)

lr_pca = evaluate_model(LogisticRegression(max_iter=1000, random_state=42),
                        X_train_pca, X_test_pca, y_train, y_test,
                        "Logistic Regression (PCA)")

knn_pca = evaluate_model(KNeighborsClassifier(n_neighbors=5),
                         X_train_pca, X_test_pca, y_train, y_test,
                         "K-Nearest Neighbours (PCA)")

dt_pca = evaluate_model(DecisionTreeClassifier(random_state=42),
                        X_train_pca, X_test_pca, y_train, y_test,
                        "Decision Tree (PCA)")

# Best model overall: Decision Tree on the ORIGINAL scaled features.
# It achieved the highest Recall (0.774) and F1 (0.755) on the high-risk class.
#
# Why this matters for credit risk:
# - False Negative (FN) = high-risk applicant labelled low-risk
#   → bank loses money when they default. Most expensive error.
# - False Positive (FP) = low-risk applicant labelled high-risk
#   → bank loses a potential good customer, but lower cost.
#
# Therefore we prioritise Recall (and F1) over pure Accuracy.
# Decision Tree wins because it captures non-linear feature interactions
# without needing PCA. KNN improves with PCA on Accuracy, but its
# high-risk Recall stays lower than the Tree.

# ============================================================
# PART F – Model Optimisation
# ============================================================

print("\n" + "#" * 70)
print("PART F – Hyperparameter Tuning (GridSearchCV)")
print("#" * 70)

param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10]
}

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

grid.fit(X_train_scaled, y_train)

print("Best parameters:", grid.best_params_)
print("Best cross-validation F1:", grid.best_score_)

best_model = grid.best_estimator_
y_pred_tuned = best_model.predict(X_test_scaled)

print("\nTuned model performance on Test set:")
print(f"Accuracy : {accuracy_score(y_test, y_pred_tuned):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_tuned):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred_tuned):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred_tuned):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_tuned))
print("\nClassification Report:")
print(classification_report(y_pred_tuned, y_test, target_names=['Low Risk (0)', 'High Risk (1)']))

# GridSearchCV tested Decision Tree complexity:
# - Smaller max_depth / larger min_samples_split → simpler tree
#   → higher bias, lower variance (may underfit).
# - Larger max_depth / smaller min_samples_split → more complex tree
#   → lower bias, higher variance (may overfit).
# GridSearchCV picks the combination with the best cross-validated F1
# so the tree is complex enough to catch high-risk applicants, but not
# so deep that it just memorises the training set.
