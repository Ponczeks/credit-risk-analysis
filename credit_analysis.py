import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Wczytanie danych
df = pd.read_csv('credit_data.csv')  

# --- 1. Usuwanie outlierów ---
# Usuwamy rekordy z Credit Amount > 10,000
df_clean = df[df['Credit Amount'] <= 10000].copy()
print(f"Liczba rekordów przed usunięciem outlierów: {len(df)}")
print(f"Liczba rekordów po usunięciu outlierów: {len(df_clean)}")

# --- 2. Feature Engineering ---
# Dodajemy nową cechę: Credit Amount / Duration jako obciążenie miesięczne
df_clean['Monthly Burden'] = df_clean['Credit Amount'] / df_clean['Duration of Credit (month)']

# --- 3. Eksploracja danych (EDA) ---
print("Podstawowe statystyki po czyszczeniu:\n", df_clean.describe())

# Rozkład Creditability
plt.figure(figsize=(8, 6))
sns.countplot(x='Creditability', data=df_clean)
plt.title('Rozkład klas Creditability (po usunięciu outlierów)')
plt.savefig('creditability_distribution_clean.png')
plt.show()

# Rozkłady kluczowych cech + nowa cecha
plt.figure(figsize=(20, 5))
for i, col in enumerate(['Age (years)', 'Credit Amount', 'Duration of Credit (month)', 'Monthly Burden'], 1):
    plt.subplot(1, 4, i)
    sns.histplot(data=df_clean, x=col, hue='Creditability', bins=20, kde=True)
    plt.title(f'Rozkład {col}')
plt.tight_layout()
plt.savefig('feature_distributions_clean.png')
plt.show()

# Boxploty
plt.figure(figsize=(20, 5))
for i, col in enumerate(['Age (years)', 'Credit Amount', 'Duration of Credit (month)', 'Monthly Burden'], 1):
    plt.subplot(1, 4, i)
    sns.boxplot(x='Creditability', y=col, data=df_clean)
    plt.title(f'{col} vs Creditability')
plt.tight_layout()
plt.savefig('boxplots_clean.png')
plt.show()

# Korelacje
plt.figure(figsize=(12, 10))
sns.heatmap(df_clean.corr(), annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Mapa korelacji cech (po czyszczeniu i feature engineeringu)')
plt.savefig('correlation_heatmap_clean.png')
plt.show()

# --- 4. Przygotowanie danych ---
X = df_clean.drop('Creditability', axis=1)
y = df_clean['Creditability']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# --- 5. Regresja logistyczna ---
log_model = LogisticRegression(max_iter=1000, class_weight='balanced')
log_model.fit(X_train, y_train)
y_pred_proba_log = log_model.predict_proba(X_test)[:, 1]
y_pred_log = log_model.predict(X_test)
auc_log = roc_auc_score(y_test, y_pred_proba_log)

print(f"Logistic Regression AUC: {auc_log:.2f}")
print("Confusion Matrix (Logistic):\n", confusion_matrix(y_test, y_pred_log))
print("Classification Report (Logistic):\n", classification_report(y_test, y_pred_log))

# Ważność cech (Logistic)
feature_importance_log = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': np.abs(log_model.coef_[0])
}).sort_values('Coefficient', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=feature_importance_log)
plt.title('Ważność cech w regresji logistycznej')
plt.savefig('logistic_feature_importance_clean.png')
plt.show()

# --- 6. Random Forest z tuningiem ---
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced']
}
rf_model = RandomForestClassifier(random_state=42)
rf_grid = GridSearchCV(rf_model, rf_param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
rf_grid.fit(X_train, y_train)

best_rf = rf_grid.best_estimator_
y_pred_proba_rf = best_rf.predict_proba(X_test)[:, 1]
y_pred_rf = best_rf.predict(X_test)
auc_rf = roc_auc_score(y_test, y_pred_proba_rf)

print("Najlepsze parametry RF:", rf_grid.best_params_)
print(f"Random Forest (tuned) AUC: {auc_rf:.2f}")
print("Confusion Matrix (RF tuned):\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report (RF tuned):\n", classification_report(y_test, y_pred_rf))

# Ważność cech (RF)
feature_importance_rf = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_rf.feature_importances_
}).sort_values('Importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_rf)
plt.title('Ważność cech w Random Forest')
plt.savefig('rf_feature_importance_clean.png')
plt.show()

# --- 7. XGBoost z tuningiem ---
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
    random_state=42,
    eval_metric='auc'
)
xgb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 1.0]
}
xgb_grid = GridSearchCV(xgb_model, xgb_param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
xgb_grid.fit(X_train, y_train)

best_xgb = xgb_grid.best_estimator_
y_pred_proba_xgb = best_xgb.predict_proba(X_test)[:, 1]
y_pred_xgb = best_xgb.predict(X_test)
auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)

print("Najlepsze parametry XGBoost:", xgb_grid.best_params_)
print(f"XGBoost AUC: {auc_xgb:.2f}")
print("Confusion Matrix (XGBoost):\n", confusion_matrix(y_test, y_pred_xgb))
print("Classification Report (XGBoost):\n", classification_report(y_test, y_pred_xgb))

# Ważność cech (XGBoost)
feature_importance_xgb = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_xgb.feature_importances_
}).sort_values('Importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_xgb)
plt.title('Ważność cech w XGBoost')
plt.savefig('xgb_feature_importance_clean.png')
plt.show()

# --- 8. Porównanie modeli ---
fpr_log, tpr_log, _ = roc_curve(y_test, y_pred_proba_log)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_proba_xgb)

plt.figure(figsize=(10, 8))
plt.plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC = {auc_log:.2f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.2f})')
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {auc_xgb:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Porównanie krzywych ROC (po czyszczeniu i feature engineeringu)')
plt.legend()
plt.savefig('roc_comparison_clean.png')
plt.show()

results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest (tuned)', 'XGBoost'],
    'AUC': [auc_log, auc_rf, auc_xgb],
    'Accuracy': [classification_report(y_test, y_pred_log, output_dict=True)['accuracy'],
                 classification_report(y_test, y_pred_rf, output_dict=True)['accuracy'],
                 classification_report(y_test, y_pred_xgb, output_dict=True)['accuracy']],
    'Recall (0)': [classification_report(y_test, y_pred_log, output_dict=True)['0']['recall'],
                   classification_report(y_test, y_pred_rf, output_dict=True)['0']['recall'],
                   classification_report(y_test, y_pred_xgb, output_dict=True)['0']['recall']],
    'Recall (1)': [classification_report(y_test, y_pred_log, output_dict=True)['1']['recall'],
                   classification_report(y_test, y_pred_rf, output_dict=True)['1']['recall'],
                   classification_report(y_test, y_pred_xgb, output_dict=True)['1']['recall']]
})
print("\nPorównanie modeli:\n", results)
