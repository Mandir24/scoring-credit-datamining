import pandas as pd
import numpy as np
import gc
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
from xgboost import XGBClassifier

# =============================================================================
# 1. FONCTIONS DE PRÉPARATION DES DONNÉES
# =============================================================================

def traiter_et_agreger(nom_fichier, prefixe):
    """
    Charge un fichier CSV, encode les variables catégorielles et agrège 
    les données par client (SK_ID_CURR).
    """
    print(f"Traitement de {nom_fichier}...")
    
    # Chargement des données
    df = pd.read_csv(nom_fichier)
    
    # Encodage One-Hot des variables catégorielles (Catégoriel -> Numérique)
    df = pd.get_dummies(df, dummy_na=True) 
    
    # Exclusion des identifiants pour l'agrégation
    cols_a_exclure = ['SK_ID_CURR', 'SK_ID_PREV']
    cols_a_agreger = [c for c in df.columns if c not in cols_a_exclure]
    
    # Calcul des statistiques (moyenne, min, max, somme, compte) par client
    agg = df.groupby('SK_ID_CURR')[cols_a_agreger].agg(['mean', 'min', 'max', 'sum', 'count'])
    
    # Renommer les colonnes avec le préfixe pour identifier la source des données
    agg.columns = [f'{prefixe}_{c[0]}_{c[1]}' for c in agg.columns]
    
    return agg

# =============================================================================
# 2. FUSION DES FICHIERS ET NETTOYAGE
# =============================================================================

print("--- PHASE DE PRÉPARATION DES DONNÉES ---")

# Chargement du fichier principal (le squelette du dataset)
df_final = pd.read_csv('application_train-LFS.txt')

# Liste des fichiers satellites à fusionner par SK_ID_CURR
fichiers = [
    ('bureau-LFS.txt', 'BUR'),
    ('previous_application-LFS.txt', 'PREV'),
    ('POS_CASH_balance-LFS.txt', 'POS'),
    ('installments_payments-LFS.txt', 'INST'),
    ('credit_card_balance-LFS.txt', 'CC')
]

for fichier, prefix in fichiers:
    df_agg = traiter_et_agreger(fichier, prefix)
    df_final = df_final.merge(df_agg, on='SK_ID_CURR', how='left')
    del df_agg
    gc.collect() # Libération de la mémoire vive

# Gestion des valeurs aberrantes et nettoyage
df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
df_final.fillna(0, inplace=True) # Gestion des valeurs manquantes

# Ingénierie des caractéristiques : Conversion des jours d'emploi en années
df_final['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
df_final['ANNEES_TRAVAIL'] = df_final['DAYS_EMPLOYED'] / -365

# Encodage final des variables du fichier principal
df_final = pd.get_dummies(df_final, dummy_na=True)

# =============================================================================
# 3. PRÉPARATION ET ENTRAÎNEMENT DU MODÈLE
# =============================================================================

# Définition de la cible (y) et des caractéristiques (X)
y = df_final['TARGET']
X = df_final.drop(['TARGET', 'SK_ID_CURR'], axis=1, errors='ignore')

# Sélection des 50 meilleures variables basées sur la corrélation avec la cible
print("Sélection des variables les plus significatives...")
correlations = {}
for col in X.columns:
    try:
        correlations[col] = abs(X[col].corr(y))
    except:
        correlations[col] = 0

top_50_cols = pd.Series(correlations).sort_values(ascending=False).head(50).index.tolist()
X = X[top_50_cols]

# Séparation en jeux d'entraînement et de test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Rééquilibrage manuel par sur-échantillonnage de la classe minoritaire (défauts)
X_train_0 = X_train[y_train == 0]
X_train_1 = X_train[y_train == 1]
ids_sup = np.random.choice(len(X_train_1), size=len(X_train_0), replace=True)
X_train_resampled = np.vstack((X_train_0.values, X_train_1.values[ids_sup]))
y_train_resampled = np.hstack((y_train[y_train == 0].values, y_train[y_train == 1].values[ids_sup]))

# Entraînement du modèle XGBoost
print("\n--- ENTRAÎNEMENT DU MODÈLE ---")
model = XGBClassifier(n_estimators=50, max_depth=5, eval_metric='logloss', n_jobs=-1, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

y_pred_proba = model.predict_proba(X_test)[:, 1]
print(f"Score AUC obtenu : {roc_auc_score(y_test, y_pred_proba):.4f}")

# =============================================================================
# 4. OPTIMISATION DU SEUIL ET SAUVEGARDE
# =============================================================================

# Optimisation du seuil métier pour minimiser les pertes bancaires
# Hypothèse : Un faux négatif (mauvais payeur accepté) coûte 10x plus qu'un faux positif
seuils = np.arange(0.1, 0.9, 0.01)
couts = []
for seuil in seuils:
    y_pred_temp = (y_pred_proba > seuil).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_temp).ravel()
    couts.append((fn * 10) + (fp * 1))

best_threshold = seuils[np.argmin(couts)]
print(f"Seuil de décision optimisé : {best_threshold:.2f}")

# Sauvegarde des fichiers pour l'application Streamlit
with open('best_model_xgboost.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('best_threshold.pkl', 'wb') as f:
    pickle.dump(best_threshold, f)
with open('features_names.pkl', 'wb') as f:
    pickle.dump(top_50_cols, f)

print("\n Fichiers sauvegardés. Prêt pour le déploiement.")