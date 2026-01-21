# Code pour une application Streamlit de scoring de crédit
# Auteur : Lucas BELOIN, Mandir DIOP, Youssouf GAYE




# Pour lancer l'appliceation, utilise la commande : streamlit run app.py sur le terminal 

# Importation des bibliothèques nécessaires
import xgboost as xgb  # Ajout indispensable
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- 1. CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Dashboard Scoring Crédit")

st.title("Outil d'Octroi de Crédit - Prêt à dépenser")
st.markdown("Ce dashboard permet aux conseillers bancaires d'évaluer le risque client.")

# --- 2. CHARGEMENT DES FICHIERS CLÉS ---
@st.cache_resource
def load_data():
    # Chargement du modèle
    with open('best_model_xgboost.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Chargement du seuil
    try:
        with open('best_threshold.pkl', 'rb') as f:
            threshold = pickle.load(f)
    except:
        threshold = 0.53
    
    # Chargement des noms de colonnes
    with open('features_names.pkl', 'rb') as f:
        features = pickle.load(f)
        
    # Chargement d'un échantillon de données
    data = pd.read_csv('application_train-LFS.txt', nrows=1000)
    
    return model, threshold, features, data

model, threshold, feature_names, df_sample = load_data()

st.sidebar.header("Sélection du dossier")

# --- 3. SAISIE DES INFORMATIONS ---
id_client = st.sidebar.selectbox("Choisir un ID Client (Simulation)", df_sample['SK_ID_CURR'].unique())

st.subheader(f"Dossier Client : {id_client}")

client_data_raw = df_sample[df_sample['SK_ID_CURR'] == id_client].iloc[0]

# --- FORMULAIRE DE SAISIE ---
st.write("### Modification des informations clés")
col1, col2 = st.columns(2)

with col1:
    val_ext3 = client_data_raw['EXT_SOURCE_3'] if 'EXT_SOURCE_3' in client_data_raw else 0.5
    input_ext3 = st.slider("Score Externe 3 (Normalisé)", 0.0, 1.0, float(val_ext3))
    
    val_ext2 = client_data_raw['EXT_SOURCE_2'] if 'EXT_SOURCE_2' in client_data_raw else 0.5
    input_ext2 = st.slider("Score Externe 2 (Normalisé)", 0.0, 1.0, float(val_ext2))

    val_age = client_data_raw['DAYS_BIRTH'] / -365 if 'DAYS_BIRTH' in client_data_raw else 40
    input_age = st.number_input("Âge du client (Années)", 20, 70, int(val_age))

with col2:
    val_emp = client_data_raw['DAYS_EMPLOYED'] / -365 if 'DAYS_EMPLOYED' in client_data_raw else 5
    if val_emp < 0: val_emp = 0
    input_emp = st.number_input("Années d'ancienneté emploi", 0, 50, int(val_emp))
    
    val_gender = 1 if 'CODE_GENDER' in client_data_raw and client_data_raw['CODE_GENDER'] == 'M' else 0
    input_gender = st.selectbox("Genre (0=F, 1=M)", [0, 1], index=val_gender)

# --- 4. PRÉPARATION DES DONNÉES POUR LE MODÈLE ---
input_df = pd.DataFrame(0, index=[0], columns=feature_names)

if 'EXT_SOURCE_3' in feature_names: input_df['EXT_SOURCE_3'] = input_ext3
if 'EXT_SOURCE_2' in feature_names: input_df['EXT_SOURCE_2'] = input_ext2
if 'DAYS_BIRTH' in feature_names: input_df['DAYS_BIRTH'] = -input_age * 365
if 'DAYS_EMPLOYED' in feature_names: input_df['DAYS_EMPLOYED'] = -input_emp * 365
if 'CODE_GENDER_M' in feature_names: input_df['CODE_GENDER_M'] = input_gender

# --- 5. LE BOUTON PRÉDICTION ---
if st.button("Lancer l'analyse du risque", type="primary"):
    
    # Prédiction
    probability = model.predict_proba(input_df)[:, 1][0]
    
    st.divider()
    st.write("### Résultat de l'analyse IA")
    
    # Affichage Jauge
    st.progress(int(probability * 100))
    st.write(f"Probabilité de défaut calculée : **{probability:.1%}**")
    st.write(f"Seuil de risque accepté : **{threshold:.1%}**")
    
    # Décision Finale
    if probability > threshold:
        st.error(f"CRÉDIT REFUSÉ")
        st.write("Le risque de défaut est trop élevé selon les critères de la banque.")
    else:
        st.success(f"CRÉDIT ACCORDÉ")
        st.write("Le client présente un profil fiable.")

    # Explication

    st.info("Note : Cette décision est basée sur le modèle XGBoost optimisé.")
