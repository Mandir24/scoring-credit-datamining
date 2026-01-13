# scoring-credit-datamining
Ce projet, réalisé dans le cadre du BUT Science des Données (IUT Caen Normandie), vise à prédire la probabilité de faillite d'un client pour aider une institution bancaire à accorder ou refuser un prêt.
# 📊 Scoring Crédit pour l'Inclusion Financière

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2C3E50?style=for-the-badge)

## 📌 Présentation du Projet
Ce projet a été réalisé dans le cadre du **BUT Science des Données** à l'IUT de Caen Normandie (2025-2026). 

L'objectif est de développer un outil d'aide à la décision pour une banque afin d'évaluer la solvabilité de clients ayant peu d'historique de crédit. Le projet combine une phase intensive de **Data Mining** (fusion de données complexes) et le déploiement d'une application web interactive.



## 🛠️ Problématique Technique
Le défi majeur de ce dataset est le **déséquilibre des classes** :
* **92%** de clients solvables (Classe 0)
* **8%** de clients en défaut de paiement (Classe 1)

Une modélisation standard ignorerait les cas de défaut. Nous avons donc mis en place des stratégies de rééchantillonnage (SMOTE/Oversampling) et d'optimisation de seuil métier.

## 🚀 Fonctionnalités
* **Pipeline de Données** : Nettoyage et fusion de sources multiples (Bureau, Prêts précédents, POS_CASH).
* **Modélisation Avancée** : Comparaison de modèles avec **XGBoost** comme modèle final.
* **Optimisation Métier** : Calcul d'un seuil de décision optimisé à **0.53** pour minimiser le risque financier.
* **Interface Streamlit** : Dashboard interactif permettant aux conseillers de tester des profils clients en temps réel.

## 📁 Structure du Dépôt
| Fichier | Description |
| :--- | :--- |
| `codedatamining.ipynb` | Notebook complet (EDA, Preprocessing, Modélisation). |
| `app.py` | Code source de l'application Streamlit. |
| `best_model_xgboost.pkl` | Modèle entraîné (Le "cerveau" de l'IA). |
| `features_names.pkl` | Liste des variables sélectionnées. |
| `best_threshold.pkl` | Le seuil de probabilité optimisé. |

## 💻 Installation
1. Clonez le dépôt :
   ```bash
   git clone [https://github.com/Mandir24/scoring-credit-datamining.git](https://github.com/Mandir24/scoring-credit-datamining.git)
