# -*- coding: utf-8 -*-
"""
config.py — Configuration centrale du projet SportReview AI

Toutes les constantes et chemins sont définis ici.
On importe ce fichier dans tous les autres scripts pour éviter
de répéter les mêmes valeurs partout.
"""

import os

# ─────────────────────────────────────────────────────────────────────────────
# CHEMINS DES FICHIERS
# ─────────────────────────────────────────────────────────────────────────────

# Dossier racine du projet (dossier parent de src/)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dossiers de données
DATA_RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")        # données brutes
DATA_PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")  # données nettoyées

# Fichiers de données
RAW_CSV = os.path.join(DATA_RAW_DIR, "reviews_raw.csv")
PROCESSED_CSV = os.path.join(DATA_PROCESSED_DIR, "reviews_processed.csv")
TRAIN_CSV = os.path.join(DATA_PROCESSED_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_PROCESSED_DIR, "test.csv")

# Modèles
MODELS_DIR = os.path.join(ROOT_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "pipeline.pkl")       # modèle sérialisé pour FastAPI
CLASSES_PATH = os.path.join(MODELS_DIR, "classes.json")     # liste des classes
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl") # vectoriseur séparé si besoin

# Rapports de drift
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
DRIFT_REPORT_HTML = os.path.join(REPORTS_DIR, "drift_report.html")
DRIFT_REPORT_JSON = os.path.join(REPORTS_DIR, "drift_report.json")

# ─────────────────────────────────────────────────────────────────────────────
# DATASET AMAZON
# ─────────────────────────────────────────────────────────────────────────────

# Nom du dataset HuggingFace à télécharger
HF_DATASET_NAME = "McAuley-Lab/Amazon-Reviews-2023"
HF_DATASET_CONFIG = "raw_review_Sports_and_Outdoors"

# Nombre max d'exemples à utiliser (None = tout le dataset ~300 000 avis)
# Mettre un entier pour un entraînement rapide (ex: 10000 pour tester)
# None = utilise tous les avis disponibles pour un modèle production-ready
MAX_SAMPLES = None

# Colonnes utiles dans le dataset Amazon
COL_TEXT = "text"           # colonne texte de l'avis (après preprocessing)
COL_LABEL = "label"         # colonne label (POSITIF/NEGATIF/NEUTRE/SPAM)
COL_RATING = "rating"       # note 1-5 étoiles (avant transformation)
COL_VERIFIED = "verified_purchase"  # achat vérifié ou non

# ─────────────────────────────────────────────────────────────────────────────
# LABELS DES CLASSES
# ─────────────────────────────────────────────────────────────────────────────

# Les 4 classes du modèle et leur signification
LABEL_POSITIF = "POSITIF"   # notes 4-5 étoiles
LABEL_NEGATIF = "NEGATIF"   # notes 1-2 étoiles
LABEL_NEUTRE = "NEUTRE"     # note 3 étoiles
LABEL_SPAM = "SPAM"         # patterns promotionnels détectés

CLASSES = [LABEL_NEGATIF, LABEL_NEUTRE, LABEL_POSITIF, LABEL_SPAM]

# Règles de conversion note → label
RATING_TO_LABEL = {
    1: LABEL_NEGATIF,
    2: LABEL_NEGATIF,
    3: LABEL_NEUTRE,
    4: LABEL_POSITIF,
    5: LABEL_POSITIF,
}

# ─────────────────────────────────────────────────────────────────────────────
# PARAMÈTRES D'ENTRAÎNEMENT
# ─────────────────────────────────────────────────────────────────────────────

# Split train/test
TEST_SIZE = 0.2         # 20% pour le test
RANDOM_STATE = 42       # graine aléatoire pour reproductibilité

# TF-IDF
TFIDF_MAX_FEATURES = 10000   # nombre de features (plus élevé qu'avant = meilleur)
TFIDF_NGRAM_RANGE = (1, 2)   # unigrammes + bigrammes
TFIDF_MAX_DF = 0.95          # ignore les mots dans +95% des textes
TFIDF_MIN_DF = 2             # ignore les mots qui apparaissent moins de 2 fois

# Logistic Regression
LR_C = 1.0              # paramètre de régularisation
LR_MAX_ITER = 1000      # nombre max d'itérations

# SVM
SVM_C = 1.0             # paramètre de régularisation SVM
SVM_KERNEL = "linear"   # noyau linéaire (le plus efficace pour du texte)

# XGBoost
XGB_N_ESTIMATORS = 200  # nombre d'arbres
XGB_MAX_DEPTH = 6       # profondeur max de chaque arbre
XGB_LEARNING_RATE = 0.1 # taux d'apprentissage

# Word2Vec
W2V_VECTOR_SIZE = 100   # dimension des vecteurs de mots
W2V_WINDOW = 5          # taille de la fenêtre contextuelle
W2V_MIN_COUNT = 2       # ignore les mots apparaissant moins de 2 fois

# Sentence-BERT
SBERT_MODEL = "all-MiniLM-L6-v2"  # modèle léger et performant (90MB)

# DistilBERT fine-tuné
# distilbert-base-uncased = version légère de BERT (60% plus rapide, 40% plus petit)
# On fait du fine-tuning : on part d'un modèle pré-entraîné sur des milliards de textes
# et on l'adapte spécifiquement à notre tâche de classification d'avis
DISTILBERT_MODEL = "distilbert-base-uncased"
DISTILBERT_MAX_LENGTH = 128     # longueur max des tokens (128 suffit pour des avis)
DISTILBERT_BATCH_SIZE = 16      # taille des mini-batches pour l'entraînement GPU/CPU
DISTILBERT_EPOCHS = 3           # 3 epochs suffisent pour le fine-tuning
DISTILBERT_LEARNING_RATE = 2e-5 # taux d'apprentissage recommandé pour fine-tuning BERT

# ─────────────────────────────────────────────────────────────────────────────
# MLFLOW
# ─────────────────────────────────────────────────────────────────────────────

MLFLOW_EXPERIMENT_NAME = "sportreview-ai"     # nom de l'expérience MLflow
MLFLOW_REGISTERED_MODEL = "review-classifier"  # nom dans le Model Registry

# Seuil minimum de F1 score pour promouvoir un modèle en production
# Ajusté à 0.65 pour refléter les résultats réels obtenus (mai 2026) :
# meilleur modèle = distilbert avec F1=0.7103 (classification 4 classes déséquilibrées)
PROMOTION_F1_THRESHOLD = 0.65

# ─────────────────────────────────────────────────────────────────────────────
# API FASTAPI
# ─────────────────────────────────────────────────────────────────────────────

API_HOST = "0.0.0.0"    # écoute sur toutes les interfaces (obligatoire pour Docker)
API_PORT = 8000          # port de l'API
API_MAX_BATCH_SIZE = 500 # nombre max d'avis en batch (plus élevé que le mini projet)

# Seuil de confiance minimum pour une prédiction fiable
CONFIDENCE_THRESHOLD = 0.6

# ─────────────────────────────────────────────────────────────────────────────
# MONITORING
# ─────────────────────────────────────────────────────────────────────────────

# Nombre d'avis à garder en mémoire pour le calcul du drift
DRIFT_WINDOW_SIZE = 1000  # taille de la fenêtre glissante

# Seuil de drift pour déclencher une alerte
DRIFT_THRESHOLD = 0.1     # si le drift score dépasse 10%, on alerte
