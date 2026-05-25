# -*- coding: utf-8 -*-
"""
train.py — Entraînement de tous les modèles avec MLflow Tracking

Ce script entraîne, évalue et compare 6 modèles de classification de texte,
en utilisant MLflow pour tracer et comparer tous les résultats.

═══════════════════════════════════════════════════════════════════════════════
LES 6 MODÈLES ENTRAÎNÉS
═══════════════════════════════════════════════════════════════════════════════

  Modèle 1 — tfidf_lr   : TF-IDF + Logistic Regression
    → Le modèle "baseline" (référence). Rapide, interprétable.
    → Gestion déséquilibre : class_weight="balanced"
    → Temps estimé : 2-3 minutes

  Modèle 2 — tfidf_svm  : TF-IDF + SVM (Support Vector Machine)
    → Excellent sur données textuelles, légèrement meilleur que LR sur les marges
    → Gestion déséquilibre : class_weight="balanced"
    → Temps estimé : 3-5 minutes

  Modèle 3 — tfidf_xgb  : TF-IDF + XGBoost (Gradient Boosting)
    → Très puissant, utilise aussi les features numériques (text_length, etc.)
    → Gestion déséquilibre : scale_pos_weight calculé automatiquement
    → Temps estimé : 5-10 minutes

  Modèle 4 — w2v_lr     : Word2Vec + Logistic Regression
    → Comprend le contexte sémantique ("chaussures" proche de "baskets")
    → Gestion déséquilibre : SMOTE (génère des exemples synthétiques)
    → Temps estimé : 10-15 minutes

  Modèle 5 — sbert_lr   : Sentence-BERT + Logistic Regression
    → Embeddings de phrases de très haute qualité (all-MiniLM-L6-v2)
    → Gestion déséquilibre : ADASYN (adaptatif, plus intelligent que SMOTE)
    → Temps estimé : 15-30 minutes

  Modèle 6 — distilbert : DistilBERT fine-tuné (Transformers + PyTorch)
    → Le plus performant. Fine-tuning d'un LLM pré-entraîné sur 300k avis
    → Gestion déséquilibre : weighted cross-entropy loss
    → Temps estimé : 30-90 minutes (selon GPU/CPU)

═══════════════════════════════════════════════════════════════════════════════
GESTION DU DÉSÉQUILIBRE DE CLASSES
═══════════════════════════════════════════════════════════════════════════════

Le problème : dans le dataset, on a ~63% POSITIF mais seulement ~6% SPAM.
Un modèle naïf va apprendre à prédire POSITIF tout le temps car c'est la classe
dominante → il aura 63% d'accuracy sans rien apprendre d'utile.

Les 4 techniques utilisées :

  1. class_weight="balanced" (LR, SVM)
     → Pénalise plus fortement les erreurs sur les classes rares
     → Automatique, aucun impact sur les données d'entraînement
     → Formule : poids(classe) = n_total / (n_classes × n_exemples_classe)

  2. SMOTE — Synthetic Minority Over-sampling Technique (Word2Vec + LR)
     → Génère de NOUVEAUX exemples synthétiques pour les classes minoritaires
     → Interpole entre exemples proches dans l'espace des features
     → Résultat : toutes les classes ont le même nombre d'exemples

  3. ADASYN — Adaptive Synthetic Sampling (SBERT + LR)
     → Variante améliorée de SMOTE : génère plus d'exemples là où le modèle
       commet le plus d'erreurs (zones de décision difficiles)
     → Meilleur que SMOTE sur des frontières complexes

  4. Weighted Loss (DistilBERT)
     → Poids inversement proportionnels à la fréquence des classes
     → Intégré directement dans la fonction de perte PyTorch

═══════════════════════════════════════════════════════════════════════════════

Usage :
    python src/train.py                           # entraîne tous les modèles
    python src/train.py --models tfidf_lr         # entraîne seulement le baseline
    python src/train.py --models tfidf_lr tfidf_svm  # entraîne 2 modèles
    python src/train.py --fast                    # mode rapide (dataset réduit)
    mlflow ui                                     # visualise les résultats
"""

import argparse
import os
# Fix conflit DLL OpenMP entre NumPy/MKL et PyTorch sur Windows.
# NumPy charge libiomp5md.dll (Intel MKL), PyTorch charge son propre OpenMP.
# Sans cette variable, c10.dll échoue avec WinError 1114 quand importé après numpy.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import pickle
import json
import warnings
from typing import Any, Dict, List, Optional, Tuple

# torch doit être importé AVANT numpy sur Windows pour éviter le conflit de DLL :
# numpy charge Intel MKL (libiomp5md.dll) qui entre en conflit avec c10.dll de PyTorch
# si numpy est chargé en premier. En chargeant torch d'abord, ses DLLs ont la priorité.
try:
    import torch
    _TORCH_AVAILABLE = True
except (ImportError, OSError):
    torch = None
    _TORCH_AVAILABLE = False

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use("Agg")  # backend non-interactif (pas d'écran requis, compatible CI/CD)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# Ajoute la racine au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    TRAIN_CSV, TEST_CSV, MODEL_PATH, MODELS_DIR,
    MLFLOW_EXPERIMENT_NAME, RANDOM_STATE,
    TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_MAX_DF, TFIDF_MIN_DF,
    LR_C, LR_MAX_ITER, SVM_C, XGB_N_ESTIMATORS, XGB_MAX_DEPTH, XGB_LEARNING_RATE,
    W2V_VECTOR_SIZE, W2V_WINDOW, W2V_MIN_COUNT,
    SBERT_MODEL, DISTILBERT_MODEL, DISTILBERT_MAX_LENGTH,
    DISTILBERT_BATCH_SIZE, DISTILBERT_EPOCHS, DISTILBERT_LEARNING_RATE,
    CLASSES, COL_TEXT, COL_LABEL
)


# =============================================================================
# VECTORISEURS PERSONNALISÉS
# Ces classes enveloppent Word2Vec et Sentence-BERT dans l'API sklearn
# pour pouvoir les utiliser dans un Pipeline sklearn standard.
# =============================================================================

class Word2VecVectorizer:
    """
    Transforme des textes en vecteurs numériques via Word2Vec.

    Word2Vec (Google, 2013) apprend des représentations vectorielles des mots
    en regardant leur contexte. Des mots similaires ont des vecteurs proches.
    Exemple : vecteur("roi") - vecteur("homme") + vecteur("femme") ≈ vecteur("reine")

    Pour représenter une phrase entière, on fait la MOYENNE des vecteurs de tous
    ses mots. C'est une simplification mais ça fonctionne bien en pratique.

    Différence avec TF-IDF :
    → TF-IDF : représentation creuse (sparse) basée sur la fréquence des mots
    → Word2Vec : représentation dense (dense) basée sur le sens des mots
    """

    def __init__(self, vector_size=100, window=5, min_count=2, workers=4):
        """
        Args:
            vector_size : dimension des vecteurs de mots (100 = 100 nombres par mot)
            window      : taille de la fenêtre contextuelle (5 mots autour)
            min_count   : ignore les mots apparaissant moins de 2 fois
            workers     : nombre de threads parallèles pour l'entraînement
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None

    def fit(self, texts, y=None):
        """
        Entraîne le modèle Word2Vec sur le corpus de textes.
        Appelé automatiquement par Pipeline.fit()
        """
        from gensim.models import Word2Vec

        # Tokenise les textes (split sur les espaces)
        # Word2Vec a besoin d'une liste de listes de mots
        sentences = [text.split() for text in texts]

        # Entraîne Word2Vec sur le corpus
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            seed=RANDOM_STATE
        )
        return self

    def transform(self, texts):
        """
        Transforme une liste de textes en matrice de vecteurs.
        Chaque texte → moyenne des vecteurs de ses mots.
        Appelé automatiquement par Pipeline.transform()
        """
        vectors = []
        for text in texts:
            words = text.split()
            # Garde seulement les mots qui existent dans le vocabulaire W2V
            word_vecs = [
                self.model.wv[word]
                for word in words
                if word in self.model.wv
            ]
            if word_vecs:
                # Moyenne des vecteurs des mots = vecteur de la phrase
                vectors.append(np.mean(word_vecs, axis=0))
            else:
                # Texte sans mots connus → vecteur zéro
                vectors.append(np.zeros(self.vector_size))

        return np.array(vectors)

    def fit_transform(self, texts, y=None):
        """Entraîne et transforme en une seule étape."""
        return self.fit(texts, y).transform(texts)


class SBERTVectorizer:
    """
    Transforme des textes en vecteurs via Sentence-BERT.

    Sentence-BERT (2019) est un modèle Transformer optimisé pour encoder
    des phrases entières en vecteurs de haute qualité (384 ou 768 dimensions).

    Avantage sur Word2Vec :
    → SBERT comprend le contexte : "le film est bon" ≠ "le film n'est pas bon"
    → SBERT prend en compte toute la phrase, pas juste les mots séparément

    Modèle utilisé : all-MiniLM-L6-v2
    → Léger (90MB), rapide, et très performant sur les tâches de classification
    → Produit des vecteurs de 384 dimensions
    """

    def __init__(self, model_name: str = SBERT_MODEL):
        self.model_name = model_name
        self.model = None

    def fit(self, texts, y=None):
        """Charge le modèle SBERT (le télécharge si nécessaire la première fois)."""
        from sentence_transformers import SentenceTransformer
        print(f"  Chargement du modèle SBERT : {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            print(f"  GPU détecté : {torch.cuda.get_device_name(0)}")
        return self

    def transform(self, texts):
        """
        Encode les textes en vecteurs denses.
        show_progress_bar=True affiche une barre de progression.
        """
        print(f"  Encodage de {len(texts):,} textes avec SBERT...")
        return self.model.encode(
            list(texts),
            show_progress_bar=True,
            batch_size=64  # traite 64 textes à la fois pour économiser la RAM
        )

    def fit_transform(self, texts, y=None):
        return self.fit(texts, y).transform(texts)


# =============================================================================
# CONSTRUCTEURS DE MODÈLES
# Chaque fonction retourne un Pipeline sklearn complet :
# [vectoriseur] → [classificateur]
# =============================================================================

def build_tfidf_lr() -> Pipeline:
    """
    Modèle 1 : TF-IDF + Logistic Regression avec class_weight="balanced"

    TF-IDF (Term Frequency - Inverse Document Frequency) :
    → Chaque mot reçoit un score = fréquence dans le document × rareté dans le corpus
    → Les mots communs ("le", "de", "et") ont un score faible (IDF bas)
    → Les mots spécifiques ("étanche", "trail", "raquette") ont un score élevé
    → Résultat : vecteur creux de 10 000 dimensions (une par mot du vocabulaire)

    Logistic Regression avec class_weight="balanced" :
    → class_weight="balanced" calcule automatiquement les poids de chaque classe
    → Formule : poids(c) = n_total / (n_classes × n_exemples(c))
    → Si POSITIF a 10x plus d'exemples que SPAM, SPAM aura un poids 10x plus élevé
    → Effet : les erreurs sur SPAM sont pénalisées 10x plus fortement

    Paramètres clés :
    → max_features=10000 : vocabulaire limité aux 10 000 mots les plus importants
    → ngram_range=(1,2) : unigrammes ("bon") ET bigrammes ("très bon", "pas bien")
    → C=1.0 : régularisation (plus petit = plus de régularisation, moins d'overfitting)
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,    # 10 000 features max
            ngram_range=TFIDF_NGRAM_RANGE,      # unigrammes + bigrammes
            max_df=TFIDF_MAX_DF,                # ignore mots dans > 95% des docs
            min_df=TFIDF_MIN_DF,                # ignore mots dans < 2 docs
            sublinear_tf=True,                  # log(TF) au lieu de TF brut (meilleur)
            strip_accents="unicode"             # supprime les accents pour uniformiser
        )),
        ("clf", LogisticRegression(
            C=LR_C,
            max_iter=LR_MAX_ITER,
            class_weight="balanced",            # compense le déséquilibre automatiquement
            random_state=RANDOM_STATE,
            n_jobs=-1                           # utilise tous les cœurs CPU
        ))
    ])


def build_tfidf_svm() -> Pipeline:
    """
    Modèle 2 : TF-IDF + SVM (Support Vector Machine) avec class_weight="balanced"

    LinearSVC (Linear Support Vector Classifier) :
    → Trouve l'hyperplan qui maximise la marge entre les classes
    → Très efficace sur des données textuelles haute dimension (sparse)
    → Plus rapide que SVM avec noyau RBF sur grandes dimensions

    Problème : LinearSVC ne retourne pas de probabilités (seulement classe prédite)
    Solution : CalibratedClassifierCV enveloppe LinearSVC et ajoute les probabilités
    → Utilise la calibration de Platt (méthode statistique) pour estimer les probas
    → Nécessaire pour notre endpoint /predict qui retourne un score de confiance
    """
    base_svm = LinearSVC(
        C=SVM_C,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=2000
    )
    # CalibratedClassifierCV ajoute predict_proba() au SVM via validation croisée interne
    calibrated_svm = CalibratedClassifierCV(base_svm, cv=3)

    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            max_df=TFIDF_MAX_DF,
            min_df=TFIDF_MIN_DF,
            sublinear_tf=True,
            strip_accents="unicode"
        )),
        ("clf", calibrated_svm)
    ])


def build_tfidf_xgb() -> Pipeline:
    """
    Modèle 3 : TF-IDF + XGBoost (Gradient Boosting)

    XGBoost (eXtreme Gradient Boosting) :
    → Ensemble de centaines d'arbres de décision entraînés successivement
    → Chaque arbre corrige les erreurs du précédent (boosting)
    → Très puissant sur des données tabulaires et des features numériques mixtes

    Gestion du déséquilibre avec scale_pos_weight :
    → Paramètre calculé automatiquement : n_négatifs / n_positifs par classe
    → XGBoost traite le déséquilibre au niveau de la fonction de perte (différent
      de class_weight qui modifie les poids des exemples dans sklearn)
    → Pour classification multiclasse, on utilise sample_weight lors du fit
    """
    from xgboost import XGBClassifier

    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            max_df=TFIDF_MAX_DF,
            min_df=TFIDF_MIN_DF,
            sublinear_tf=True,
            strip_accents="unicode"
        )),
        ("clf", XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS,     # 200 arbres
            max_depth=XGB_MAX_DEPTH,           # profondeur max 6
            learning_rate=XGB_LEARNING_RATE,   # 0.1 = pas d'apprentissage
            use_label_encoder=False,
            eval_metric="mlogloss",            # log-loss multiclasse
            random_state=RANDOM_STATE,
            n_jobs=-1                          # tous les cœurs
        ))
    ])


def build_w2v_lr() -> Pipeline:
    """
    Modèle 4 : Word2Vec + Logistic Regression avec SMOTE

    SMOTE est appliqué AVANT l'entraînement (hors du pipeline sklearn).
    Voir la fonction train_model() pour les détails.

    Word2Vec apprend des embeddings de mots à partir du corpus d'entraînement.
    C'est une des premières techniques à capturer la sémantique des mots.
    """
    return Pipeline([
        ("w2v", Word2VecVectorizer(
            vector_size=W2V_VECTOR_SIZE,   # vecteurs de 100 dimensions
            window=W2V_WINDOW,             # fenêtre de 5 mots
            min_count=W2V_MIN_COUNT        # ignore les mots rares
        )),
        ("clf", LogisticRegression(
            C=LR_C,
            max_iter=LR_MAX_ITER,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])


def build_sbert_lr() -> Pipeline:
    """
    Modèle 5 : Sentence-BERT + Logistic Regression avec ADASYN

    ADASYN (Adaptive Synthetic Sampling) est appliqué AVANT l'entraînement.
    Voir la fonction train_model() pour les détails d'application.

    SBERT all-MiniLM-L6-v2 produit des vecteurs de 384 dimensions.
    Ces vecteurs capturent le sens de phrases entières (pas juste des mots).
    """
    return Pipeline([
        ("sbert", SBERTVectorizer(model_name=SBERT_MODEL)),
        ("clf", LogisticRegression(
            C=LR_C,
            max_iter=LR_MAX_ITER,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])


def build_distilbert(n_labels: int = 4) -> Any:
    """
    Modèle 6 : DistilBERT fine-tuné pour la classification de sentiment

    DistilBERT est une version allégée de BERT :
    → 60% plus rapide que BERT original
    → 40% moins de paramètres
    → Conserve 97% des performances de BERT

    Fine-tuning :
    → On part d'un modèle pré-entraîné sur Wikipedia + BookCorpus (milliards de textes)
    → On ajoute une couche de classification (4 classes)
    → On réentraîne le tout sur nos avis Amazon (transfert d'apprentissage)
    → Résultat : le modèle comprend déjà le langage, il apprend juste à classer

    Gestion déséquilibre : weighted cross-entropy loss
    → La fonction de perte (cross-entropy) est pondérée par l'inverse des fréquences
    → Les erreurs sur SPAM (rare) sont autant pénalisées que les erreurs sur POSITIF (fréquent)

    Prérequis : pip install transformers torch
    """
    try:
        import torch
        from transformers import (
            DistilBertForSequenceClassification,
            DistilBertTokenizer
        )
    except ImportError:
        raise ImportError(
            "Installe les dépendances : pip install transformers torch"
        )

    print(f"  Chargement du modèle DistilBERT : {DISTILBERT_MODEL}...")
    tokenizer = DistilBertTokenizer.from_pretrained(DISTILBERT_MODEL)
    model = DistilBertForSequenceClassification.from_pretrained(
        DISTILBERT_MODEL,
        num_labels=n_labels
    )

    return model, tokenizer


def apply_imbalance_strategy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    strategy: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applique une stratégie de rééchantillonnage pour corriger le déséquilibre.

    Cette fonction est appelée APRÈS la vectorisation TF-IDF ou W2V/SBERT,
    directement sur les vecteurs numériques.

    Args:
        X_train  : matrice de features (numpy array ou sparse matrix)
        y_train  : labels d'entraînement
        strategy : "smote", "adasyn", ou "none"

    Returns:
        X_resampled, y_resampled : données rééchantillonnées
    """
    if strategy == "none":
        print("  Aucun rééchantillonnage appliqué (class_weight utilisé à la place)")
        return X_train, y_train

    try:
        from imblearn.over_sampling import SMOTE, ADASYN
    except ImportError:
        print("  AVERTISSEMENT : imbalanced-learn non installé. Ignoré.")
        print("  Lance : pip install imbalanced-learn")
        return X_train, y_train

    # Compte les exemples par classe avant rééchantillonnage
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"  Distribution AVANT rééchantillonnage :")
    for cls, cnt in zip(unique, counts):
        print(f"    {cls}: {cnt:,} exemples")

    if strategy == "smote":
        # SMOTE : génère des exemples synthétiques pour les classes minoritaires
        # en interpolant entre des exemples proches dans l'espace des features
        # k_neighbors=5 = utilise les 5 voisins les plus proches pour interpoler
        print("\n  Application de SMOTE...")
        sampler = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)

    elif strategy == "adasyn":
        # ADASYN : comme SMOTE mais génère plus d'exemples dans les zones
        # où le modèle commet le plus d'erreurs (frontières de décision)
        # sampling_strategy='not majority' : rééchantillonne toutes les classes
        # minoritaires jusqu'au niveau de la classe majoritaire (NEGATIF ~90k)
        print("\n  Application de ADASYN...")
        sampler = ADASYN(random_state=RANDOM_STATE, sampling_strategy='not majority')

    try:
        # Convertit la matrice sparse en dense si nécessaire (requis par SMOTE)
        if hasattr(X_train, "toarray"):
            print("  Conversion sparse → dense pour le rééchantillonnage...")
            X_dense = X_train.toarray()
        else:
            X_dense = X_train

        X_resampled, y_resampled = sampler.fit_resample(X_dense, y_train)

        unique_new, counts_new = np.unique(y_resampled, return_counts=True)
        print(f"  Distribution APRÈS rééchantillonnage :")
        for cls, cnt in zip(unique_new, counts_new):
            print(f"    {cls}: {cnt:,} exemples")

        return X_resampled, y_resampled

    except Exception as e:
        print(f"  AVERTISSEMENT : {strategy.upper()} échoué ({e}).")
        if strategy == "adasyn":
            print("  Fallback sur SMOTE (plus stable en haute dimension)...")
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy='not majority', k_neighbors=5)
                X_resampled, y_resampled = smote.fit_resample(X_dense, y_train)
                unique_new, counts_new = np.unique(y_resampled, return_counts=True)
                print(f"  Distribution APRÈS SMOTE (fallback) :")
                for cls, cnt in zip(unique_new, counts_new):
                    print(f"    {cls}: {cnt:,} exemples")
                return X_resampled, y_resampled
            except Exception as e2:
                print(f"  SMOTE aussi échoué ({e2}). Données originales utilisées.")
        else:
            print(f"  Données originales utilisées.")
        return X_train, y_train


# =============================================================================
# ENTRAÎNEMENT D'UN MODÈLE + LOGGING MLFLOW
# =============================================================================

def train_model(
    model_name: str,
    pipeline: Pipeline,
    X_train: pd.Series,
    y_train: pd.Series,
    X_test: pd.Series,
    y_test: pd.Series,
    imbalance_strategy: str = "none",
    is_distilbert: bool = False
) -> Dict:
    """
    Entraîne un modèle, évalue ses performances, et logge tout dans MLflow.

    MLflow Tracking enregistre pour chaque run :
    - Les PARAMÈTRES : hyperparamètres du modèle (C, max_features, etc.)
    - Les MÉTRIQUES : accuracy, F1 weighted, F1 par classe, CV score
    - Le MODÈLE : le pipeline serialisé, récupérable via le Model Registry
    - Les ARTEFACTS : rapport de classification, matrice de confusion

    Args:
        model_name          : identifiant du modèle ("tfidf_lr", "tfidf_svm", etc.)
        pipeline            : Pipeline sklearn complet
        X_train, y_train    : données d'entraînement
        X_test, y_test      : données de test
        imbalance_strategy  : "none", "smote", ou "adasyn"
        is_distilbert       : True pour le fine-tuning DistilBERT

    Returns:
        dict avec les métriques principales
    """
    print(f"\n{'=' * 60}")
    print(f"ENTRAÎNEMENT : {model_name.upper()}")
    print(f"{'=' * 60}")
    print(f"Stratégie déséquilibre : {imbalance_strategy}")
    print(f"Taille train : {len(X_train):,} exemples")
    print(f"Taille test  : {len(X_test):,} exemples")

    # Lance un run MLflow pour tracker cet entraînement
    with mlflow.start_run(run_name=model_name):

        # ── Étape 1 : Log des paramètres dans MLflow ─────────────────────────
        # Ces paramètres seront visibles dans l'UI MLflow (http://localhost:5000)
        # et permettront de comparer les configurations entre runs
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("imbalance_strategy", imbalance_strategy)
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("n_classes", len(CLASSES))

        # Log des hyperparamètres spécifiques selon le modèle
        if "tfidf" in model_name:
            mlflow.log_param("tfidf_max_features", TFIDF_MAX_FEATURES)
            mlflow.log_param("tfidf_ngram_range", str(TFIDF_NGRAM_RANGE))
        if "lr" in model_name or model_name == "w2v_lr" or model_name == "sbert_lr":
            mlflow.log_param("lr_C", LR_C)
        if model_name == "tfidf_xgb":
            mlflow.log_param("xgb_n_estimators", XGB_N_ESTIMATORS)
            mlflow.log_param("xgb_max_depth", XGB_MAX_DEPTH)
            mlflow.log_param("xgb_learning_rate", XGB_LEARNING_RATE)
        if model_name == "w2v_lr":
            mlflow.log_param("w2v_vector_size", W2V_VECTOR_SIZE)
            mlflow.log_param("w2v_window", W2V_WINDOW)
        if model_name == "sbert_lr":
            mlflow.log_param("sbert_model", SBERT_MODEL)
        if model_name == "distilbert":
            mlflow.log_param("distilbert_model", DISTILBERT_MODEL)
            mlflow.log_param("distilbert_epochs", DISTILBERT_EPOCHS)
            mlflow.log_param("distilbert_learning_rate", DISTILBERT_LEARNING_RATE)

        # ── Étape 2 : Entraînement ────────────────────────────────────────────
        print("\nEntraînement en cours...")

        if is_distilbert:
            # Cas spécial : fine-tuning DistilBERT avec PyTorch
            metrics = _train_distilbert(pipeline, X_train, y_train, X_test, y_test)
            trained_pipeline = pipeline  # le modèle DistilBERT est déjà entraîné
            y_pred = metrics.pop("y_pred")  # extrait y_pred pour MLflow (confusion matrix + report)

        else:
            # Cas standard : Pipeline sklearn (TF-IDF, W2V, SBERT)

            # Pour les modèles qui utilisent SMOTE/ADASYN, on vectorise d'abord,
            # puis on rééchantillonne, puis on entraîne le classificateur séparément
            if imbalance_strategy in ["smote", "adasyn"]:

                # Phase 1 : vectorisation (le vectoriseur apprend sur X_train)
                vectorizer_step = pipeline.steps[0]  # ex: ("tfidf", TfidfVectorizer(...))
                clf_step = pipeline.steps[1]          # ex: ("clf", LogisticRegression(...))

                print(f"  Vectorisation avec {vectorizer_step[0]}...")
                vectorizer_step[1].fit(X_train)
                X_train_vec = vectorizer_step[1].transform(X_train)
                X_test_vec = vectorizer_step[1].transform(X_test)

                # Phase 2 : rééchantillonnage sur les vecteurs
                X_train_res, y_train_res = apply_imbalance_strategy(
                    X_train_vec, y_train.values, imbalance_strategy
                )

                # Phase 3 : entraînement du classificateur sur données rééchantillonnées
                print(f"  Entraînement du classificateur {clf_step[0]}...")
                clf_step[1].fit(X_train_res, y_train_res)

                # Prédiction sur le test (avec le vectoriseur déjà ajusté)
                y_pred = clf_step[1].predict(X_test_vec)
                y_proba = (
                    clf_step[1].predict_proba(X_test_vec)
                    if hasattr(clf_step[1], "predict_proba") else None
                )
                trained_pipeline = pipeline

            else:
                # Sans rééchantillonnage : entraîne le pipeline complet directement
                # XGBoost requiert des labels entiers (pas des strings) :
                # on encode CLASSES → [0,1,2,3] pour le fit, puis on décode y_pred
                # le_xgb est aussi utilisé plus bas pour cross_val_score
                from sklearn.preprocessing import LabelEncoder
                le_xgb = LabelEncoder()
                le_xgb.fit(CLASSES)
                if model_name == "tfidf_xgb":
                    y_train_enc = le_xgb.transform(y_train)
                    # Calcul des poids par classe pour gérer le déséquilibre :
                    # SPAM (~5%) et NEUTRE (~19%) reçoivent un poids plus élevé
                    # que POSITIF (~38%) et NEGATIF (~38%)
                    from sklearn.utils.class_weight import compute_class_weight
                    class_weights = compute_class_weight(
                        "balanced",
                        classes=np.unique(y_train_enc),
                        y=y_train_enc
                    )
                    cw_dict = dict(zip(np.unique(y_train_enc), class_weights))
                    sample_weights = np.array([cw_dict[lbl] for lbl in y_train_enc])
                    # clf__sample_weight : syntaxe Pipeline sklearn pour passer
                    # sample_weight directement au XGBClassifier (étape "clf")
                    pipeline.fit(X_train, y_train_enc, clf__sample_weight=sample_weights)
                    y_pred = le_xgb.inverse_transform(pipeline.predict(X_test))
                else:
                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)
                y_proba = (
                    pipeline.predict_proba(X_test)
                    if hasattr(pipeline, "predict_proba") else None
                )
                trained_pipeline = pipeline

            # ── Étape 3 : Évaluation ─────────────────────────────────────────
            accuracy = accuracy_score(y_test, y_pred)
            f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

            # F1 par classe individuelle
            f1_per_class = f1_score(y_test, y_pred, average=None,
                                     labels=CLASSES, zero_division=0)

            # Validation croisée sur le train (évalue la stabilité du modèle)
            # cv=3 : 3 folds (compromis vitesse/fiabilité)
            # Seulement pour les modèles rapides (pas SBERT ni DistilBERT)
            cv_score = None
            if model_name in ["tfidf_lr", "tfidf_svm", "tfidf_xgb"]:
                print("  Validation croisée (3 folds)...")
                # XGBoost requiert des labels entiers → on passe les labels encodés
                y_train_cv = le_xgb.transform(y_train) if model_name == "tfidf_xgb" else y_train
                cv_scores = cross_val_score(
                    trained_pipeline, X_train, y_train_cv,
                    cv=3, scoring="f1_weighted", n_jobs=-1
                )
                cv_score = cv_scores.mean()
                cv_std = cv_scores.std()
                mlflow.log_metric("cv_f1_mean", cv_score)
                mlflow.log_metric("cv_f1_std", cv_std)
                print(f"  CV F1 : {cv_score:.4f} ± {cv_std:.4f}")

            metrics = {
                "accuracy": accuracy,
                "f1_weighted": f1_weighted,
                "f1_macro": f1_macro,
                "f1_per_class": dict(zip(CLASSES, f1_per_class)),
                "cv_f1": cv_score
            }

        # ── Étape 4 : Affichage des résultats ─────────────────────────────────
        print(f"\nRÉSULTATS {model_name.upper()} :")
        print(f"  Accuracy   : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  F1 weighted: {metrics['f1_weighted']:.4f}")
        print(f"  F1 macro   : {metrics['f1_macro']:.4f}")
        print(f"\n  F1 par classe :")
        for cls, f1 in metrics["f1_per_class"].items():
            bar = "█" * int(f1 * 20)
            print(f"    {cls:<10} : {f1:.4f}  {bar}")

        # Rapport complet de classification
        if not is_distilbert:
            print(f"\n  Rapport de classification détaillé :")
            print(classification_report(y_test, y_pred, labels=CLASSES,
                                        target_names=CLASSES, zero_division=0))

        # ── Étape 5 : Log des métriques dans MLflow ───────────────────────────
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("f1_weighted", metrics["f1_weighted"])
        mlflow.log_metric("f1_macro", metrics["f1_macro"])
        for cls, f1 in metrics["f1_per_class"].items():
            mlflow.log_metric(f"f1_{cls.lower()}", f1)

        # ── Étape 6 : Sauvegarde du modèle dans MLflow ───────────────────────
        # mlflow.sklearn.log_model échoue silencieusement dans MLflow 3.x.
        # On utilise pickle + mlflow.log_artifact directement, qui est fiable
        # dans toutes les versions de MLflow.
        if not is_distilbert:
            import pickle
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                model_path = os.path.join(tmpdir, "model.pkl")
                with open(model_path, "wb") as f:
                    pickle.dump(trained_pipeline, f)
                mlflow.log_artifact(model_path, artifact_path="model")
        else:
            # Pour DistilBERT, on sauvegarde avec mlflow.transformers (ou custom)
            pass

        # ── Étape 5b : Log du classification report (texte) ──────────────────
        # Le classification report montre precision, recall, F1 pour chaque classe.
        # Dans l'interface MLflow → onglet "Artifacts" → classification_report.txt
        report_str = classification_report(
            y_test, y_pred,
            labels=CLASSES,
            target_names=CLASSES,
            zero_division=0
        )
        mlflow.log_text(report_str, "classification_report.txt")

        # ── Étape 5c : Log de la matrice de confusion (image PNG) ─────────────
        # La matrice de confusion montre les erreurs de classification :
        # - Ligne = label réel, Colonne = label prédit
        # - La diagonale = prédictions correctes
        # - Hors diagonale = erreurs (ex: NEUTRE prédit comme POSITIF)
        # Dans l'interface MLflow → onglet "Artifacts" → confusion_matrix.png
        cm = confusion_matrix(y_test, y_pred, labels=CLASSES)
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(
            cm,
            annot=True,           # affiche les chiffres dans chaque case
            fmt="d",              # format entier (pas de décimales)
            cmap="Blues",         # palette de couleurs bleues
            xticklabels=CLASSES,  # labels colonnes (prédictions)
            yticklabels=CLASSES,  # labels lignes (réels)
            ax=ax
        )
        ax.set_title(f"Matrice de confusion — {model_name}", fontsize=13, pad=12)
        ax.set_xlabel("Label prédit", fontsize=11)
        ax.set_ylabel("Label réel", fontsize=11)
        plt.tight_layout()
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)  # libère la mémoire matplotlib

        # Log du tag pour identifier le type de modèle
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("imbalance_strategy", imbalance_strategy)

    return {**metrics, "model_name": model_name, "pipeline": trained_pipeline}


def _train_distilbert(
    model_and_tokenizer,
    X_train: pd.Series,
    y_train: pd.Series,
    X_test: pd.Series,
    y_test: pd.Series
) -> Dict:
    """
    Fine-tuning de DistilBERT avec PyTorch.

    Le fine-tuning consiste à :
    1. Partir du modèle pré-entraîné (déjà entraîné sur des milliards de textes)
    2. Ajouter une couche de classification (4 classes)
    3. Réentraîner tout le modèle sur nos avis Amazon pendant 3 epochs
    4. Utiliser une weighted cross-entropy loss pour gérer le déséquilibre

    Weighted cross-entropy loss :
    → La perte standard cross-entropy pénalise toutes les erreurs pareil
    → La version pondérée multiplie la perte par un poids inversement proportionnel
      à la fréquence de la classe → les erreurs sur SPAM (rare) coûtent plus cher

    Returns:
        dict avec accuracy, f1_weighted, f1_macro, f1_per_class
    """
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
        from torch.optim import AdamW
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
        from transformers import get_linear_schedule_with_warmup
    except ImportError:
        print("  ERREUR : torch et transformers requis pour DistilBERT")
        print("  Lance : pip install transformers torch")
        raise

    model, tokenizer = model_and_tokenizer

    # Encode les labels en entiers (NEGATIF→0, NEUTRE→1, POSITIF→2, SPAM→3)
    le = LabelEncoder()
    le.fit(CLASSES)
    y_train_enc = le.transform(y_train.values)
    y_test_enc = le.transform(y_test.values)

    # ── Dataset PyTorch ──────────────────────────────────────────────────────
    class ReviewDataset(Dataset):
        """Dataset PyTorch pour les avis Amazon."""
        def __init__(self, texts, labels, tokenizer, max_length):
            self.encodings = tokenizer(
                list(texts),
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt"
            )
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                "input_ids": self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx],
                "labels": self.labels[idx]
            }

    # Crée les datasets
    print("  Tokenisation des textes pour DistilBERT...")
    train_dataset = ReviewDataset(X_train.values, y_train_enc, tokenizer, DISTILBERT_MAX_LENGTH)
    test_dataset = ReviewDataset(X_test.values, y_test_enc, tokenizer, DISTILBERT_MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=DISTILBERT_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=DISTILBERT_BATCH_SIZE)

    # ── Device (GPU si disponible, sinon CPU) ────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device} {'(GPU détecté !)' if device.type == 'cuda' else '(CPU)'}")
    model = model.to(device)

    # ── Calcul des poids pour la weighted loss ───────────────────────────────
    # Inversement proportionnel à la fréquence de chaque classe
    class_counts = np.bincount(y_train_enc)
    class_weights = len(y_train_enc) / (len(CLASSES) * class_counts)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    # ── Optimiseur et scheduler ──────────────────────────────────────────────
    # AdamW = Adam avec weight decay (régularisation L2) — recommandé pour BERT
    optimizer = AdamW(model.parameters(), lr=DISTILBERT_LEARNING_RATE)

    # Scheduler linéaire : réduit progressivement le lr après un warmup
    total_steps = len(train_loader) * DISTILBERT_EPOCHS
    warmup_steps = total_steps // 10  # 10% de warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # ── Boucle d'entraînement ─────────────────────────────────────────────────
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights_tensor)  # weighted loss

    for epoch in range(DISTILBERT_EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        n_total = 0

        print(f"\n  Epoch {epoch + 1}/{DISTILBERT_EPOCHS}")
        for batch_idx, batch in enumerate(train_loader):
            # Transfert des données vers le device (GPU/CPU)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass : calcule les prédictions
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Calcul de la perte pondérée
            loss = loss_fn(logits, labels)

            # Backward pass : calcule les gradients
            loss.backward()

            # Clip les gradients pour éviter l'explosion (problème courant avec BERT)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Mise à jour des poids
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            n_total += len(labels)

            if (batch_idx + 1) % 50 == 0:
                print(f"    Batch {batch_idx + 1}/{len(train_loader)} "
                      f"| Loss: {total_loss / (batch_idx + 1):.4f} "
                      f"| Acc: {correct / n_total:.4f}")

        epoch_loss = total_loss / len(train_loader)
        epoch_acc = correct / n_total
        print(f"  → Epoch {epoch + 1} terminée | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")
        mlflow.log_metric(f"train_loss_epoch_{epoch+1}", epoch_loss)
        mlflow.log_metric(f"train_acc_epoch_{epoch+1}", epoch_acc)

    # ── Évaluation sur le test ────────────────────────────────────────────────
    print("\n  Évaluation sur le test set...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Convertit les encodages back en labels string
    y_pred_str = le.inverse_transform(all_preds)
    y_test_str = le.inverse_transform(all_labels)

    accuracy = accuracy_score(y_test_str, y_pred_str)
    f1_weighted = f1_score(y_test_str, y_pred_str, average="weighted", zero_division=0)
    f1_macro = f1_score(y_test_str, y_pred_str, average="macro", zero_division=0)
    f1_per_class = f1_score(y_test_str, y_pred_str, average=None,
                             labels=CLASSES, zero_division=0)

    # Sauvegarde le modèle DistilBERT fine-tuné
    distilbert_dir = os.path.join(MODELS_DIR, "distilbert_finetuned")
    os.makedirs(distilbert_dir, exist_ok=True)
    model.save_pretrained(distilbert_dir)
    model_and_tokenizer[1].save_pretrained(distilbert_dir)
    print(f"  Modèle DistilBERT sauvegardé dans {distilbert_dir}")

    return {
        "accuracy": accuracy,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "f1_per_class": dict(zip(CLASSES, f1_per_class)),
        "y_pred": y_pred_str   # nécessaire pour confusion matrix + classification report
    }


# =============================================================================
# SAUVEGARDE DU MEILLEUR MODÈLE
# =============================================================================

def save_best_model(results: List[Dict]) -> Dict:
    """
    Identifie le meilleur modèle et sauvegarde le pipeline pour FastAPI.

    Critère de sélection : F1 score weighted (le plus équitable sur toutes les classes)

    FastAPI chargera ce fichier au démarrage via joblib.load(MODEL_PATH).
    Le format .pkl (pickle) est le format standard de sérialisation Python.

    Args:
        results : liste des résultats de tous les entraînements

    Returns:
        Le meilleur résultat (dict avec model_name, f1_weighted, pipeline)
    """
    # Trie les modèles par F1 weighted décroissant
    results_sorted = sorted(results, key=lambda x: x["f1_weighted"], reverse=True)
    best = results_sorted[0]

    print(f"\n{'=' * 60}")
    print(f"CLASSEMENT DES MODÈLES PAR F1 WEIGHTED")
    print(f"{'=' * 60}")
    for i, r in enumerate(results_sorted, 1):
        marker = "← MEILLEUR" if i == 1 else ""
        print(f"  {i}. {r['model_name']:<15} F1={r['f1_weighted']:.4f}  {marker}")

    # Sauvegarde le pipeline du meilleur modèle
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best["pipeline"], f)

    # Sauvegarde aussi les classes (nécessaire pour FastAPI)
    classes_path = os.path.join(MODELS_DIR, "classes.json")
    with open(classes_path, "w") as f:
        json.dump({"classes": CLASSES, "best_model": best["model_name"]}, f)

    print(f"\nMEILLEUR MODÈLE : {best['model_name']} (F1={best['f1_weighted']:.4f})")
    print(f"Pipeline sauvegardé : {MODEL_PATH}")
    print(f"\nÉtape suivante : python src/promote_model.py")

    return best


# =============================================================================
# PROGRAMME PRINCIPAL
# =============================================================================

def main():
    """
    Point d'entrée principal :
    1. Charge les données d'entraînement et de test
    2. Configure MLflow
    3. Entraîne les modèles demandés
    4. Sauvegarde le meilleur modèle
    """
    # ── Arguments en ligne de commande ───────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Entraîne et compare les modèles de classification d'avis"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["tfidf_lr", "tfidf_svm", "tfidf_xgb", "w2v_lr", "sbert_lr", "distilbert"],
        default=["tfidf_lr", "tfidf_svm", "tfidf_xgb", "w2v_lr", "sbert_lr", "distilbert"],
        help="Modèles à entraîner (défaut : tous)"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Mode rapide : dataset réduit à 5000 exemples, 1 epoch pour DistilBERT"
    )
    args = parser.parse_args()

    # ── Chargement des données ───────────────────────────────────────────────
    print("=" * 60)
    print("CHARGEMENT DES DONNÉES")
    print("=" * 60)

    if not os.path.exists(TRAIN_CSV) or not os.path.exists(TEST_CSV):
        print(f"ERREUR : Données introuvables.")
        print("Lance d'abord : python spark/preprocess.py --no-spark")
        sys.exit(1)

    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    # Mode rapide : réduit le dataset pour tester rapidement
    if args.fast:
        print("MODE RAPIDE : dataset réduit à 5000 exemples")
        # Boucle explicite pour éviter le bug pandas 2.2+ où groupby().apply()
        # + reset_index(drop=True) supprime la colonne de groupement (COL_LABEL)
        samples = []
        for _, group in train_df.groupby(COL_LABEL):
            n = min(len(group), 1000)
            samples.append(group.sample(n, random_state=RANDOM_STATE))
        train_df = pd.concat(samples).reset_index(drop=True)
        test_df = test_df.sample(min(len(test_df), 1000), random_state=RANDOM_STATE)

    X_train = train_df[COL_TEXT]
    y_train = train_df[COL_LABEL]
    X_test = test_df[COL_TEXT]
    y_test = test_df[COL_LABEL]

    print(f"Train : {len(X_train):,} exemples")
    print(f"Test  : {len(X_test):,} exemples")
    print(f"\nDistribution des classes dans le train :")
    for cls, cnt in y_train.value_counts().items():
        pct = cnt / len(y_train) * 100
        print(f"  {cls:<10} : {cnt:>6,} ({pct:.1f}%)")

    # ── Configuration MLflow ──────────────────────────────────────────────────
    # L'expérience MLflow regroupe tous les runs sous le même nom
    # Tous les modèles entraînés ici seront visibles dans la même expérience
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    print(f"\nExpérience MLflow : '{MLFLOW_EXPERIMENT_NAME}'")
    print("Visualisation : mlflow ui → http://localhost:5000")

    # ── Registre des modèles ─────────────────────────────────────────────────
    # Associe chaque nom de modèle à sa fonction de construction et sa stratégie
    # d'équilibrage des classes
    MODELS_REGISTRY = {
        "tfidf_lr": {
            "builder": build_tfidf_lr,
            "imbalance": "none",        # utilise class_weight="balanced" dans LR
            "distilbert": False
        },
        "tfidf_svm": {
            "builder": build_tfidf_svm,
            "imbalance": "none",        # utilise class_weight="balanced" dans SVM
            "distilbert": False
        },
        "tfidf_xgb": {
            "builder": build_tfidf_xgb,
            "imbalance": "none",        # utilise sample_weight XGBoost
            "distilbert": False
        },
        "w2v_lr": {
            "builder": build_w2v_lr,
            "imbalance": "smote",       # génère des exemples synthétiques
            "distilbert": False
        },
        "sbert_lr": {
            "builder": build_sbert_lr,
            "imbalance": "adasyn",      # rééchantillonnage adaptatif
            "distilbert": False
        },
        "distilbert": {
            "builder": lambda: build_distilbert(n_labels=len(CLASSES)),
            "imbalance": "weighted_loss",  # gestion dans la loss PyTorch
            "distilbert": True
        },
    }

    # ── Entraînement des modèles ──────────────────────────────────────────────
    all_results = []
    models_to_train = args.models

    print(f"\nModèles à entraîner : {', '.join(models_to_train)}")
    print("Chaque modèle sera enregistré dans MLflow automatiquement.\n")

    for model_name in models_to_train:
        config = MODELS_REGISTRY[model_name]

        try:
            # Construction du pipeline ou du modèle DistilBERT
            pipeline = config["builder"]()

            # Entraînement + évaluation + logging MLflow
            result = train_model(
                model_name=model_name,
                pipeline=pipeline,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                imbalance_strategy=config["imbalance"],
                is_distilbert=config["distilbert"]
            )
            all_results.append(result)

        except Exception as e:
            print(f"\nERREUR lors de l'entraînement de {model_name} : {e}")
            print("Passage au modèle suivant...")
            import traceback
            traceback.print_exc()
            continue

    # ── Sauvegarde du meilleur modèle ─────────────────────────────────────────
    if all_results:
        best = save_best_model(all_results)
        return best
    else:
        print("\nAUCUN MODÈLE N'A ÉTÉ ENTRAÎNÉ AVEC SUCCÈS.")
        sys.exit(1)


if __name__ == "__main__":
    main()
