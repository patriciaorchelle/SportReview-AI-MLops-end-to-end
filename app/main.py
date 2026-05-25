# -*- coding: utf-8 -*-
"""
main.py — API FastAPI pour SportReview AI

Cette API expose le modèle de classification d'avis produits via HTTP REST.
Elle est conçue pour être robuste, monitorée et prête pour la production.

Endpoints disponibles :
  GET  /health          → liveness probe Kubernetes
  GET  /model/info      → métadonnées du modèle en production
  POST /predict         → prédiction sur un seul avis (avec SHAP optionnel)
  POST /predict/batch   → prédiction sur jusqu'à 500 avis
  POST /feedback        → signaler une mauvaise prédiction
  GET  /drift           → dernier rapport de drift
  GET  /metrics         → métriques Prometheus (pour Grafana)

Démarrage :
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import json
import os
import pickle
import sys
import time
import uuid
from datetime import datetime
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# Ajoute le dossier racine au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.schemas import (
    ReviewInput, BatchInput, FeedbackInput,
    PredictionOutput, BatchOutput, ModelInfoOutput,
    HealthOutput, FeedbackOutput, DriftOutput, SHAPExplanation
)
from app.monitoring import (
    record_prediction, record_request, RequestTimer,
    FEEDBACK_COUNT, ERROR_COUNT, BATCH_SIZE
)
from src.config import (
    MODEL_PATH, CLASSES_PATH, CONFIDENCE_THRESHOLD,
    API_MAX_BATCH_SIZE, DRIFT_REPORT_JSON, REPORTS_DIR
)


# ───────────────────────────────────────────────────────────────────────────
# CHARGEMENT DU MODÈLE AU DÉMARRAGE
# ───────────────────────────────────────────────────────────────────────────

def load_model():
    """
    Charge le modèle ML depuis le fichier pickle au démarrage de l'API.

    Le modèle est chargé UNE SEULE FOIS en mémoire au démarrage.
    Toutes les requêtes utilisent ensuite le même objet en mémoire
    (beaucoup plus rapide que de charger le modèle à chaque requête).

    Raises:
        FileNotFoundError : si le fichier pipeline.pkl n'existe pas
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Modele introuvable : {MODEL_PATH}\n"
            "Lance d'abord :\n"
            "  python data/load_data.py\n"
            "  python spark/preprocess.py --no-spark\n"
            "  python src/train.py"
        )

    # Charge le pipeline sklearn (TF-IDF + classifieur)
    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)

    # Charge la liste des classes
    # classes.json contient {"classes": [...], "best_model": "..."} (format de train.py)
    with open(CLASSES_PATH, "r") as f:
        classes_data = json.load(f)
    if isinstance(classes_data, dict):
        classes = classes_data["classes"]
    else:
        classes = classes_data  # compatibilité si c'est déjà une liste

    # Charge les métadonnées du modèle si disponibles
    metadata_path = os.path.join(os.path.dirname(MODEL_PATH), "metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

    print(f"Modele charge depuis {MODEL_PATH}")
    print(f"Classes : {classes}")
    print(f"Type : {metadata.get('model_name', 'Inconnu')}")

    return pipeline, classes, metadata


# Charge le modèle une fois au démarrage
# Ces variables sont partagées entre toutes les requêtes (thread-safe pour la lecture)
pipeline, CLASSES, MODEL_METADATA = load_model()

# Pour DistilBERT : force le chargement immédiat du modèle HuggingFace
# (évite le timeout au premier appel dû au lazy loading)
if hasattr(pipeline, '_load_model'):
    print("Pré-chargement de DistilBERT en mémoire...")
    pipeline._load_model()
    print("DistilBERT prêt.")

SERVER_START_TIME = time.time()  # pour calculer l'uptime
LOADED_AT = datetime.utcnow().isoformat()


# ───────────────────────────────────────────────────────────────────────────
# INITIALISATION DE L'APPLICATION FASTAPI
# ───────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SportReview AI — API de Classification d'Avis",
    description="""
    API REST pour classifier des avis produits sportifs en :
    **POSITIF**, **NEGATIF**, **NEUTRE** ou **SPAM**

    Fonctionnalités :
    - Prédiction unitaire avec score de confiance
    - Prédiction en batch (jusqu'à 500 avis)
    - Explication SHAP des prédictions
    - Monitoring via Prometheus/Grafana
    - Détection de drift
    """,
    version="1.0.0",
    docs_url="/docs",          # Swagger UI accessible sur /docs
    redoc_url="/redoc",        # ReDoc accessible sur /redoc
)

# Middleware CORS : permet les appels depuis n'importe quel domaine
# (nécessaire pour la démo Gradio et pour les tests depuis le navigateur)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # en production, remplacer par la liste des domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ────────────────────────────────────────────────────────────────────────────
# FONCTIONS UTILITAIRES
# ────────────────────────────────────────────────────────────────────────────

def compute_shap_explanation(text: str, label: str) -> SHAPExplanation:
    """
    Calcule les valeurs SHAP pour expliquer une prédiction.

    SHAP (SHapley Additive exPlanations) identifie les mots
    qui ont le plus influencé la prédiction du modèle.

    Pour TF-IDF + LR/SVM : utilise les coefficients du modèle directement
    (plus rapide que le SHAP standard).

    Args:
        text  : texte de l'avis
        label : label prédit

    Returns:
        SHAPExplanation avec les mots les plus importants
    """
    try:
        # DistilBERT (et autres non-sklearn) : pas d'explication SHAP disponible
        # SHAP classique nécessite un vectoriseur TF-IDF — pour les transformers
        # il faudrait BertViz ou Integrated Gradients (hors scope ici)
        if not hasattr(pipeline, "named_steps"):
            return SHAPExplanation(top_positive_words={}, top_negative_words={})

        # Récupère le vectoriseur TF-IDF du pipeline sklearn
        tfidf = pipeline.named_steps.get("tfidf")
        clf = pipeline.named_steps.get("clf") or pipeline.named_steps.get("sbert")

        if tfidf is None:
            # Pour Word2Vec ou SBERT, on ne peut pas facilement extraire les features
            return SHAPExplanation(
                top_positive_words={},
                top_negative_words={}
            )

        # Transforme le texte en vecteur TF-IDF
        tfidf_vector = tfidf.transform([text])

        # Récupère les coefficients du classifieur pour la classe prédite
        if hasattr(clf, "coef_"):
            # Logistic Regression ou SVM calibré
            label_idx = list(pipeline.classes_).index(label)

            if clf.coef_.ndim == 2:
                # Multi-classe : une ligne de coefficients par classe
                coefficients = clf.coef_[label_idx]
            else:
                coefficients = clf.coef_[0]

            # Multiplie les coefficients par les valeurs TF-IDF du texte
            # → les mots avec un produit élevé sont les plus importants
            feature_importance = np.array(tfidf_vector.todense()).flatten() * coefficients

            # Récupère le vocabulaire TF-IDF
            feature_names = tfidf.get_feature_names_out()

            # Garde seulement les features présentes dans le texte
            nonzero_indices = tfidf_vector.nonzero()[1]
            word_scores = {
                feature_names[i]: float(feature_importance[i])
                for i in nonzero_indices
            }

            # Trie par importance
            sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)

            # Top 5 mots positifs (contribuent VERS le label prédit)
            top_positive = {w: round(s, 4) for w, s in sorted_words[:5] if s > 0}
            # Top 5 mots négatifs (vont CONTRE le label prédit)
            top_negative = {w: round(s, 4) for w, s in sorted_words[-5:] if s < 0}

            return SHAPExplanation(
                top_positive_words=top_positive,
                top_negative_words=top_negative
            )
        else:
            # XGBoost ou autre modèle sans coef_
            return SHAPExplanation(
                top_positive_words={},
                top_negative_words={}
            )

    except Exception as e:
        # En cas d'erreur SHAP, on renvoie quand même la prédiction sans explication
        print(f"Erreur SHAP (non critique) : {e}")
        return SHAPExplanation(top_positive_words={}, top_negative_words={})


def make_prediction(text: str, include_explanation: bool = False) -> PredictionOutput:
    """
    Effectue une prédiction sur un texte et retourne le résultat structuré.

    Args:
        text               : texte de l'avis à classifier
        include_explanation: si True, calcule l'explication SHAP

    Returns:
        PredictionOutput avec label, confiance, probabilités
    """
    # Prédiction du label
    label = pipeline.predict([text])[0]

    # Probabilités pour toutes les classes
    probas = pipeline.predict_proba([text])[0]
    confidence = float(max(probas))

    # Dictionnaire label → probabilité
    proba_dict = {
        cls: round(float(p), 4)
        for cls, p in zip(CLASSES, probas)
    }

    # Explication SHAP (optionnelle car coûteuse en temps)
    explanation = None
    if include_explanation:
        explanation = compute_shap_explanation(text, label)

    # Enregistre les métriques Prometheus
    record_prediction(label, confidence, CONFIDENCE_THRESHOLD)

    return PredictionOutput(
        text=text,
        label=label,
        confidence=round(confidence, 4),
        probabilities=proba_dict,
        is_reliable=confidence >= CONFIDENCE_THRESHOLD,
        explanation=explanation,
        timestamp=datetime.utcnow().isoformat(),
    )


# ───────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthOutput, tags=["Monitoring"])
def health_check():
    """
    Liveness probe pour Kubernetes et Docker.

    Kubernetes appelle cet endpoint toutes les 30 secondes.
    Si la réponse n'est pas 200, Kubernetes redémarre le pod.

    Vérifie que :
    - Le serveur répond
    - Le modèle est chargé en mémoire
    """
    uptime = time.time() - SERVER_START_TIME

    return HealthOutput(
        status="healthy",
        model_loaded=pipeline is not None,
        classes=CLASSES,
        uptime_seconds=round(uptime, 2),
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/model/info", response_model=ModelInfoOutput, tags=["Monitoring"])
def model_info():
    """
    Retourne les métadonnées du modèle actuellement en production.

    Utile pour vérifier quelle version du modèle tourne
    sans avoir à aller dans MLflow.
    """
    # Récupère les infos du pipeline
    # DistilBertWrapper n'a pas named_steps (attribut sklearn uniquement)
    if hasattr(pipeline, "named_steps"):
        tfidf = pipeline.named_steps.get("tfidf")
    else:
        tfidf = None

    return ModelInfoOutput(
        model_type=MODEL_METADATA.get("model_name", "Inconnu"),
        classes=CLASSES,
        version=str(MODEL_METADATA.get("model_version", "1.0")),
        f1_score=MODEL_METADATA.get("f1_weighted"),
        run_id=MODEL_METADATA.get("run_id"),
        loaded_at=LOADED_AT,
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Prédiction"])
def predict(
    review: ReviewInput,
    explain: bool = Query(default=False, description="Inclure l'explication SHAP ?")
):
    """
    Classifie un seul avis produit.

    Retourne :
    - **label** : POSITIF, NEGATIF, NEUTRE ou SPAM
    - **confidence** : score de confiance entre 0 et 1
    - **probabilities** : probabilités pour chaque classe
    - **is_reliable** : True si la confiance dépasse 60%
    - **explanation** : mots les plus importants (si explain=True)
    """
    with RequestTimer() as timer:
        try:
            result = make_prediction(review.text, include_explanation=explain)
        except Exception as e:
            ERROR_COUNT.labels(error_type="model_error").inc()
            raise HTTPException(
                status_code=500,
                detail=f"Erreur lors de la prediction : {str(e)}"
            )

    # Enregistre les métriques de la requête
    record_request("/predict", "POST", 200, timer.duration)

    return result


@app.post("/predict/batch", response_model=BatchOutput, tags=["Prédiction"])
def predict_batch(batch: BatchInput):
    """
    Classifie plusieurs avis en une seule requête (jusqu'à 500).

    Plus efficace que de faire 500 appels individuels :
    - Réduit la latence réseau
    - Le modèle traite tous les textes en parallèle
    """
    if len(batch.texts) > API_MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {API_MAX_BATCH_SIZE} textes par requete (recu: {len(batch.texts)})"
        )

    with RequestTimer() as timer:
        try:
            predictions = []
            for text in batch.texts:
                pred = make_prediction(text, include_explanation=False)
                predictions.append(pred)

            # Enregistre la taille du batch dans les métriques
            BATCH_SIZE.observe(len(batch.texts))

        except Exception as e:
            ERROR_COUNT.labels(error_type="model_error").inc()
            raise HTTPException(
                status_code=500,
                detail=f"Erreur batch : {str(e)}"
            )

    record_request("/predict/batch", "POST", 200, timer.duration)

    return BatchOutput(
        predictions=predictions,
        count=len(predictions),
        processing_time_ms=round(timer.duration_ms, 2),
    )


@app.post("/feedback", response_model=FeedbackOutput, tags=["Amélioration"])
def submit_feedback(feedback: FeedbackInput):
    """
    Soumet une correction de prédiction.

    Permet aux utilisateurs de signaler quand le modèle se trompe.
    Ces corrections sont sauvegardées et peuvent être utilisées
    pour ré-entraîner le modèle avec des données corrigées.
    C'est le début d'un pipeline Human-in-the-Loop.
    """
    feedback_id = str(uuid.uuid4())[:8]   # ID court pour référence

    # Sauvegarde le feedback dans un fichier CSV
    feedback_dir = os.path.join(REPORTS_DIR, "feedback")
    os.makedirs(feedback_dir, exist_ok=True)

    feedback_file = os.path.join(feedback_dir, "feedback_log.csv")
    feedback_entry = {
        "id": feedback_id,
        "text": feedback.text,
        "predicted": feedback.predicted_label,
        "correct": feedback.correct_label,
        "comment": feedback.comment or "",
        "timestamp": datetime.utcnow().isoformat(),
    }

    import csv
    file_exists = os.path.exists(feedback_file)
    with open(feedback_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=feedback_entry.keys())
        if not file_exists:
            writer.writeheader()   # écrit les en-têtes si nouveau fichier
        writer.writerow(feedback_entry)

    # Incrémente le compteur Prometheus
    FEEDBACK_COUNT.inc()

    return FeedbackOutput(
        message="Feedback enregistre. Merci pour la correction !",
        feedback_id=feedback_id,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/drift", response_model=DriftOutput, tags=["Monitoring"])
def get_drift_report():
    """
    Retourne le dernier rapport de drift disponible.

    Le rapport est généré par le script src/drift_detection.py
    qui doit être lancé périodiquement (ex: chaque nuit via cron ou CI/CD).
    """
    if not os.path.exists(DRIFT_REPORT_JSON):
        return DriftOutput(
            drift_score=0.0,
            drift_detected=False,
            threshold=0.1,
            generated_at=None,
            report_available=False,
        )

    with open(DRIFT_REPORT_JSON, "r") as f:
        drift_data = json.load(f)

    return DriftOutput(
        drift_score=drift_data.get("drift_score", 0.0),
        drift_detected=drift_data.get("drift_detected", False),
        threshold=drift_data.get("threshold", 0.1),
        label_distribution=drift_data.get("label_drift", {}).get("label_distribution"),
        generated_at=drift_data.get("generated_at"),
        report_available=True,
    )


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    """
    Endpoint Prometheus — expose toutes les métriques au format texte.

    Prometheus scrape cet endpoint toutes les 15 secondes pour
    collecter les métriques, qui sont ensuite affichées dans Grafana.

    Format : texte brut (pas du JSON)
    """
    # generate_latest() retourne toutes les métriques au format Prometheus
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
