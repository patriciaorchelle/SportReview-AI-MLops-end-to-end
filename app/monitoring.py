# -*- coding: utf-8 -*-
"""
monitoring.py — Métriques Prometheus pour le monitoring de l'API

Prometheus est un système de monitoring qui collecte des métriques
en appelant régulièrement l'endpoint /metrics de l'API.
Grafana se connecte à Prometheus pour afficher les dashboards.

Métriques exposées :
- Nombre de requêtes par endpoint et status code
- Latence des requêtes (histogramme)
- Distribution des prédictions par label
- Score de confiance moyen
- Nombre de prédictions peu fiables (confiance < seuil)
"""

import time
from prometheus_client import (
    Counter,        # compteur qui ne peut qu'augmenter (ex: nombre de requêtes)
    Histogram,      # distribution de valeurs (ex: latence)
    Gauge,          # valeur instantanée (ex: mémoire utilisée)
    Summary,        # résumé statistique (moyenne, percentiles)
)


# ─────────────────────────────────────────────────────────────────────────────
# DÉFINITION DES MÉTRIQUES
# Ces objets sont créés une seule fois au démarrage de l'API
# ─────────────────────────────────────────────────────────────────────────────

# Compteur total de requêtes — segmenté par endpoint et status code
# Labels = dimensions de segmentation (comme GROUP BY en SQL)
REQUEST_COUNT = Counter(
    "sportreview_requests_total",           # nom de la métrique dans Prometheus
    "Nombre total de requetes API",          # description
    ["endpoint", "method", "status_code"]   # labels pour segmenter
)

# Histogramme de latence — mesure la distribution des temps de réponse
# Buckets = intervalles de temps en secondes (10ms, 25ms, 50ms, etc.)
REQUEST_LATENCY = Histogram(
    "sportreview_request_duration_seconds",
    "Duree des requetes en secondes",
    ["endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
    # Ex: si 95% des requêtes sont dans le bucket 0.1 → latence p95 = 100ms
)

# Compteur de prédictions — segmenté par label prédit
# Permet de détecter si la distribution des prédictions change (target drift)
PREDICTION_COUNT = Counter(
    "sportreview_predictions_total",
    "Nombre total de predictions par label",
    ["label"]   # POSITIF, NEGATIF, NEUTRE, SPAM
)

# Histogramme des scores de confiance
# Permet de voir si le modèle devient moins certain au fil du temps
CONFIDENCE_HISTOGRAM = Histogram(
    "sportreview_confidence_score",
    "Distribution des scores de confiance",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Compteur de prédictions peu fiables (confiance < 60%)
LOW_CONFIDENCE_COUNT = Counter(
    "sportreview_low_confidence_predictions_total",
    "Nombre de predictions avec confiance < seuil"
)

# Compteur d'erreurs API
ERROR_COUNT = Counter(
    "sportreview_errors_total",
    "Nombre d'erreurs API",
    ["error_type"]  # "validation_error", "model_error", "server_error"
)

# Jauge : nombre de requêtes en cours de traitement
REQUESTS_IN_PROGRESS = Gauge(
    "sportreview_requests_in_progress",
    "Nombre de requetes actuellement en cours de traitement"
)

# Jauge : taille du batch moyen (pour /predict/batch)
BATCH_SIZE = Histogram(
    "sportreview_batch_size",
    "Distribution des tailles de batch",
    buckets=[1, 5, 10, 25, 50, 100, 200, 500]
)

# Compteur de feedbacks reçus (corrections humaines)
FEEDBACK_COUNT = Counter(
    "sportreview_feedback_total",
    "Nombre de corrections de prediction soumises par les utilisateurs"
)


# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONS UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────

def record_prediction(label: str, confidence: float,
                       confidence_threshold: float = 0.6) -> None:
    """
    Enregistre les métriques d'une prédiction.

    À appeler après chaque prédiction réussie.

    Args:
        label               : label prédit (POSITIF, NEGATIF, etc.)
        confidence          : score de confiance de la prédiction
        confidence_threshold: seuil en dessous duquel la prédiction est peu fiable
    """
    # Incrémente le compteur pour ce label
    PREDICTION_COUNT.labels(label=label).inc()

    # Enregistre le score de confiance dans l'histogramme
    CONFIDENCE_HISTOGRAM.observe(confidence)

    # Si la confiance est faible, on le note
    if confidence < confidence_threshold:
        LOW_CONFIDENCE_COUNT.inc()


def record_request(endpoint: str, method: str,
                   status_code: int, duration: float) -> None:
    """
    Enregistre les métriques d'une requête HTTP.

    À appeler après chaque requête traitée.

    Args:
        endpoint    : nom de l'endpoint (ex: "/predict")
        method      : méthode HTTP (GET, POST)
        status_code : code de statut HTTP (200, 400, 500)
        duration    : durée de la requête en secondes
    """
    REQUEST_COUNT.labels(
        endpoint=endpoint,
        method=method,
        status_code=str(status_code)
    ).inc()

    REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)


class RequestTimer:
    """
    Context manager pour mesurer la durée d'une opération.

    Usage :
        with RequestTimer() as timer:
            result = do_something()
        duration = timer.duration
    """
    def __enter__(self):
        self.start = time.time()
        REQUESTS_IN_PROGRESS.inc()  # +1 requête en cours
        return self

    def __exit__(self, *args):
        self.duration = time.time() - self.start
        REQUESTS_IN_PROGRESS.dec()  # -1 requête en cours

    @property
    def duration_ms(self) -> float:
        """Durée en millisecondes."""
        return self.duration * 1000
