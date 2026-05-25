# -*- coding: utf-8 -*-
"""
test_api.py — Tests unitaires pour l'API FastAPI

Ces tests vérifient que tous les endpoints fonctionnent correctement.
Ils utilisent TestClient de FastAPI qui simule des requêtes HTTP
sans lancer un vrai serveur (rapide, pas de port réseau nécessaire).

Lancer :
    pytest tests/test_api.py -v
    pytest tests/test_api.py -v --tb=short   (affichage condensé des erreurs)
"""

import pytest
from fastapi.testclient import TestClient

# Importe l'application FastAPI — TestClient l'utilisera directement
from app.main import app

# Crée un client de test partagé entre tous les tests
client = TestClient(app)


# ─────────────────────────────────────────────────────────────────────────────
# TESTS HEALTH CHECK
# ─────────────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    """Tests de l'endpoint GET /health."""

    def test_health_returns_200(self):
        """Le health check doit retourner le code HTTP 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_status_is_healthy(self):
        """Le champ 'status' doit valoir 'healthy'."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_model_is_loaded(self):
        """Le modèle doit être chargé en mémoire."""
        response = client.get("/health")
        data = response.json()
        assert data["model_loaded"] is True

    def test_health_has_classes(self):
        """Le health check doit retourner la liste des classes."""
        response = client.get("/health")
        data = response.json()
        assert "classes" in data
        assert len(data["classes"]) == 4   # POSITIF, NEGATIF, NEUTRE, SPAM
        assert "POSITIF" in data["classes"]
        assert "SPAM" in data["classes"]

    def test_health_has_uptime(self):
        """Le health check doit inclure l'uptime du serveur."""
        response = client.get("/health")
        data = response.json()
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0


# ─────────────────────────────────────────────────────────────────────────────
# TESTS PRÉDICTION UNITAIRE
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictEndpoint:
    """Tests de l'endpoint POST /predict."""

    def test_predict_returns_200(self):
        """Une requête valide doit retourner 200."""
        response = client.post("/predict", json={"text": "Bon produit"})
        assert response.status_code == 200

    def test_predict_positive_review(self):
        """Un avis clairement positif doit être classifié POSITIF."""
        response = client.post("/predict", json={
            "text": "Excellent produit, tres bonne qualite, je recommande vivement !"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["label"] == "POSITIF"

    def test_predict_spam_review(self):
        """Un avis spam (achat non vérifié + patterns spam) doit être classifié SPAM."""
        response = client.post("/predict", json={
            "text": "PROMO INCROYABLE cliquez maintenant sur notre site web !!!!",
            "verified_purchase": False   # achat non vérifié = critère spam activé
        })
        assert response.status_code == 200
        data = response.json()
        assert data["label"] == "SPAM"

    def test_predict_response_has_all_fields(self):
        """La réponse doit contenir tous les champs requis."""
        response = client.post("/predict", json={"text": "Bon produit"})
        data = response.json()

        # Vérifie la présence de tous les champs
        assert "text" in data
        assert "label" in data
        assert "confidence" in data
        assert "probabilities" in data
        assert "is_reliable" in data
        assert "timestamp" in data

    def test_predict_probabilities_sum_to_one(self):
        """Les probabilités de toutes les classes doivent sommer à 1."""
        response = client.post("/predict", json={"text": "Produit correct"})
        data = response.json()
        probs = data["probabilities"]
        total = sum(probs.values())
        assert abs(total - 1.0) < 0.01   # tolérance de 1% pour les arrondis

    def test_predict_confidence_between_0_and_1(self):
        """Le score de confiance doit être entre 0 et 1."""
        response = client.post("/predict", json={"text": "Produit correct"})
        data = response.json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_label_is_valid_class(self):
        """Le label prédit doit être l'une des 4 classes valides."""
        valid_labels = {"POSITIF", "NEGATIF", "NEUTRE", "SPAM"}
        response = client.post("/predict", json={"text": "Test de prediction"})
        data = response.json()
        assert data["label"] in valid_labels

    def test_predict_with_explanation(self):
        """La prédiction avec explain=True doit inclure une explication."""
        response = client.post("/predict?explain=true", json={"text": "Bon produit"})
        assert response.status_code == 200
        data = response.json()
        # explanation peut être None si le modèle ne supporte pas SHAP
        # mais le champ doit exister
        assert "explanation" in data

    def test_predict_with_verified_purchase_flag(self):
        """La prédiction doit fonctionner avec le flag verified_purchase."""
        response = client.post("/predict", json={
            "text": "Super produit !",
            "verified_purchase": False
        })
        assert response.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# TESTS CAS D'ERREUR
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictErrorHandling:
    """Tests de gestion des erreurs sur /predict."""

    def test_empty_text_returns_422(self):
        """Un texte vide doit retourner une erreur de validation 422."""
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 422

    def test_whitespace_text_returns_422(self):
        """Un texte de seulement des espaces doit retourner 422."""
        response = client.post("/predict", json={"text": "   "})
        assert response.status_code == 422

    def test_missing_text_field_returns_422(self):
        """Une requête sans champ 'text' doit retourner 422."""
        response = client.post("/predict", json={"wrong_field": "test"})
        assert response.status_code == 422

    def test_text_too_short_returns_422(self):
        """Un texte trop court (< 5 caractères) doit retourner 422."""
        response = client.post("/predict", json={"text": "ok"})
        assert response.status_code == 422

    def test_text_too_long_returns_422(self):
        """Un texte trop long (> 5000 caractères) doit retourner 422."""
        long_text = "a" * 5001
        response = client.post("/predict", json={"text": long_text})
        assert response.status_code == 422


# ─────────────────────────────────────────────────────────────────────────────
# TESTS BATCH
# ─────────────────────────────────────────────────────────────────────────────

class TestBatchEndpoint:
    """Tests de l'endpoint POST /predict/batch."""

    def test_batch_returns_200(self):
        """Un batch valide doit retourner 200."""
        response = client.post("/predict/batch", json={
            "texts": ["Bon produit", "Mauvaise qualite", "SPAM promo"]
        })
        assert response.status_code == 200

    def test_batch_count_matches_input(self):
        """Le nombre de prédictions doit correspondre au nombre d'avis envoyés."""
        texts = ["Avis 1", "Avis 2", "Avis 3", "Avis 4"]
        response = client.post("/predict/batch", json={"texts": texts})
        data = response.json()
        assert data["count"] == 4
        assert len(data["predictions"]) == 4

    def test_batch_has_processing_time(self):
        """La réponse batch doit inclure le temps de traitement."""
        response = client.post("/predict/batch", json={
            "texts": ["Test 1", "Test 2"]
        })
        data = response.json()
        assert "processing_time_ms" in data
        assert data["processing_time_ms"] >= 0

    def test_batch_empty_list_returns_error(self):
        """Un batch vide doit retourner une erreur de validation."""
        response = client.post("/predict/batch", json={"texts": []})
        assert response.status_code == 422

    def test_batch_too_many_texts_returns_error(self):
        """Un batch avec trop de textes (> 500) doit retourner une erreur 400 ou 422.
        Pydantic intercepte la validation avant l'endpoint → retourne 422.
        Si la contrainte est dans l'endpoint → retourne 400.
        Les deux sont corrects (validation échouée)."""
        texts = [f"Avis numero {i}" for i in range(600)]  # > 500
        response = client.post("/predict/batch", json={"texts": texts})
        assert response.status_code in (400, 422)


# ─────────────────────────────────────────────────────────────────────────────
# TESTS MODEL INFO
# ─────────────────────────────────────────────────────────────────────────────

class TestModelInfoEndpoint:
    """Tests de l'endpoint GET /model/info."""

    def test_model_info_returns_200(self):
        """L'endpoint model/info doit retourner 200."""
        response = client.get("/model/info")
        assert response.status_code == 200

    def test_model_info_has_required_fields(self):
        """La réponse doit contenir les champs requis."""
        response = client.get("/model/info")
        data = response.json()
        assert "model_type" in data
        assert "classes" in data
        assert "loaded_at" in data

    def test_model_info_classes_are_valid(self):
        """Les classes dans model/info doivent être les 4 classes valides."""
        response = client.get("/model/info")
        data = response.json()
        expected = {"POSITIF", "NEGATIF", "NEUTRE", "SPAM"}
        assert set(data["classes"]) == expected


# ─────────────────────────────────────────────────────────────────────────────
# TESTS FEEDBACK
# ─────────────────────────────────────────────────────────────────────────────

class TestFeedbackEndpoint:
    """Tests de l'endpoint POST /feedback."""

    def test_feedback_returns_200(self):
        """Un feedback valide doit retourner 200."""
        response = client.post("/feedback", json={
            "text": "Super produit !",
            "predicted_label": "NEUTRE",
            "correct_label": "POSITIF"
        })
        assert response.status_code == 200

    def test_feedback_returns_id(self):
        """La réponse doit inclure un feedback_id."""
        response = client.post("/feedback", json={
            "text": "Produit decevant",
            "predicted_label": "NEUTRE",
            "correct_label": "NEGATIF"
        })
        data = response.json()
        assert "feedback_id" in data
        assert len(data["feedback_id"]) > 0

    def test_feedback_with_comment(self):
        """Le feedback avec commentaire doit fonctionner."""
        response = client.post("/feedback", json={
            "text": "Avis test",
            "predicted_label": "SPAM",
            "correct_label": "NEGATIF",
            "comment": "Ce n est pas du spam mais un vrai avis negatif"
        })
        assert response.status_code == 200


# ─────────────────────────────────────────────────────────────────────────────
# TESTS METRICS (PROMETHEUS)
# ─────────────────────────────────────────────────────────────────────────────

class TestMetricsEndpoint:
    """Tests de l'endpoint GET /metrics (Prometheus)."""

    def test_metrics_returns_200(self):
        """L'endpoint /metrics doit retourner 200."""
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_contains_sportreview_metrics(self):
        """Les métriques doivent contenir nos métriques personnalisées."""
        # Fait d'abord une prédiction pour générer des métriques
        client.post("/predict", json={"text": "Bon produit"})

        response = client.get("/metrics")
        content = response.text

        # Vérifie la présence de nos métriques personnalisées
        assert "sportreview_predictions_total" in content
        assert "sportreview_requests_total" in content
