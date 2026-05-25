# -*- coding: utf-8 -*-
"""
test_promote_model.py — Tests unitaires pour src/promote_model.py

Pourquoi tester promote_model ?
→ C'est lui qui décide quel modèle va en production.
→ Un bug ici = mauvais modèle déployé = prédictions silencieusement fausses.

Comment on teste sans un vrai serveur MLflow ?
→ On utilise unittest.mock pour simuler (mocker) le client MLflow.
→ On crée de faux objets "run" qui imitent la structure retournée par MLflow.
→ Ainsi les tests sont rapides (< 1s) et indépendants de l'environnement.

Lancer :
    pytest tests/test_promote_model.py -v
"""

import json
import os
import sys
import pytest
from unittest.mock import MagicMock, patch, mock_open

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.promote_model import get_best_run, get_production_model_f1, promote_model


# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES — Fabriquer de faux runs MLflow
# ─────────────────────────────────────────────────────────────────────────────

def make_fake_run(run_id: str, model_name: str, f1: float, accuracy: float = 0.80):
    """
    Crée un faux objet MLflow Run pour les tests.

    MLflow retourne des objets avec .info.run_id, .data.metrics, .data.params.
    On crée un MagicMock qui imite exactement cette structure.
    """
    run = MagicMock()
    run.info.run_id = run_id
    run.data.params = {"model_name": model_name}
    run.data.tags   = {}
    run.data.metrics = {
        "f1_weighted": f1,
        "f1_macro":    f1 - 0.05,
        "accuracy":    accuracy,
        "f1_negatif":  f1 + 0.02,
        "f1_neutre":   f1 - 0.10,
        "f1_positif":  f1 + 0.05,
        "f1_spam":     f1 - 0.20,
    }
    return run


def make_fake_experiment(experiment_id: str = "1"):
    """Crée un faux objet MLflow Experiment."""
    exp = MagicMock()
    exp.experiment_id = experiment_id
    return exp


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — get_best_run
# ─────────────────────────────────────────────────────────────────────────────

class TestGetBestRun:
    """
    Teste get_best_run(experiment_name).

    Cette fonction cherche dans MLflow le run avec le meilleur F1.
    On mocke MlflowClient pour simuler différents scénarios.
    """

    def test_retourne_le_run_avec_meilleur_f1(self):
        """Doit retourner le run avec le F1 le plus élevé."""
        fake_run = make_fake_run("run-abc", "tfidf_lr", f1=0.70)

        with patch("src.promote_model.MlflowClient") as MockClient:
            client = MockClient.return_value
            client.get_experiment_by_name.return_value = make_fake_experiment()
            client.search_runs.return_value = [fake_run]

            result = get_best_run("sportreview-ai")

        assert result.info.run_id == "run-abc"
        assert result.data.metrics["f1_weighted"] == 0.70

    def test_leve_erreur_si_experience_introuvable(self):
        """Doit lever ValueError si l'expérience MLflow n'existe pas."""
        with patch("src.promote_model.MlflowClient") as MockClient:
            client = MockClient.return_value
            client.get_experiment_by_name.return_value = None  # expérience absente

            with pytest.raises(ValueError, match="introuvable"):
                get_best_run("experience-inexistante")

    def test_leve_erreur_si_aucun_run(self):
        """Doit lever ValueError si l'expérience existe mais n'a aucun run."""
        with patch("src.promote_model.MlflowClient") as MockClient:
            client = MockClient.return_value
            client.get_experiment_by_name.return_value = make_fake_experiment()
            client.search_runs.return_value = []  # liste vide

            with pytest.raises(ValueError, match="Aucun run"):
                get_best_run("sportreview-ai")

    def test_appelle_search_runs_avec_tri_par_f1(self):
        """Vérifie que MLflow est appelé avec le bon tri (f1_weighted DESC)."""
        fake_run = make_fake_run("run-xyz", "distilbert", f1=0.71)

        with patch("src.promote_model.MlflowClient") as MockClient:
            client = MockClient.return_value
            client.get_experiment_by_name.return_value = make_fake_experiment()
            client.search_runs.return_value = [fake_run]

            get_best_run("sportreview-ai")

            # Vérifie que search_runs a été appelé avec le bon order_by
            call_kwargs = client.search_runs.call_args
            assert "f1_weighted DESC" in str(call_kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — get_production_model_f1
# ─────────────────────────────────────────────────────────────────────────────

class TestGetProductionModelF1:
    """
    Teste get_production_model_f1(client).

    Cette fonction récupère le F1 du modèle actuellement en production.
    Si aucun modèle n'est en prod, elle doit retourner 0.0.
    """

    def test_retourne_0_si_aucun_modele_en_production(self):
        """Doit retourner 0.0 si aucun modèle n'est en production."""
        mock_client = MagicMock()
        mock_client.get_latest_versions.return_value = []  # aucune version en prod

        result = get_production_model_f1(mock_client)

        assert result == 0.0

    def test_retourne_f1_du_modele_en_production(self):
        """Doit retourner le F1 du modèle actuellement en production."""
        mock_client = MagicMock()

        # Simule une version en production
        prod_version = MagicMock()
        prod_version.run_id = "run-prod-123"
        mock_client.get_latest_versions.return_value = [prod_version]

        # Simule les métriques du run en production
        prod_run = make_fake_run("run-prod-123", "tfidf_svm", f1=0.66)
        mock_client.get_run.return_value = prod_run

        result = get_production_model_f1(mock_client)

        assert result == pytest.approx(0.66, abs=0.001)

    def test_retourne_0_si_exception(self):
        """Doit retourner 0.0 en cas d'erreur (ex: MLflow non disponible)."""
        mock_client = MagicMock()
        mock_client.get_latest_versions.side_effect = Exception("MLflow unavailable")

        result = get_production_model_f1(mock_client)

        assert result == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — promote_model (logique de décision)
# ─────────────────────────────────────────────────────────────────────────────

class TestPromoteModel:
    """
    Teste promote_model(threshold).

    C'est la fonction principale — elle orchestre toute la logique de promotion.
    On teste les cas de REFUS (retourne False) sans toucher au système de fichiers.
    """

    def test_retourne_false_si_experience_introuvable(self):
        """Doit retourner False si l'expérience MLflow n'existe pas."""
        with patch("src.promote_model.MlflowClient") as MockClient:
            client = MockClient.return_value
            client.get_experiment_by_name.return_value = None

            result = promote_model(threshold=0.65)

        assert result is False

    def test_retourne_false_si_aucun_run(self):
        """Doit retourner False si aucun run n'est trouvé dans l'expérience."""
        with patch("src.promote_model.MlflowClient") as MockClient:
            client = MockClient.return_value
            client.get_experiment_by_name.return_value = make_fake_experiment()
            client.search_runs.return_value = []

            result = promote_model(threshold=0.65)

        assert result is False

    def test_retourne_false_si_f1_sous_seuil(self):
        """Doit retourner False si le meilleur F1 est en dessous du seuil."""
        fake_run = make_fake_run("run-bad", "tfidf_lr", f1=0.50)

        with patch("src.promote_model.MlflowClient") as MockClient:
            client = MockClient.return_value
            client.get_experiment_by_name.return_value = make_fake_experiment()
            client.search_runs.return_value = [fake_run]

            # Seuil à 0.65 → F1=0.50 doit être refusé
            result = promote_model(threshold=0.65)

        assert result is False

    def test_retourne_false_si_modele_pas_meilleur_que_prod(self, tmp_path):
        """
        Doit retourner False si le nouveau modèle n'est pas meilleur que
        le modèle actuellement en production.
        """
        # Simule un modèle en prod avec F1=0.80 (meilleur que le candidat)
        metadata_prod = {"f1_weighted": 0.80, "model_name": "tfidf_svm"}
        metadata_file = tmp_path / "metadata.json"
        metadata_file.write_text(json.dumps(metadata_prod))

        fake_run = make_fake_run("run-new", "tfidf_lr", f1=0.70)

        with patch("src.promote_model.MlflowClient") as MockClient, \
             patch("src.promote_model.MODELS_DIR", str(tmp_path)):

            client = MockClient.return_value
            client.get_experiment_by_name.return_value = make_fake_experiment()
            client.search_runs.return_value = [fake_run]

            # F1=0.70 < prod F1=0.80 → refus
            result = promote_model(threshold=0.65)

        assert result is False

    def test_seuil_personnalise_est_respecte(self):
        """Un seuil personnalisé doit être pris en compte."""
        fake_run = make_fake_run("run-ok", "tfidf_lr", f1=0.75)

        with patch("src.promote_model.MlflowClient") as MockClient:
            client = MockClient.return_value
            client.get_experiment_by_name.return_value = make_fake_experiment()
            client.search_runs.return_value = [fake_run]

            # Avec seuil 0.80 → F1=0.75 doit être refusé
            result_refuse = promote_model(threshold=0.80)

        assert result_refuse is False
