# -*- coding: utf-8 -*-
"""
test_drift_detection.py — Tests unitaires pour src/drift_detection.py

Pourquoi tester drift_detection ?
→ Si compute_label_drift est bugué, on ne détecte pas les changements
  de distribution des données → le modèle se dégrade silencieusement
  en production sans qu'on s'en aperçoive.

Fonctions testées :
  - compute_label_drift   : calcul du drift par classe (logique pure pandas)
  - save_drift_summary    : sauvegarde JSON du résumé
  - load_reference_and_current : chargement des CSV de référence et courant

Fonctions NON testées ici :
  - compute_drift_report  : nécessite EvidentlyAI installé + modèle en prod
                            → testée manuellement via python src/drift_detection.py

Lancer :
    pytest tests/test_drift_detection.py -v
"""

import json
import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.drift_detection import (
    compute_label_drift,
    save_drift_summary,
    load_reference_and_current,
)


# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES — Créer de petits DataFrames de test
# ─────────────────────────────────────────────────────────────────────────────

def make_df(labels: list) -> pd.DataFrame:
    """Crée un DataFrame minimal avec une colonne 'label'."""
    return pd.DataFrame({
        "text":  [f"avis {i}" for i in range(len(labels))],
        "label": labels
    })


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — compute_label_drift
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeLabelDrift:
    """
    Teste compute_label_drift(reference_df, current_df).

    Cette fonction compare la distribution des labels entre deux datasets.
    C'est le cœur de la détection de drift — si les distributions changent,
    c'est que le comportement des utilisateurs ou le modèle a changé.
    """

    def test_drift_nul_si_distributions_identiques(self):
        """Si référence et courant ont la même distribution → drift = 0%."""
        labels = ["POSITIF", "POSITIF", "NEGATIF", "NEGATIF", "NEUTRE", "SPAM"]
        ref_df = make_df(labels)
        cur_df = make_df(labels)

        result = compute_label_drift(ref_df, cur_df)

        assert result["mean_drift_pct"] == pytest.approx(0.0, abs=0.1)
        assert result["alert"] == False

    def test_detecte_drift_fort(self):
        """Si une classe passe de 50% à 0%, le drift doit être élevé."""
        ref_labels = ["POSITIF"] * 50 + ["NEGATIF"] * 50
        cur_labels = ["POSITIF"] * 100   # NEGATIF a disparu

        ref_df = make_df(ref_labels)
        cur_df = make_df(cur_labels)

        result = compute_label_drift(ref_df, cur_df)

        # NEGATIF : 50% → 0% = drift de 50% → bien au-dessus du seuil
        assert result["mean_drift_pct"] > 10.0
        assert result["alert"] == True

    def test_alerte_si_drift_superieur_a_10_pourcent(self):
        """L'alerte doit se déclencher si le drift moyen dépasse 10%."""
        # Référence : distribution équilibrée
        ref_labels = ["POSITIF"] * 40 + ["NEGATIF"] * 40 + ["NEUTRE"] * 20
        # Courant : SPAM apparaît massivement (20% → signal fort)
        cur_labels = ["POSITIF"] * 30 + ["NEGATIF"] * 30 + ["NEUTRE"] * 10 + ["SPAM"] * 30

        ref_df = make_df(ref_labels)
        cur_df = make_df(cur_labels)

        result = compute_label_drift(ref_df, cur_df)

        assert result["alert"] == True

    def test_pas_alerte_si_drift_faible(self):
        """Pas d'alerte si la distribution change légèrement (< 10%)."""
        # Référence : 50% POSITIF, 50% NEGATIF
        ref_labels = ["POSITIF"] * 50 + ["NEGATIF"] * 50
        # Courant : 52% POSITIF, 48% NEGATIF → changement de 2%
        cur_labels = ["POSITIF"] * 52 + ["NEGATIF"] * 48

        ref_df = make_df(ref_labels)
        cur_df = make_df(cur_labels)

        result = compute_label_drift(ref_df, cur_df)

        assert result["alert"] == False

    def test_retourne_dict_vide_si_colonne_label_absente(self):
        """Doit retourner un dict vide si la colonne 'label' n'existe pas."""
        ref_df = pd.DataFrame({"text": ["avis 1", "avis 2"]})
        cur_df = pd.DataFrame({"text": ["avis 3", "avis 4"]})

        result = compute_label_drift(ref_df, cur_df)

        assert result == {}

    def test_contient_distribution_par_classe(self):
        """Le résultat doit contenir la distribution pour chaque classe."""
        ref_labels = ["POSITIF"] * 60 + ["NEGATIF"] * 40
        cur_labels = ["POSITIF"] * 70 + ["NEGATIF"] * 30

        ref_df = make_df(ref_labels)
        cur_df = make_df(cur_labels)

        result = compute_label_drift(ref_df, cur_df)

        assert "label_distribution" in result
        assert "POSITIF" in result["label_distribution"]
        assert "NEGATIF" in result["label_distribution"]

    def test_distribution_contient_pourcentages_corrects(self):
        """Les pourcentages de référence doivent être corrects."""
        ref_labels = ["POSITIF"] * 75 + ["NEGATIF"] * 25   # 75% / 25%
        cur_labels = ["POSITIF"] * 50 + ["NEGATIF"] * 50   # 50% / 50%

        ref_df = make_df(ref_labels)
        cur_df = make_df(cur_labels)

        result = compute_label_drift(ref_df, cur_df)

        positif_stats = result["label_distribution"]["POSITIF"]
        assert positif_stats["reference_pct"] == pytest.approx(75.0, abs=0.1)
        assert positif_stats["current_pct"]   == pytest.approx(50.0, abs=0.1)
        assert positif_stats["drift_abs"]     == pytest.approx(25.0, abs=0.1)

    def test_gere_classe_absente_dans_courant(self):
        """Doit gérer le cas où une classe est dans la référence mais pas dans le courant."""
        ref_labels = ["POSITIF"] * 50 + ["SPAM"] * 50
        cur_labels = ["POSITIF"] * 100   # SPAM absent du courant

        ref_df = make_df(ref_labels)
        cur_df = make_df(cur_labels)

        result = compute_label_drift(ref_df, cur_df)

        # SPAM doit apparaître dans les résultats avec current_pct = 0
        assert "SPAM" in result["label_distribution"]
        assert result["label_distribution"]["SPAM"]["current_pct"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — save_drift_summary
# ─────────────────────────────────────────────────────────────────────────────

class TestSaveDriftSummary:
    """
    Teste save_drift_summary(drift_results, label_drift).

    Cette fonction sauvegarde le résumé JSON utilisé par l'API FastAPI
    pour exposer les métriques de drift via /drift.
    """

    def test_cree_le_fichier_json(self, tmp_path):
        """Le fichier JSON doit être créé après l'appel."""
        drift_results = {"drift_score": 0.05, "drift_detected": False}
        label_drift   = {"mean_drift_pct": 2.0, "alert": False}

        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "src.drift_detection.DRIFT_REPORT_JSON",
            str(tmp_path / "drift_summary.json")
        ), __import__("unittest.mock", fromlist=["patch"]).patch(
            "src.drift_detection.REPORTS_DIR",
            str(tmp_path)
        ):
            save_drift_summary(drift_results, label_drift)

        assert (tmp_path / "drift_summary.json").exists()

    def test_json_contient_drift_results(self, tmp_path):
        """Le JSON doit contenir les résultats de drift."""
        drift_results = {"drift_score": 0.12, "drift_detected": True, "threshold": 0.10}
        label_drift   = {"mean_drift_pct": 15.0, "alert": True}

        json_path = tmp_path / "drift_summary.json"

        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "src.drift_detection.DRIFT_REPORT_JSON", str(json_path)
        ), __import__("unittest.mock", fromlist=["patch"]).patch(
            "src.drift_detection.REPORTS_DIR", str(tmp_path)
        ):
            save_drift_summary(drift_results, label_drift)

        with open(json_path) as f:
            saved = json.load(f)

        assert saved["drift_score"]    == 0.12
        assert saved["drift_detected"] is True
        assert saved["label_drift"]["alert"] is True

    def test_json_contient_timestamp(self, tmp_path):
        """Le JSON doit contenir un timestamp de génération."""
        json_path = tmp_path / "drift_summary.json"

        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "src.drift_detection.DRIFT_REPORT_JSON", str(json_path)
        ), __import__("unittest.mock", fromlist=["patch"]).patch(
            "src.drift_detection.REPORTS_DIR", str(tmp_path)
        ):
            save_drift_summary({}, {})

        with open(json_path) as f:
            saved = json.load(f)

        assert "generated_at" in saved


# ─────────────────────────────────────────────────────────────────────────────
# TESTS — load_reference_and_current
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadReferenceAndCurrent:
    """
    Teste load_reference_and_current(reference_path, current_path).

    Vérifie que les CSV sont bien chargés et que la colonne texte
    est correctement identifiée (text_clean si disponible, sinon text).
    """

    def test_charge_les_deux_csv(self, tmp_path):
        """Les deux DataFrames doivent être chargés correctement."""
        ref_csv = tmp_path / "train.csv"
        cur_csv = tmp_path / "test.csv"

        ref_csv.write_text("text,label\navis positif,POSITIF\navis negatif,NEGATIF\n")
        cur_csv.write_text("text,label\navis neutre,NEUTRE\n")

        ref_df, cur_df, text_col = load_reference_and_current(str(ref_csv), str(cur_csv))

        assert len(ref_df) == 2
        assert len(cur_df) == 1

    def test_utilise_text_clean_si_disponible(self, tmp_path):
        """Doit utiliser 'text_clean' si la colonne existe."""
        ref_csv = tmp_path / "train.csv"
        cur_csv = tmp_path / "test.csv"

        ref_csv.write_text("text,text_clean,label\navis,avis propre,POSITIF\n")
        cur_csv.write_text("text,text_clean,label\navis,avis net,NEGATIF\n")

        _, _, text_col = load_reference_and_current(str(ref_csv), str(cur_csv))

        assert text_col == "text_clean"

    def test_utilise_text_si_text_clean_absent(self, tmp_path):
        """Doit utiliser 'text' si 'text_clean' n'existe pas."""
        ref_csv = tmp_path / "train.csv"
        cur_csv = tmp_path / "test.csv"

        ref_csv.write_text("text,label\navis brut,POSITIF\n")
        cur_csv.write_text("text,label\nautre avis,NEGATIF\n")

        _, _, text_col = load_reference_and_current(str(ref_csv), str(cur_csv))

        assert text_col == "text"
