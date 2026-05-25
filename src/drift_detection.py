# -*- coding: utf-8 -*-
"""
drift_detection.py — Détection du drift avec EvidentlyAI

Le drift est la dégradation progressive des performances d'un modèle
en production, causée par un changement dans les données réelles.

Deux types de drift détectés :
1. Data Drift    : la distribution des textes entrants change
                   (ex: les gens utilisent un nouveau vocabulaire)
2. Target Drift  : la distribution des prédictions change
                   (ex: soudainement 60% d'avis SPAM au lieu de 20%)

Ce script génère un rapport HTML consultable dans le navigateur.

Usage :
    python src/drift_detection.py
    python src/drift_detection.py --reference data/processed/train.csv
                                  --current   data/processed/test.csv
"""

import argparse
import os
import sys
import json
from datetime import datetime, timezone

import pandas as pd
import numpy as np

# Ajoute le dossier racine au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    TRAIN_CSV, TEST_CSV, REPORTS_DIR,
    DRIFT_REPORT_HTML, DRIFT_REPORT_JSON,
    DRIFT_THRESHOLD, MODEL_PATH, CLASSES_PATH
)


def load_reference_and_current(reference_path: str, current_path: str):
    """
    Charge les datasets de référence et courant pour la comparaison.

    - Dataset de référence : les données d'entraînement (distribution "normale")
    - Dataset courant      : les nouvelles données en production

    Args:
        reference_path : chemin vers le CSV de référence (train)
        current_path   : chemin vers le CSV courant (test ou nouvelles données)

    Returns:
        reference_df, current_df : deux DataFrames pandas
    """
    print(f"Chargement du dataset de référence : {reference_path}")
    reference_df = pd.read_csv(reference_path)

    print(f"Chargement du dataset courant : {current_path}")
    current_df = pd.read_csv(current_path)

    # Utilise text_clean si disponible, sinon text brut
    text_col = "text_clean" if "text_clean" in reference_df.columns else "text"

    return reference_df, current_df, text_col


def add_model_predictions(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """
    Ajoute les prédictions du modèle au DataFrame.

    Pour calculer le target drift, on a besoin des prédictions
    du modèle sur les nouvelles données.

    Args:
        df       : DataFrame avec une colonne texte
        text_col : nom de la colonne texte

    Returns:
        DataFrame avec colonnes 'prediction' et 'confidence' ajoutées
    """
    import pickle

    if not os.path.exists(MODEL_PATH):
        print(f"Modèle non trouvé : {MODEL_PATH}")
        print("Les prédictions ne seront pas calculées.")
        df["prediction"] = "INCONNU"
        df["confidence"] = 0.0
        return df

    try:
        # Charge le modèle
        with open(MODEL_PATH, "rb") as f:
            pipeline = pickle.load(f)

        with open(CLASSES_PATH, "r") as f:
            classes = json.load(f)

        texts = df[text_col].fillna("").tolist()

        # Prédictions
        predictions = pipeline.predict(texts)
        df["prediction"] = predictions

        # Probabilités de confiance
        try:
            probas = pipeline.predict_proba(texts)
            df["confidence"] = probas.max(axis=1)
        except AttributeError:
            df["confidence"] = 1.0  # SVM sans calibration n'a pas predict_proba

    except Exception as e:
        print(f"Avertissement : impossible de charger le modèle ({type(e).__name__}).")
        print("Le rapport de drift sera généré sans les prédictions du modèle.")
        df["prediction"] = "INCONNU"
        df["confidence"] = 0.0

    return df


def compute_drift_report(reference_df: pd.DataFrame,
                          current_df: pd.DataFrame,
                          text_col: str) -> dict:
    """
    Calcule le rapport de drift avec EvidentlyAI.

    EvidentlyAI compare la distribution des données de référence
    et des données courantes, et calcule des tests statistiques
    pour détecter si les distributions ont changé significativement.

    Args:
        reference_df : données d'entraînement (référence)
        current_df   : nouvelles données en production
        text_col     : nom de la colonne texte

    Returns:
        dict avec les scores de drift
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
        from evidently.metrics import (
            ColumnDriftMetric,
            DatasetDriftMetric,
            DatasetMissingValuesMetric,
        )
    except ImportError:
        print("EvidentlyAI non installé. Lance : pip install evidently")
        return {}

    print("Calcul du rapport de drift avec EvidentlyAI...")

    # ── Préparation des features numériques ──────────────────────────────────
    # EvidentlyAI compare des features numériques — on utilise
    # les features extraites pendant le preprocessing

    feature_cols = [
        col for col in ["text_length", "word_count",
                         "exclamation_count", "uppercase_ratio", "has_url"]
        if col in reference_df.columns and col in current_df.columns
    ]

    # Si pas de features numériques, on en crée des basiques
    if not feature_cols:
        print("Features numériques non trouvées — calcul des features de base...")
        for df in [reference_df, current_df]:
            df["text_length"] = df[text_col].str.len()
            df["word_count"] = df[text_col].str.split().str.len()
        feature_cols = ["text_length", "word_count"]

    # Ajoute les prédictions pour le target drift
    reference_df = add_model_predictions(reference_df, text_col)
    current_df = add_model_predictions(current_df, text_col)

    # Encode les labels en nombres pour EvidentlyAI
    label_mapping = {"NEGATIF": 0, "NEUTRE": 1, "POSITIF": 2, "SPAM": 3}
    if "label" in reference_df.columns:
        reference_df["label_encoded"] = reference_df["label"].map(label_mapping)
    if "label" in current_df.columns:
        current_df["label_encoded"] = current_df["label"].map(label_mapping)
    if "prediction" in reference_df.columns:
        reference_df["prediction_encoded"] = reference_df["prediction"].map(label_mapping)
        current_df["prediction_encoded"] = current_df["prediction"].map(label_mapping)

    # Sélectionne les colonnes pour la comparaison
    # N'inclut confidence que si les prédictions ont réussi (colonne non uniforme)
    cols_to_compare = list(feature_cols)
    if "confidence" in reference_df.columns and reference_df["confidence"].nunique() > 1:
        cols_to_compare.append("confidence")
    if "label_encoded" in reference_df.columns:
        cols_to_compare.append("label_encoded")
    if "prediction_encoded" in reference_df.columns and reference_df["prediction_encoded"].nunique() > 1:
        cols_to_compare.append("prediction_encoded")

    ref_subset = reference_df[cols_to_compare].dropna()
    cur_subset = current_df[cols_to_compare].dropna()

    # ── Création du rapport Evidently ─────────────────────────────────────────
    report = Report(metrics=[
        DatasetDriftMetric(),           # drift global du dataset
        DatasetMissingValuesMetric(),   # valeurs manquantes
    ] + [
        ColumnDriftMetric(column_name=col)  # drift par colonne
        for col in feature_cols[:5]         # maximum 5 colonnes
    ])

    # Calcule le rapport en comparant référence vs courant
    report.run(reference_data=ref_subset, current_data=cur_subset)

    # ── Sauvegarde du rapport HTML ────────────────────────────────────────────
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report.save_html(DRIFT_REPORT_HTML)
    print(f"Rapport HTML sauvegardé : {DRIFT_REPORT_HTML}")

    # ── Extraction du score de drift ─────────────────────────────────────────
    report_dict = report.as_dict()
    drift_score = 0.0

    try:
        # Navigue dans la structure JSON du rapport pour trouver le score
        for metric in report_dict.get("metrics", []):
            if "DatasetDriftMetric" in str(metric.get("metric", "")):
                result = metric.get("result", {})
                # evidently 0.4.x → "share_of_drifted_columns" (pas "drift_share")
                drift_score = float(result.get("share_of_drifted_columns",
                                   result.get("drift_share", 0)))
                break
    except Exception:
        drift_score = 0.0

    return {
        "drift_score": drift_score,
        "drift_detected": bool(drift_score > DRIFT_THRESHOLD),
        "threshold": DRIFT_THRESHOLD,
        "n_reference": len(reference_df),
        "n_current": len(current_df),
        "report_path": DRIFT_REPORT_HTML,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def compute_label_drift(reference_df: pd.DataFrame,
                         current_df: pd.DataFrame) -> dict:
    """
    Calcule le drift de la distribution des labels (target drift).

    Compare la proportion de chaque classe entre les données
    de référence et les données courantes.
    Une forte variation peut indiquer un changement dans le comportement
    des utilisateurs ou une dégradation du modèle.

    Args:
        reference_df, current_df : DataFrames avec colonne 'label'

    Returns:
        dict avec la distribution des labels et le drift score
    """
    if "label" not in reference_df.columns or "label" not in current_df.columns:
        return {}

    # Calcule les distributions en pourcentage
    ref_dist = reference_df["label"].value_counts(normalize=True).to_dict()
    cur_dist = current_df["label"].value_counts(normalize=True).to_dict()

    # Calcule la différence absolue par classe
    all_labels = set(list(ref_dist.keys()) + list(cur_dist.keys()))
    drift_per_class = {}
    for label in all_labels:
        ref_pct = ref_dist.get(label, 0)
        cur_pct = cur_dist.get(label, 0)
        drift_per_class[label] = {
            "reference_pct": round(ref_pct * 100, 1),
            "current_pct": round(cur_pct * 100, 1),
            "drift_abs": round(abs(ref_pct - cur_pct) * 100, 1),
        }

    # Score de drift global = moyenne des dérives absolues
    mean_drift = np.mean([v["drift_abs"] for v in drift_per_class.values()])

    print("\nDrift de distribution des labels :")
    print(f"{'Label':<15} {'Reference':>12} {'Courant':>12} {'Derive':>10}")
    print("-" * 50)
    for label, stats in sorted(drift_per_class.items()):
        alert = " ⚠️ ALERTE" if stats["drift_abs"] > 10 else ""
        print(f"{label:<15} {stats['reference_pct']:>11.1f}% {stats['current_pct']:>11.1f}% {stats['drift_abs']:>9.1f}%{alert}")
    print(f"\nDrift moyen : {mean_drift:.1f}%")

    return {
        "label_distribution": drift_per_class,
        "mean_drift_pct": float(mean_drift),
        "alert": bool(mean_drift > 10),
    }


def save_drift_summary(drift_results: dict, label_drift: dict) -> None:
    """
    Sauvegarde un résumé JSON du rapport de drift.

    Ce fichier JSON est utilisé par l'API FastAPI pour exposer
    les métriques de drift via l'endpoint /drift.

    Args:
        drift_results : résultats EvidentlyAI
        label_drift   : drift de distribution des labels
    """
    summary = {
        **drift_results,
        "label_drift": label_drift,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    class _NumpyEncoder(json.JSONEncoder):
        """Encodeur JSON qui gère les types numpy non sérialisables."""
        def default(self, obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.bool_):   return bool(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(DRIFT_REPORT_JSON, "w") as f:
        json.dump(summary, f, indent=2, cls=_NumpyEncoder)

    print(f"\nRésumé JSON sauvegardé : {DRIFT_REPORT_JSON}")


def main():
    parser = argparse.ArgumentParser(
        description="Calcule le drift entre données de référence et courantes"
    )
    parser.add_argument(
        "--reference", default=TRAIN_CSV,
        help=f"Dataset de référence (défaut: {TRAIN_CSV})"
    )
    parser.add_argument(
        "--current", default=TEST_CSV,
        help=f"Dataset courant (défaut: {TEST_CSV})"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("DÉTECTION DU DRIFT — SportReview AI")
    print("=" * 60)

    # Charge les données
    reference_df, current_df, text_col = load_reference_and_current(
        args.reference, args.current
    )

    # Calcule le drift des features
    drift_results = compute_drift_report(reference_df, current_df, text_col)

    # Calcule le drift des labels
    label_drift = compute_label_drift(reference_df, current_df)

    # Sauvegarde le résumé
    if drift_results:
        save_drift_summary(drift_results, label_drift)

        print("\n" + "=" * 60)
        if drift_results.get("drift_detected"):
            print("⚠️  DRIFT DÉTECTÉ ! Le modèle doit être ré-entraîné.")
            print(f"Score de drift : {drift_results['drift_score']:.3f} > seuil={DRIFT_THRESHOLD}")
        else:
            print("✓ Pas de drift significatif détecté.")
            print(f"Score de drift : {drift_results.get('drift_score', 0):.3f}")

        print(f"\nRapport HTML : {DRIFT_REPORT_HTML}")
        print("Ouvre ce fichier dans ton navigateur pour la visualisation complète.")
        print("=" * 60)


if __name__ == "__main__":
    main()
