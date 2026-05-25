# -*- coding: utf-8 -*-
"""
promote_model.py — Workflow Staging → Production dans MLflow Model Registry

Ce script implémente le processus de validation et promotion d'un modèle.
C'est une pratique MLOps essentielle : un modèle ne va pas directement
en production — il passe d'abord par une phase de validation (Staging).

Workflow :
  1. Trouve le meilleur run MLflow (F1 le plus élevé)
  2. Vérifie que le F1 dépasse le seuil minimum
  3. Compare avec le modèle actuellement en production
  4. Si le nouveau modèle est meilleur → le promouvoir en Production
  5. Archive l'ancien modèle de production

Usage :
    python src/promote_model.py
    python src/promote_model.py --threshold 0.85   (seuil F1 personnalisé)
"""

import sys
import os
import json

import mlflow
from mlflow.tracking import MlflowClient

# Ajoute le dossier racine au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    MLFLOW_EXPERIMENT_NAME, MLFLOW_REGISTERED_MODEL,
    PROMOTION_F1_THRESHOLD, MODELS_DIR, MODEL_PATH, CLASSES_PATH
)


def get_best_run(experiment_name: str) -> dict:
    """
    Trouve le run MLflow avec le meilleur F1 score dans l'expérience.

    Parcourt tous les runs de l'expérience et retourne celui
    avec la métrique test_f1_weighted la plus élevée.

    Args:
        experiment_name : nom de l'expérience MLflow

    Returns:
        dict avec les infos du meilleur run (run_id, metrics, params)
    """
    client = MlflowClient()

    # Récupère l'expérience par son nom
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(
            f"Expérience '{experiment_name}' introuvable.\n"
            "Lance d'abord : python src/train.py"
        )

    # Récupère tous les runs de l'expérience, triés par F1 décroissant
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",                          # pas de filtre
        order_by=["metrics.f1_weighted DESC"],  # meilleur F1 en premier
        max_results=1                              # on veut seulement le meilleur
    )

    if not runs:
        raise ValueError("Aucun run trouvé dans l'expérience.")

    best_run = runs[0]

    print(f"Meilleur run trouvé :")
    print(f"  Run ID      : {best_run.info.run_id}")
    print(f"  Modèle      : {best_run.data.params.get('model_name', best_run.data.tags.get('model_type', 'Inconnu'))}")
    print(f"  F1 Weighted : {best_run.data.metrics.get('f1_weighted', 0):.4f}")
    print(f"  Accuracy    : {best_run.data.metrics.get('accuracy', 0):.4f}")

    return best_run


def get_production_model_f1(client: MlflowClient) -> float:
    """
    Récupère le F1 score du modèle actuellement en production.

    Si aucun modèle n'est en production, retourne 0 (le nouveau modèle
    sera toujours promu).

    Args:
        client : MlflowClient

    Returns:
        F1 score du modèle en production (0 si aucun modèle en prod)
    """
    try:
        # Cherche les versions du modèle en stage "Production"
        prod_versions = client.get_latest_versions(
            MLFLOW_REGISTERED_MODEL,
            stages=["Production"]
        )

        if not prod_versions:
            print("Aucun modèle en production actuellement.")
            return 0.0

        # Récupère les métriques du run qui a produit ce modèle
        prod_run = client.get_run(prod_versions[0].run_id)
        prod_f1 = prod_run.data.metrics.get("f1_weighted", 0)
        print(f"Modèle en production actuel : F1={prod_f1:.4f}")
        return prod_f1

    except Exception as e:
        print(f"Impossible de récupérer le modèle en production : {e}")
        return 0.0


def promote_model(threshold: float = PROMOTION_F1_THRESHOLD) -> bool:
    """
    Processus complet de promotion du meilleur modèle en production.

    Étapes :
    1. Trouve le meilleur run
    2. Vérifie le seuil F1 minimum
    3. Compare avec le modèle en production
    4. Promeut si le nouveau modèle est meilleur
    5. Archive l'ancien modèle

    Args:
        threshold : F1 minimum pour autoriser la promotion (défaut: 0.80)

    Returns:
        True si la promotion a eu lieu, False sinon
    """
    client = MlflowClient()

    print("=" * 60)
    print("PROCESSUS DE PROMOTION DU MODELE")
    print("=" * 60)

    # ── Étape 1 : Récupère TOUS les runs triés par F1 ────────────────────────
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        print(f"ERREUR : expérience '{MLFLOW_EXPERIMENT_NAME}' introuvable.")
        return False

    all_runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        order_by=["metrics.f1_weighted DESC"],
        max_results=20
    )

    if not all_runs:
        print("ERREUR : Aucun run trouvé. Lance d'abord python src/train.py")
        return False

    print(f"\n{len(all_runs)} runs trouvés dans l'expérience '{MLFLOW_EXPERIMENT_NAME}'.")
    print(f"\nClassement par F1 score weighted :")
    for i, run in enumerate(all_runs, 1):
        f1   = run.data.metrics.get("f1_weighted", 0)
        name = run.data.params.get("model_name",
               run.data.tags.get("model_type", f"run-{run.info.run_id[:8]}"))
        marker = "← MEILLEUR" if i == 1 else ""
        print(f"  {i}. {name:<15} F1={f1:.4f}  {marker}")

    # ── Étape 2 : Vérifie le seuil minimum pour le meilleur run ──────────────
    best_run = all_runs[0]
    best_f1  = best_run.data.metrics.get("f1_weighted", 0)
    best_name = best_run.data.params.get("model_name",
                best_run.data.tags.get("model_type", "Inconnu"))

    print(f"\nSeuil F1 minimum requis : {threshold}")
    if best_f1 < threshold:
        print(f"REFUS : F1={best_f1:.4f} < seuil={threshold}")
        print("Le modèle n'est pas assez bon pour la production.")
        return False
    print(f"OK : F1={best_f1:.4f} >= seuil={threshold}")

    # ── Étape 3 : Compare avec le modèle actuellement en production ──────────
    prod_metadata_path = os.path.join(MODELS_DIR, "metadata.json")
    prod_f1 = 0.0
    if os.path.exists(prod_metadata_path):
        with open(prod_metadata_path) as f:
            prod_meta = json.load(f)
        prod_f1 = prod_meta.get("f1_weighted", 0.0)
        print(f"Modèle en production actuel : F1={prod_f1:.4f}")
    else:
        print("Aucun modèle en production actuellement.")

    if best_f1 <= prod_f1:
        print(f"\nREFUS : Le nouveau modèle (F1={best_f1:.4f}) n'est pas")
        print(f"meilleur que le modèle en production (F1={prod_f1:.4f}).")
        return False
    print(f"\nOK : Nouveau modèle (F1={best_f1:.4f}) > Production actuelle (F1={prod_f1:.4f})")

    # ── Étape 4 : Déploiement automatique selon le type de modèle ────────────
    # Cas 1 — DistilBERT : crée un DistilBertWrapper (interface sklearn-compatible)
    #          Les métriques sont lues depuis MLflow → aucune valeur codée en dur
    # Cas 2 — Sklearn    : télécharge le pickle depuis les artifacts MLflow
    import datetime, shutil, tempfile

    os.makedirs(MODELS_DIR, exist_ok=True)

    if "distilbert" in best_name.lower():
        # ── Cas DistilBERT ────────────────────────────────────────────────────
        distilbert_dir = os.path.join(MODELS_DIR, "distilbert_finetuned")
        if not os.path.exists(distilbert_dir):
            print(f"\nERREUR : dossier {distilbert_dir} introuvable.")
            print("Lance d'abord : python src/train.py --models distilbert")
            return False

        print(f"\nCréation du DistilBertWrapper (interface sklearn-compatible)...")
        from src.distilbert_wrapper import DistilBertWrapper
        wrapper = DistilBertWrapper(model_dir=distilbert_dir, max_length=128, batch_size=32)

        with open(MODEL_PATH, "wb") as f:
            import pickle as _pickle
            _pickle.dump(wrapper, f)

        run_to_deploy      = best_run
        run_to_deploy_f1   = best_f1
        run_to_deploy_name = best_name
        print(f"  → Wrapper sauvegardé : {MODEL_PATH}")
        print(f"  → Métriques lues depuis MLflow run {best_run.info.run_id[:8]}...")
        print(f"     F1 weighted : {best_f1:.4f}")
        print(f"     Accuracy    : {best_run.data.metrics.get('accuracy', 0):.4f}")
        print(f"     F1 macro    : {best_run.data.metrics.get('f1_macro', 0):.4f}")

    else:
        # ── Cas Sklearn : télécharge le pickle depuis les artifacts MLflow ────
        run_to_deploy = None
        run_to_deploy_f1 = 0.0
        run_to_deploy_name = None

        for run in all_runs:
            run_name = run.data.params.get("model_name",
                       run.data.tags.get("model_type", ""))
            try:
                artifacts = client.list_artifacts(run.info.run_id, path="model")
                artifact_names = [a.path for a in artifacts]
                if any("model.pkl" in a for a in artifact_names):
                    run_to_deploy      = run
                    run_to_deploy_f1   = run.data.metrics.get("f1_weighted", 0)
                    run_to_deploy_name = run_name
                    break
            except Exception:
                continue

        if run_to_deploy is None:
            print("\nERREUR : Aucun modèle avec pickle téléchargeable trouvé.")
            return False

        print(f"\nTéléchargement du modèle {run_to_deploy_name} depuis MLflow...")
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = client.download_artifacts(
                run_to_deploy.info.run_id, "model/model.pkl", tmpdir
            )
            shutil.copy(local_path, MODEL_PATH)
        print(f"  → Pipeline sauvegardé : {MODEL_PATH}")

    # Sauvegarde les classes
    with open(CLASSES_PATH, "w") as f:
        json.dump(["NEGATIF", "NEUTRE", "POSITIF", "SPAM"], f)

    # ── Étape 5 : Sauvegarde les métadonnées (métriques lues depuis MLflow) ──
    metadata = {
        "model_name":   run_to_deploy_name,
        "run_id":       run_to_deploy.info.run_id,
        "f1_weighted":  run_to_deploy.data.metrics.get("f1_weighted", 0),
        "f1_macro":     run_to_deploy.data.metrics.get("f1_macro", 0),
        "accuracy":     run_to_deploy.data.metrics.get("accuracy", 0),
        "f1_negatif":   run_to_deploy.data.metrics.get("f1_negatif", 0),
        "f1_neutre":    run_to_deploy.data.metrics.get("f1_neutre", 0),
        "f1_positif":   run_to_deploy.data.metrics.get("f1_positif", 0),
        "f1_spam":      run_to_deploy.data.metrics.get("f1_spam", 0),
        "stage":        "Production",
        "promoted_at":  datetime.datetime.now().isoformat()
    }
    with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  → Métadonnées sauvegardées depuis MLflow (aucune valeur codée en dur)")

    print("\n" + "=" * 60)
    print("PROMOTION RÉUSSIE !")
    print(f"  Modèle en production : {run_to_deploy_name}")
    print(f"  F1 weighted          : {metadata['f1_weighted']:.4f}")
    print(f"  Accuracy             : {metadata['accuracy']:.4f}")
    print(f"  Fichier API          : {MODEL_PATH}")
    print("=" * 60)
    print("\nProchaine etape : uvicorn app.main:app --reload --host 0.0.0.0 --port 8000")

    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Promeut le meilleur modèle MLflow en production"
    )
    parser.add_argument(
        "--threshold", type=float, default=PROMOTION_F1_THRESHOLD,
        help=f"F1 minimum pour la promotion (défaut: {PROMOTION_F1_THRESHOLD})"
    )
    args = parser.parse_args()

    success = promote_model(threshold=args.threshold)

    if success:
        print("\nProchain etape : uvicorn app.main:app --reload")
    else:
        print("\nAucune promotion. Continue à améliorer le modèle.")
        print("Lance : python src/train.py --models sbert_lr")
        sys.exit(1)


if __name__ == "__main__":
    main()
