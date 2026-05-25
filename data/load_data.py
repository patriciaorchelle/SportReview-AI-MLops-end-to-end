# -*- coding: utf-8 -*-
"""
load_data.py — Téléchargement et préparation du dataset Amazon Reviews

Ce script télécharge le dataset Amazon Sports & Outdoors depuis HuggingFace,
applique les règles de labellisation (note → POSITIF/NEGATIF/NEUTRE/SPAM),
et sauvegarde les données brutes en CSV pour le preprocessing Spark.

Usage :
    python data/load_data.py
    python data/load_data.py --max_samples 10000   (pour tester rapidement)
"""

import argparse
import os
import re
import sys
import pandas as pd

# Ajoute le dossier racine au path pour pouvoir importer src/config.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    DATA_RAW_DIR, RAW_CSV,
    HF_DATASET_NAME, HF_DATASET_CONFIG,
    RATING_TO_LABEL, LABEL_SPAM,
    MAX_SAMPLES
)


# ─────────────────────────────────────────────────────────────────────────────
# RÈGLES DE DÉTECTION DU SPAM
# Ces patterns sont des expressions régulières qui détectent
# les avis promotionnels ou suspects
# ─────────────────────────────────────────────────────────────────────────────

SPAM_PATTERNS = [
    r"http[s]?://",              # contient un lien URL
    r"www\.",                    # contient www.
    r"click here",               # appel à l'action typique du spam
    r"buy now",                  # "achetez maintenant"
    r"discount code",            # code promo
    r"promo code",               # code promo
    r"free shipping",            # livraison gratuite (souvent spam)
    r"limited offer",            # offre limitée
    r"contact me",               # contact direct suspect
    r"message me",               # message privé suspect
    r"\$\d+",                    # montant en dollars (ex: $50 off)
    r"!!{3,}",                   # 3 points d'exclamation ou plus !!!
    r"[A-Z]{10,}",               # 10 majuscules consécutives ou plus (CRIS)
]

# Compile les patterns une seule fois pour la performance
SPAM_REGEX = re.compile("|".join(SPAM_PATTERNS), re.IGNORECASE)


def is_spam(row: pd.Series) -> bool:
    """
    Détecte si un avis est du spam en combinant plusieurs critères.

    Un avis est considéré SPAM si :
    1. L'achat n'est PAS vérifié (non vérifié = suspect)
    2. ET le texte contient un pattern promotionnel

    Cette approche évite de classer tous les avis non vérifiés en spam —
    certains avis légitimes ne sont pas vérifiés.

    Args:
        row : une ligne du DataFrame avec colonnes 'text' et 'verified_purchase'

    Returns:
        True si l'avis est du spam, False sinon
    """
    text = str(row.get("text", ""))
    verified = row.get("verified_purchase", True)

    # Si l'achat n'est pas vérifié ET que le texte contient un pattern spam
    if not verified and SPAM_REGEX.search(text):
        return True

    # Avis trop court et non vérifié = souvent du faux avis
    if not verified and len(text.strip()) < 20:
        return True

    return False


def assign_label(row: pd.Series) -> str:
    """
    Assigne une classe à un avis selon ses caractéristiques.

    Ordre de priorité :
    1. SPAM (détecté par règles) — priorité absolue
    2. POSITIF/NEGATIF/NEUTRE selon la note

    Args:
        row : une ligne du DataFrame

    Returns:
        La classe : 'POSITIF', 'NEGATIF', 'NEUTRE', ou 'SPAM'
    """
    # Vérifie d'abord si c'est du spam
    if is_spam(row):
        return LABEL_SPAM

    # Sinon, conversion note → label
    rating = int(row.get("rating", 3))
    return RATING_TO_LABEL.get(rating, "NEUTRE")


def load_amazon_reviews(max_samples: int = None) -> pd.DataFrame:
    """
    Télécharge le dataset Amazon Sports & Outdoors depuis HuggingFace.

    HuggingFace met en cache le dataset localement après le premier téléchargement
    → les téléchargements suivants sont instantanés.

    Args:
        max_samples : nombre max d'exemples à garder (None = tout)

    Returns:
        DataFrame avec colonnes : text, rating, verified_purchase, label
    """
    print("Telechargement du dataset Amazon Sports & Outdoors...")
    print("(Le premier telechargement peut prendre quelques minutes)")

    # ── Chargement du dataset ─────────────────────────────────────────────────
    # NOTE VERSION : ce dataset utilise un script de chargement personnalisé.
    # Il requiert la librairie datasets < 3.0 avec trust_remote_code=True.
    # Si tu as une erreur "Dataset scripts are no longer supported", exécute :
    #   pip install "datasets<3.0" --force-reinstall
    #
    # Colonnes disponibles dans le dataset :
    #   rating            → note 1.0-5.0 étoiles
    #   title             → titre de l'avis
    #   text              → texte complet de l'avis
    #   verified_purchase → achat vérifié (True/False)
    #   helpful_vote      → nombre de votes "utile"
    #   asin              → identifiant du produit Amazon
    #   timestamp         → date de l'avis

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "La librairie 'datasets' n'est pas installée.\n"
            "Lance : pip install 'datasets<3.0'"
        )

    # Vérifie la version de datasets
    import datasets as _ds
    _version = tuple(int(x) for x in _ds.__version__.split(".")[:2])
    if _version >= (3, 0):
        raise RuntimeError(
            f"Version datasets incompatible : {_ds.__version__}\n"
            "Ce dataset nécessite datasets < 3.0.\n"
            "Lance : pip install 'datasets<3.0' --force-reinstall"
        )

    print(f"Librairie datasets v{_ds.__version__} — OK")
    print(f"Téléchargement de {HF_DATASET_NAME} / {HF_DATASET_CONFIG}...")

    dataset = load_dataset(
        HF_DATASET_NAME,
        HF_DATASET_CONFIG,
        trust_remote_code=True,
        split="full"   # prend tout (pas de split train/test par défaut)
    )

    print(f"Dataset téléchargé : {len(dataset):,} avis au total")

    # Convertit en DataFrame pandas pour la manipulation
    df = dataset.to_pandas()

    # ── Sélection et normalisation des colonnes utiles ────────────────────────
    # Le fichier JSONL brut contient : rating, title, text, verified_purchase,
    # helpful_vote, asin, parent_asin, user_id, timestamp, images
    # On garde seulement les colonnes dont on a besoin pour le pipeline ML.

    # Texte principal de l'avis
    if "text" in df.columns:
        df = df.rename(columns={"text": "review_text"})
    elif "reviewText" in df.columns:
        df = df.rename(columns={"reviewText": "review_text"})
    else:
        # Cherche une colonne contenant du texte
        text_cols = [c for c in df.columns if "text" in c.lower() or "review" in c.lower()]
        if text_cols:
            df = df.rename(columns={text_cols[0]: "review_text"})
        else:
            raise ValueError(
                f"Colonne texte introuvable. Colonnes disponibles : {list(df.columns)}"
            )

    # Note (1 à 5 étoiles)
    if "rating" not in df.columns and "overall" in df.columns:
        df = df.rename(columns={"overall": "rating"})

    # Achat vérifié
    if "verified_purchase" not in df.columns and "verified" in df.columns:
        df = df.rename(columns={"verified": "verified_purchase"})
    elif "verified_purchase" not in df.columns:
        # Si la colonne n'existe pas, on assume que tous les achats sont vérifiés
        df["verified_purchase"] = True

    # Garde seulement les colonnes nécessaires
    df = df[["review_text", "rating", "verified_purchase"]].copy()

    # ── Nettoyage de base ─────────────────────────────────────────────────────
    # Supprime les lignes avec texte manquant ou note manquante
    df = df.dropna(subset=["review_text", "rating"])

    # Convertit la note en entier
    df["rating"] = df["rating"].astype(int)

    # Garde seulement les notes valides (1 à 5)
    df = df[df["rating"].between(1, 5)]

    # Supprime les textes trop courts (moins de 10 caractères)
    df = df[df["review_text"].str.len() >= 10]

    # Renomme pour la suite du pipeline
    df = df.rename(columns={"review_text": "text"})

    # ── Limite le nombre d'exemples ──────────────────────────────────────────
    if max_samples and len(df) > max_samples:
        # Échantillonnage stratifié par note pour garder la distribution
        # Note : on utilise une boucle explicite pour éviter les incompatibilités
        # de pandas avec include_groups (qui supprime la colonne 'rating' du résultat)
        samples = []
        for _, group in df.groupby("rating"):
            n = min(len(group), max_samples // 5)
            samples.append(group.sample(n, random_state=42))
        df = pd.concat(samples).reset_index(drop=True)
        print(f"Dataset reduit a {len(df)} exemples (echantillonnage stratifie)")

    # ── Assignation des labels ────────────────────────────────────────────────
    print("Assignation des labels (POSITIF/NEGATIF/NEUTRE/SPAM)...")
    df["label"] = df.apply(assign_label, axis=1)

    # ── Statistiques ─────────────────────────────────────────────────────────
    print("\nDistribution des labels :")
    print(df["label"].value_counts())
    print(f"\nDistribution des notes :")
    print(df["rating"].value_counts().sort_index())

    return df


def save_raw_data(df: pd.DataFrame) -> None:
    """
    Sauvegarde les données brutes en CSV dans data/raw/.

    Args:
        df : DataFrame à sauvegarder
    """
    # Crée le dossier s'il n'existe pas
    os.makedirs(DATA_RAW_DIR, exist_ok=True)

    # Sauvegarde en CSV
    df.to_csv(RAW_CSV, index=False, encoding="utf-8")
    print(f"\nDonnees sauvegardees : {RAW_CSV}")
    print(f"Taille du fichier : {os.path.getsize(RAW_CSV) / 1024 / 1024:.1f} MB")


def main():
    """Point d'entrée principal du script."""
    # ── Arguments en ligne de commande ───────────────────────────────────────
    parser = argparse.ArgumentParser(description="Telecharge le dataset Amazon Reviews")
    parser.add_argument(
        "--max_samples", type=int, default=MAX_SAMPLES,
        help=f"Nombre max d'exemples (defaut: {MAX_SAMPLES})"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CHARGEMENT DU DATASET AMAZON SPORTS & OUTDOORS")
    print("=" * 60)

    # Télécharge et prépare les données
    df = load_amazon_reviews(max_samples=args.max_samples)

    # Sauvegarde
    save_raw_data(df)

    print("\nEtape suivante : python spark/preprocess.py")
    print("(ou python spark/preprocess.py --no-spark si Java non installe)")


if __name__ == "__main__":
    main()
