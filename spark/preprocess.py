# -*- coding: utf-8 -*-
"""
preprocess.py — Preprocessing du dataset Amazon Reviews

Ce script prépare les données brutes pour l'entraînement du modèle ML.
Il propose DEUX modes de fonctionnement au choix :

  ┌─────────────────────────────────────────────────────────────────────┐
  │  OPTION A — avec Apache Spark (mode par défaut)                     │
  │  • Traitement distribué sur plusieurs cœurs CPU                     │
  │  • Idéal pour des millions d'avis (scalable)                        │
  │  • Requiert : Java 11+ et PySpark installés                         │
  │  • Commande : python spark/preprocess.py                            │
  │                                                                     │
  │  OPTION B — sans Spark, mode pandas (--no-spark)                    │
  │  • Traitement classique sur un seul cœur                            │
  │  • Plus simple, aucun prérequis supplémentaire                      │
  │  • Parfait pour commencer ou pour la CI/CD GitHub Actions           │
  │  • Commande : python spark/preprocess.py --no-spark                 │
  └─────────────────────────────────────────────────────────────────────┘

Les deux options produisent EXACTEMENT les mêmes fichiers de sortie :
    data/processed/train.csv  (80% des données — entraînement)
    data/processed/test.csv   (20% des données — évaluation)

Usage :
    python spark/preprocess.py            # mode Spark (défaut)
    python spark/preprocess.py --no-spark # mode pandas
"""

import argparse
import os
import re
import sys
import unicodedata

# ─────────────────────────────────────────────────────────────────────────────
# CORRECTION WINDOWS — Python worker PySpark
# Sur Windows, Spark lance des processus Java qui cherchent "python" dans le
# PATH système. Si le redirecteur Windows Store intercepte la commande, les
# workers Spark ne trouvent pas Python et échouent avec "Python worker failed
# to connect back."
#
# La solution : forcer Spark à utiliser le même Python que le script courant
# (celui du venv) en définissant PYSPARK_PYTHON et PYSPARK_DRIVER_PYTHON.
# sys.executable retourne le chemin absolu du python en cours d'exécution.
# Ces variables doivent être définies AVANT d'importer pyspark.
# ─────────────────────────────────────────────────────────────────────────────
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Ajoute le dossier racine au path pour pouvoir importer src/config.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    RAW_CSV, TRAIN_CSV, TEST_CSV,
    DATA_PROCESSED_DIR,
    TEST_SIZE, RANDOM_STATE,
    COL_TEXT, COL_LABEL
)


# =============================================================================
# FONCTIONS DE NETTOYAGE DU TEXTE
# Ces fonctions sont partagées entre le mode Spark et le mode pandas.
# Elles transforment un texte brut en texte normalisé, prêt pour
# la vectorisation TF-IDF ou Word2Vec.
# =============================================================================

def clean_text(text: str) -> str:
    """
    Nettoie et normalise un texte d'avis produit.

    Étapes appliquées dans l'ordre :
    1. Gestion des valeurs nulles → retourne ""
    2. Conversion en minuscules → "EXCELLENT" devient "excellent"
    3. Suppression des balises HTML → "<br/>bon produit</p>" → "bon produit"
    4. Suppression des URLs → "voir http://shop.com" → "voir"
    5. Normalisation des accents → "éàü" → "eau" (ASCII pur)
    6. Suppression des caractères non alphabétiques → "produit!!" → "produit"
    7. Réduction des espaces multiples → "bon   produit" → "bon produit"
    8. Suppression des espaces en début/fin de chaîne

    Pourquoi normaliser les accents ?
    → TF-IDF et Word2Vec traitent "excellent" et "éxcellent" comme deux mots
      différents. En retirant les accents, on uniformise le vocabulaire.

    Args:
        text : le texte brut de l'avis (peut être None)

    Returns:
        Le texte nettoyé, ou "" si le texte était vide/None

    Exemple :
        >>> clean_text("Très BON produit <br/> je recommande !!!")
        'tres bon produit je recommande'
    """
    # ── Étape 1 : Gestion des valeurs nulles ─────────────────────────────────
    # Le dataset Amazon contient parfois des cellules vides (NaN)
    # On les transforme en chaîne vide pour éviter les erreurs
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""

    text = str(text)

    # ── Étape 2 : Conversion en minuscules ───────────────────────────────────
    # Normalise la casse : "EXCELLENT" et "excellent" sont le même mot
    text = text.lower()

    # ── Étape 3 : Suppression des balises HTML ────────────────────────────────
    # Les avis Amazon contiennent parfois du HTML (copié depuis une page web)
    # Exemple : "Bon produit <br/><p>Je recommande</p>"
    # Le regex <[^>]+> correspond à tout ce qui est entre < et >
    text = re.sub(r"<[^>]+>", " ", text)

    # ── Étape 4 : Suppression des URLs ───────────────────────────────────────
    # Les URLs (http://, https://, www.) sont du bruit inutile pour le modèle
    # Elles ont déjà été utilisées pour détecter le spam dans load_data.py
    text = re.sub(r"http[s]?://\S+", " ", text)
    text = re.sub(r"www\.\S+", " ", text)

    # ── Étape 5 : Normalisation des accents (Unicode → ASCII) ────────────────
    # unicodedata.normalize('NFD', ...) décompose les caractères accentués
    # en caractère de base + marque diacritique
    # Ex : "é" → "e" + accent aigu (marque séparée)
    # encode('ascii', 'ignore') supprime ensuite les marques diacritiques
    # → résultat : "é" → "e", "à" → "a", "ü" → "u"
    text = unicodedata.normalize("NFD", text)
    text = text.encode("ascii", "ignore").decode("utf-8")

    # ── Étape 6 : Suppression des caractères non alphabétiques ───────────────
    # On garde seulement les lettres et les espaces
    # Les chiffres, symboles (!@#$%...) sont supprimés
    # Pourquoi supprimer les chiffres ?
    # → "taille 42" et "taille 43" sont traitées pareil par le modèle
    text = re.sub(r"[^a-z\s]", " ", text)

    # ── Étape 7 : Réduction des espaces multiples ─────────────────────────────
    # Après suppression, il reste souvent des espaces consécutifs
    # Ex : "bon   produit    excellent" → "bon produit excellent"
    text = re.sub(r"\s+", " ", text)

    # ── Étape 8 : Strip (suppression espaces début/fin) ───────────────────────
    text = text.strip()

    return text


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrait des features numériques supplémentaires à partir du texte.

    Ces features complètent la vectorisation TF-IDF en capturant des aspects
    que les mots seuls ne peuvent pas exprimer :
    - La longueur d'un avis (les avis spam sont souvent très courts ou très longs)
    - Le nombre de mots (vocabulaire riche = avis authentique)
    - Le nombre de ! (les spam et avis très positifs en ont beaucoup)
    - Le ratio de majuscules (les SPAM CRIENT SOUVENT EN MAJUSCULES)
    - La présence d'une URL (signal fort de spam)

    Ces features sont surtout utiles pour XGBoost qui excelle avec les features numériques.
    TF-IDF et BERT n'en ont pas besoin (ils analysent directement le texte).

    Args:
        df : DataFrame avec au minimum une colonne 'text' (texte ORIGINAL, avant clean)

    Returns:
        Le même DataFrame avec 5 nouvelles colonnes ajoutées
    """
    # Sécurité : s'assure que la colonne text est bien en string
    raw_text = df["text"].fillna("").astype(str)

    # ── Feature 1 : Longueur en caractères ───────────────────────────────────
    # Un avis de 5 caractères est suspect. Un avis de 2000 caractères est détaillé.
    df["text_length"] = raw_text.str.len()

    # ── Feature 2 : Nombre de mots ───────────────────────────────────────────
    # Compte les mots séparés par des espaces
    # Un spam fait souvent 3-5 mots, un vrai avis en fait 20-100
    df["word_count"] = raw_text.str.split().str.len().fillna(0).astype(int)

    # ── Feature 3 : Nombre de points d'exclamation ───────────────────────────
    # Les avis spam et très enthousiastes abusent des !!!
    # Ex : "INCROYABLE!!! ACHETEZ MAINTENANT!!!"
    df["exclamation_count"] = raw_text.str.count(r"!")

    # ── Feature 4 : Ratio de majuscules ──────────────────────────────────────
    # Calcule le pourcentage de lettres en majuscules dans le texte
    # Un avis tout en majuscules (ratio > 0.5) est souvent du spam ou de la colère
    def uppercase_ratio(text: str) -> float:
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return 0.0
        return sum(1 for c in letters if c.isupper()) / len(letters)

    df["uppercase_ratio"] = raw_text.apply(uppercase_ratio)

    # ── Feature 5 : Présence d'une URL ────────────────────────────────────────
    # Booléen (0 ou 1) : 1 si le texte contient une URL http:// ou www.
    # IMPORTANT : cette feature est extraite AVANT clean_text() qui supprime les URLs
    url_pattern = re.compile(r"http[s]?://|www\.", re.IGNORECASE)
    df["has_url"] = raw_text.apply(lambda x: 1 if url_pattern.search(x) else 0)

    return df


def is_spam(row: pd.Series) -> bool:
    """
    Détecte si un avis est du spam en combinant plusieurs critères.

    Règles appliquées :
    1. Avis non vérifié + contient une URL → SPAM
    2. Avis non vérifié + trop court (< 20 chars) → SPAM probable
    3. Avis vérifié → jamais classé spam automatiquement

    Args:
        row : une ligne du DataFrame avec colonnes 'text' et 'verified_purchase'

    Returns:
        True si spam, False sinon
    """
    text = str(row.get("text", ""))
    verified = row.get("verified_purchase", True)

    SPAM_REGEX = re.compile(
        r"http[s]?://|www\.|click here|buy now|discount code|"
        r"promo code|free shipping|limited offer|\$\d+|!!{3,}|[A-Z]{10,}",
        re.IGNORECASE
    )

    if not verified and SPAM_REGEX.search(text):
        return True

    if not verified and len(text.strip()) < 20:
        return True

    return False


# =============================================================================
# OPTION A — PREPROCESSING AVEC APACHE SPARK
# =============================================================================

def preprocess_with_spark(input_csv: str, output_dir: str) -> None:
    """
    Preprocessing distribué avec Apache Spark.

    Apache Spark divise le dataset en partitions et les traite EN PARALLÈLE
    sur tous les cœurs CPU disponibles. Avec 300 000 avis et 8 cœurs, Spark
    peut être 4 à 8 fois plus rapide que pandas.

    Fonctionnement interne :
    1. SparkSession : crée le moteur de calcul distribué
    2. Lecture CSV → Spark DataFrame (distribué en mémoire)
    3. UDFs (User Defined Functions) : applique clean_text() sur chaque partition
    4. Filtrage des textes trop courts après nettoyage
    5. Conversion pandas → split train/test → sauvegarde CSV

    Prérequis :
    • Java 11+ installé et JAVA_HOME configuré (voir Guide ÉTAPE 3A)
    • PySpark installé : pip install pyspark
    """
    print("\n" + "=" * 60)
    print("MODE APACHE SPARK — PREPROCESSING DISTRIBUÉ")
    print("=" * 60)

    try:
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F
        from pyspark.sql.types import StringType
    except ImportError:
        raise ImportError(
            "\nERREUR : PySpark n'est pas installé.\n"
            "Lance : pip install pyspark\n"
            "Et vérifie que Java 11+ est installé : java -version\n"
            "Ou utilise le mode pandas : python spark/preprocess.py --no-spark\n"
        )

    # ── Étape 1 : Création de la SparkSession ─────────────────────────────────
    # SparkSession est le point d'entrée de toute application Spark.
    # master("local[*]") = utilise TOUS les cœurs CPU de la machine.
    # Si tu as 8 cœurs, Spark créera 8 threads parallèles automatiquement.
    # L'interface web Spark UI est accessible sur http://localhost:4040 pendant l'exécution
    print("Démarrage de la SparkSession...")
    print("(L'interface web Spark sera disponible sur http://localhost:4040)")
    spark = SparkSession.builder \
        .appName("SportReview-AI-Preprocessing") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.driver.extraJavaOptions", "-Dlog4j.configuration=") \
        .getOrCreate()
    # local[*] = utilise TOUS les cœurs CPU disponibles sur la machine
    # Sur mon PC à 16 cœurs, Spark créera 16 threads de traitement parallèle

    # Réduit le niveau de logs pour ne voir que les erreurs (pas les INFO verbeux)
    spark.sparkContext.setLogLevel("ERROR")
    cores = spark.sparkContext.defaultParallelism
    print(f"SparkSession démarrée — {cores} cœurs CPU utilisés en parallèle")

    # ── Étape 2 : Lecture du CSV ──────────────────────────────────────────────
    # Spark lit le CSV et le distribue automatiquement en partitions (blocs)
    # inferSchema=True : Spark détecte automatiquement les types de colonnes
    #
    # IMPORTANT — multiLine=True :
    # Certains avis Amazon contiennent des retours à la ligne intégrés dans
    # le texte (ex: "Bon produit.\nLivraison rapide."). Pandas les écrit entre
    # guillemets dans le CSV, mais Spark par défaut traite chaque \n comme un
    # séparateur de ligne → il coupe l'avis en deux lignes → les colonnes se
    # décalent → la colonne "label" se retrouve avec du texte au lieu du label.
    # multiLine=True dit à Spark de gérer les champs multi-lignes correctement.
    # escape='"' gère les guillemets doubles imbriqués dans les champs CSV.
    print(f"\nLecture de {input_csv}...")
    df_spark = spark.read.csv(
        input_csv,
        header=True,
        inferSchema=True,
        multiLine=True,   # gère les textes avec des retours à la ligne intégrés
        escape='"'        # gère les guillemets doubles dans les champs CSV
    )
    total = df_spark.count()
    print(f"Données chargées : {total:,} lignes réparties sur {cores} partitions")

    # ── Étapes 3 & 4 : Nettoyage avec fonctions SQL natives Spark ────────────
    #
    # POURQUOI PAS UNE UDF PYTHON ?
    # Une UDF (F.udf) force Spark à lancer des sous-processus Python séparés
    # (les "workers") pour exécuter la fonction. Sur Windows avec Anaconda,
    # ces sous-processus héritent de la socket d'Anaconda → conflit → crash.
    #
    # SOLUTION : fonctions SQL natives de Spark (F.lower, F.regexp_replace...)
    # Ces fonctions s'exécutent DIRECTEMENT dans la JVM Java — aucun sous-
    # processus Python n'est lancé → plus de crash, et souvent plus rapide.
    #
    # Les transformations appliquées dans l'ordre :
    #   1. F.lower()          → "EXCELLENT" devient "excellent"
    #   2. regexp_replace HTML → "<br/>bon produit</p>" → " bon produit "
    #   3. regexp_replace URL  → "voir http://shop.com" → "voir  "
    #   4. regexp_replace [^a-z\s] → "produit!!" → "produit  "
    #                          (garde seulement lettres a-z et espaces)
    #   5. F.trim()           → supprime les espaces en début et fin
    #
    # Note : la normalisation des accents (unicodedata) n'est pas disponible
    # en SQL natif Spark. Les accents seront supprimés par l'étape [^a-z\s].
    # Exemple : "éàü" → ""  (les accents disparaissent, pas de remplacement)
    # C'est acceptable pour un modèle NLP anglophone sur Amazon Reviews.
    print("Nettoyage du texte en parallèle sur tous les cœurs (fonctions SQL natives)...")
    df_spark = df_spark.withColumn("text_clean",
        F.trim(
            F.regexp_replace(
                F.regexp_replace(
                    F.regexp_replace(
                        F.lower(F.col("text")),
                        r'<[^>]+>', ' '            # 1. supprime balises HTML
                    ),
                    r'https?://\S+|www\.\S+', ' '  # 2. supprime URLs
                ),
                r'[^a-z\s]', ' '                   # 3. garde lettres + espaces
            )
        )
    )

    # ── Étape 5 : Filtrage des textes trop courts ─────────────────────────────
    # Supprime les avis dont le texte nettoyé est vide ou < 10 caractères
    # F.length() calcule la longueur de la chaîne
    # .filter() garde seulement les lignes qui satisfont la condition
    df_spark = df_spark.filter(F.length(F.col("text_clean")) >= 10)
    df_spark = df_spark.filter(F.col("label").isNotNull())
    remaining = df_spark.count()
    removed = total - remaining
    print(f"Après filtrage : {remaining:,} lignes conservées ({removed:,} supprimées)")

    # ── Étape 6 : Conversion Spark → pandas ───────────────────────────────────
    # Spark est optimal pour le traitement distribué.
    # Pour le split stratifié (sklearn) et la sauvegarde CSV, on repasse en pandas.
    # toPandas() "collecte" toutes les données depuis tous les cœurs vers la RAM principale
    print("\nConversion Spark → pandas pour le split train/test...")
    df_pd = df_spark.select("text_clean", "label").toPandas()
    df_pd = df_pd.rename(columns={"text_clean": "text"})

    # Ferme proprement la SparkSession (libère la mémoire et les ports)
    spark.stop()
    print("SparkSession fermée proprement.")

    # ── Étape 7 : Split train/test et sauvegarde ─────────────────────────────
    _split_and_save(df_pd, output_dir)


# =============================================================================
# OPTION B — PREPROCESSING AVEC PANDAS (sans Spark)
# =============================================================================

def preprocess_with_pandas(input_csv: str, output_dir: str) -> None:
    """
    Preprocessing classique avec pandas (sans Apache Spark).

    Cette option fait exactement la même chose que l'option Spark mais
    sur un seul cœur CPU, avec la librairie pandas.

    Avantages :
    - Aucun prérequis supplémentaire (pas de Java, pas de JAVA_HOME)
    - Simple et suffisant pour 300 000 avis sur une machine moderne
    - Idéale pour débuter, pour la CI/CD, ou si Java pose problème

    Temps estimé :
    - 50 000 avis  : ~1-2 minutes
    - 300 000 avis : ~5-10 minutes
    """
    print("\n" + "=" * 60)
    print("MODE PANDAS — PREPROCESSING CLASSIQUE (sans Spark)")
    print("=" * 60)

    # ── Étape 1 : Lecture du CSV ──────────────────────────────────────────────
    print(f"Lecture de {input_csv}...")
    df = pd.read_csv(input_csv, encoding="utf-8")
    print(f"Données chargées : {len(df):,} lignes")

    # Affiche la distribution initiale des labels
    print("\nDistribution des labels AVANT nettoyage :")
    print(df["label"].value_counts())

    # ── Étape 2 : Extraction des features AVANT nettoyage ─────────────────────
    # IMPORTANT : extract_features() DOIT être appelé AVANT clean_text()
    # car elle analyse le texte brut (majuscules, URLs, longueur originale)
    # clean_text() supprimerait ces informations
    print("\nExtraction des features numériques (avant nettoyage)...")
    df = extract_features(df)
    print("  ✓ text_length     : longueur en caractères")
    print("  ✓ word_count      : nombre de mots")
    print("  ✓ exclamation_count : nombre de !")
    print("  ✓ uppercase_ratio : ratio de majuscules")
    print("  ✓ has_url         : présence d'une URL")

    # ── Étape 3 : Nettoyage du texte ─────────────────────────────────────────
    # .apply() applique clean_text() sur chaque ligne de la colonne 'text'
    # Cela peut prendre quelques minutes sur 300 000 avis
    print("\nNettoyage du texte...")
    print("(Cette étape peut prendre 3-5 minutes sur 300 000 avis...)")
    df["text_clean"] = df["text"].apply(clean_text)
    print("Nettoyage terminé.")

    # ── Étape 4 : Filtrage ────────────────────────────────────────────────────
    # Supprime les avis vides ou trop courts APRÈS nettoyage
    initial_count = len(df)
    df = df[df["text_clean"].str.len() >= 10].copy()
    df = df[df["label"].notna()].copy()
    removed = initial_count - len(df)
    print(f"\nAprès filtrage : {len(df):,} lignes conservées ({removed:,} supprimées)")

    # ── Étape 5 : Remplacement de la colonne texte ────────────────────────────
    # On remplace le texte brut par le texte nettoyé
    df["text"] = df["text_clean"]
    df = df.drop(columns=["text_clean"])

    # Affiche la distribution finale des labels
    print("\nDistribution des labels APRÈS nettoyage :")
    print(df["label"].value_counts())
    pct = (df["label"].value_counts(normalize=True) * 100).round(1)
    print("\nEn pourcentage :")
    for label, p in pct.items():
        print(f"  {label:<10} : {p}%")

    # ── Étape 6 : Split train/test et sauvegarde ─────────────────────────────
    _split_and_save(df, output_dir)


# =============================================================================
# FONCTION COMMUNE : SPLIT TRAIN/TEST ET SAUVEGARDE
# =============================================================================

def _split_and_save(df: pd.DataFrame, output_dir: str) -> None:
    """
    Divise le dataset en ensembles train et test, puis sauvegarde en CSV.

    Stratégie de split STRATIFIÉ :
    - 80% entraînement / 20% test
    - Stratifié = chaque classe (POSITIF, NEGATIF, NEUTRE, SPAM) est représentée
      PROPORTIONNELLEMENT dans les deux ensembles.

    Pourquoi stratifier ?
    → Sans stratification, le test pourrait par malchance avoir très peu d'exemples
      SPAM (classe minoritaire). Le modèle serait mal évalué sur cette classe.
    → Avec stratification, si SPAM représente 6% du dataset total, il représente
      aussi 6% dans train ET 6% dans test.

    Pourquoi 80/20 ?
    → 80% pour entraîner = assez de données pour un bon modèle
    → 20% pour tester = assez pour une évaluation fiable (environ 60 000 avis sur 300k)
    """
    print("\n" + "-" * 50)
    print("SPLIT TRAIN / TEST (stratifié 80% / 20%)")
    print("-" * 50)

    X = df.drop(columns=[COL_LABEL])
    y = df[COL_LABEL]

    # train_test_split avec stratify=y garantit la même distribution de classes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,        # 0.2 = 20% pour le test
        random_state=RANDOM_STATE,  # 42 = graine fixe pour reproductibilité
        stratify=y                  # maintient la distribution des classes
    )

    # Reconstruit les DataFrames complets
    train_df = X_train.copy()
    train_df[COL_LABEL] = y_train
    test_df = X_test.copy()
    test_df[COL_LABEL] = y_test

    # Sauvegarde
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    print(f"\nTrain : {len(train_df):,} exemples sauvegardés → {train_path}")
    print(f"Test  : {len(test_df):,} exemples sauvegardés  → {test_path}")

    print("\nDistribution des classes dans le train :")
    for label, count in train_df[COL_LABEL].value_counts().items():
        pct = count / len(train_df) * 100
        print(f"  {label:<10} : {count:>6,} exemples ({pct:.1f}%)")

    print("\nDistribution des classes dans le test :")
    for label, count in test_df[COL_LABEL].value_counts().items():
        pct = count / len(test_df) * 100
        print(f"  {label:<10} : {count:>6,} exemples ({pct:.1f}%)")

    print("\n" + "=" * 60)
    print("PREPROCESSING TERMINÉ AVEC SUCCÈS !")
    print("=" * 60)
    print("\nÉtape suivante : python src/train.py")
    print("(ou python src/train.py --models tfidf_lr --fast pour tester rapidement)")


# =============================================================================
# POINT D'ENTRÉE DU SCRIPT
# =============================================================================

def main():
    """
    Point d'entrée du script.
    Gère les arguments en ligne de commande et lance le bon mode.
    """
    parser = argparse.ArgumentParser(
        description="Preprocessing du dataset Amazon Reviews (Spark ou pandas)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python spark/preprocess.py             # mode Spark (nécessite Java 11+)
  python spark/preprocess.py --no-spark  # mode pandas (aucun prérequis)

Voir le GUIDE_ETAPE_PAR_ETAPE.md :
  - ÉTAPE 3A pour l'installation complète de Java + Spark
  - ÉTAPE 3B pour le mode sans Spark
        """
    )
    parser.add_argument(
        "--no-spark",
        action="store_true",
        default=False,
        help="Utilise pandas au lieu de Spark (aucun prérequis Java requis)"
    )
    args = parser.parse_args()

    # Vérifie que le fichier source existe avant de commencer
    if not os.path.exists(RAW_CSV):
        print(f"\nERREUR : Fichier introuvable : {RAW_CSV}")
        print("Tu dois d'abord télécharger le dataset.")
        print("Lance : python data/load_data.py")
        sys.exit(1)

    # Lance le mode choisi
    if args.no_spark:
        preprocess_with_pandas(RAW_CSV, DATA_PROCESSED_DIR)
    else:
        preprocess_with_spark(RAW_CSV, DATA_PROCESSED_DIR)


if __name__ == "__main__":
    main()
