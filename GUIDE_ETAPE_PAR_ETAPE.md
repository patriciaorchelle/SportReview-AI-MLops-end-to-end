# Guide Complet — SportReview AI (Projet MLOps Professionnel)

## Ce projet en une phrase

Un **système d'analyse d'avis produits sportifs** capable de classer automatiquement
des avis Amazon en 4 catégories : **POSITIF / NEGATIF / NEUTRE / SPAM**.

Le pipeline complet couvre tout le cycle de vie d'un modèle ML en production :
collecte de données → preprocessing → entraînement → API → monitoring → CI/CD → Kubernetes.

**Dataset** : Amazon Sports & Outdoors Reviews (19,5 millions d'avis authentiques — 300 000 utilisés pour l'entraînement)
**Stack** : Python · Spark · scikit-learn · Transformers (DistilBERT) · MLflow · FastAPI · Gradio · Docker · Kubernetes · GitHub Actions · Prometheus · Grafana · EvidentlyAI

---

## Ce qu'on va construire étape par étape

| Étape | Ce que tu fais | Temps estimé |
|-------|---------------|--------------|
| 1 | Créer l'environnement Python | 15 min |
| 2 | Télécharger le dataset Amazon (19,5M avis) + créer échantillon 300 000 | 15-30 min |
| 3A | Preprocessing avec Apache Spark | 20-30 min (installation) + 10 min |
| 3B | Preprocessing sans Spark (pandas) | 5-10 min |
| 4 | Entraîner 6 modèles avec MLflow | 30-90 min |
| 5 | Promouvoir le meilleur modèle | 5 min |
| 6 | Lancer l'API FastAPI | 5 min |
| 7 | Lancer l'interface Gradio | 5 min |
| 8 | Lancer les tests automatisés | 5 min |
| 9 | Détection de drift EvidentlyAI | 5 min |
| 10 | Docker + docker-compose | 15 min |
| 11 | GitHub Actions CI/CD | 10 min |
| 12 | Kubernetes avec minikube | 30 min (installation) + 10 min |

---

## Prérequis de base (obligatoires pour tout le monde)

- **Python 3.11** — recommandé pour la compatibilité avec toutes les librairies ML
  → Télécharge depuis https://www.python.org/downloads/
  → Pendant l'installation, **coche "Add Python to PATH"**
- **Git** — pour versionner le code et déclencher GitHub Actions
  → Télécharge depuis https://git-scm.com/downloads
- **Docker Desktop** — pour containeriser l'application
  → Télécharge depuis https://www.docker.com/products/docker-desktop/
  → Lance Docker Desktop avant d'utiliser les commandes docker
- **Compte GitHub** — pour héberger le code et activer la CI/CD

---

## ETAPE 1 — Création de l'environnement Python (15 min)

### Pourquoi un environnement virtuel ?

Un environnement virtuel crée une installation Python **isolée** pour ce projet.
Cela évite les conflits entre les versions des librairies d'un projet à l'autre.
Par exemple, ce projet utilise MLflow 2.15+, mais un autre projet pourrait utiliser
une version plus ancienne. Sans environnement virtuel, les deux installations se
mélangeraient et créeraient des erreurs.

### Commandes à exécuter

```bash
# 1. Ouvre un terminal dans le dossier du projet
# Sur Windows : clic droit dans le dossier -> "Ouvrir dans le terminal"

# 2. Crée l'environnement virtuel dans un sous-dossier "venv"
python -m venv venv

# 3. Active l'environnement virtuel
# Sur Windows (PowerShell ou cmd) :
venv\Scripts\activate

# Sur Linux / Mac :
# source venv/bin/activate

# 4. Vérifie que l'environnement est bien activé
# Tu dois voir "(venv)" au début de la ligne de commande, comme ceci :
# (venv) C:\Users\ton_nom\sportreview_ai>

# 5. Installe toutes les dépendances du projet
pip install -r requirements.txt
# Cette commande peut prendre 5-10 minutes car elle télécharge et installe :
# scikit-learn, MLflow, FastAPI, Gradio, Transformers, XGBoost, etc.
```

### Comment vérifier que l'installation a réussi

Exécute cette commande de vérification :

```bash
python -c "import sklearn, mlflow, fastapi, gradio, xgboost; print('Toutes les librairies principales sont installées — OK')"
```

**Ce que tu dois voir :**
```
Toutes les librairies principales sont installées — OK
```

Si on voit une erreur ImportError, notons le nom de la librairie manquante et installons-la :
```bash
pip install nom_de_la_librairie
```

---

## ETAPE 2 — Téléchargement du dataset Amazon (15-30 min)

### Qu'est-ce qu'on télécharge ?

Le dataset **Amazon Reviews 2023** (McAuley-Lab) contient **19,5 millions d'avis**
authentiques sur des produits Amazon. On utilise la catégorie **Sports & Outdoors**
(chaussures, raquettes, vélos, équipements de randonnée, etc.).

Ce dataset est hébergé sur **HuggingFace**. La librairie `datasets` le télécharge
et le met en cache automatiquement (~9 GB sur disque). Les téléchargements suivants
seront instantanés car tout reste en cache local.

> **Pourquoi 300 000 pour l'entraînement ?**
> Le dataset complet fait 19,5 millions d'avis et ~4 GB de CSV — trop lourd pour
> entraîner des modèles ML sur un PC standard (RAM insuffisante pour TF-IDF sur 19M lignes,
> temps d'entraînement de plusieurs heures). Dans l'industrie, on travaille toujours
> avec un échantillon représentatif en développement, puis on passe au full dataset
> sur un cluster cloud (AWS SageMaker, GCP Vertex AI) en production.
> 300 000 avis = échantillon stratifié par note qui préserve la distribution originale.

### Les règles de labellisation appliquées

| Note | Label | Logique |
|------|-------|---------|
| 4 ou 5 étoiles | POSITIF | L'utilisateur est satisfait |
| 3 étoiles | NEUTRE | L'utilisateur est mitigé |
| 1 ou 2 étoiles | NEGATIF | L'utilisateur est insatisfait |
| Peu importe | SPAM | Avis non vérifié + URL ou pattern promotionnel |

La règle SPAM a la **priorité absolue** : un avis 5 étoiles mais détecté comme spam
sera classé SPAM, pas POSITIF.

### Commande à exécuter

```bash
# Option A — Télécharge TOUT (19,5M avis) puis crée l'échantillon 300 000 (recommandé)
# Le téléchargement complet fait ~9 GB et peut prendre 15-30 min selon la connexion
python data/load_data.py --max_samples 300000

# Option B — Pour tester rapidement avec très peu de données (< 1 min)
python data/load_data.py --max_samples 10000

# Note : si tu as déjà téléchargé le dataset complet une fois,
# toutes les relances suivantes sont INSTANTANÉES (cache HuggingFace local)
```

**Ce qu'on voit pendant l'exécution :**
```
============================================================
CHARGEMENT DU DATASET AMAZON SPORTS & OUTDOORS
============================================================
Telechargement du dataset Amazon Sports & Outdoors...
(Le premier telechargement peut prendre quelques minutes)
Librairie datasets v2.21.0 — OK
Téléchargement de McAuley-Lab/Amazon-Reviews-2023 / raw_review_Sports_and_Outdoors...
Downloading data: 100%|████████████| 9.26G/9.26G [16:59<00:00, 9.08MB/s]
Generating full split: 19595170 examples [15:43, 20760.18 examples/s]
Dataset téléchargé : 19,595,170 avis au total

Dataset reduit a 300000 exemples (echantillonnage stratifie)
Assignation des labels (POSITIF/NEGATIF/NEUTRE/SPAM)...

Distribution des labels :
label
NEGATIF    113843
POSITIF    113929
NEUTRE      56982
SPAM        15246

Donnees sauvegardees : data/raw/reviews_raw.csv
Taille du fichier : 73.2 MB
```

> **Important — taille du fichier :**
> - Avec `--max_samples 300000` → fichier CSV de ~60 MB → parfait pour l'entraînement
> - Sans argument (tous les avis) → fichier CSV de ~4 GB → trop lourd pour ML sur PC
> - Le cache HuggingFace (~9 GB) reste sur ton disque à `C:\Users\User\.cache\huggingface\`
>   → ne jamais supprimer ce dossier si tu veux éviter de re-télécharger


## Les notes (ratings) → les labels se font en 2 temps :

## **Étape 1 — Échantillonnage stratifié par note :**
60 000 avis par note → 300 000 au total, distribution parfaitement égale.

## **Étape 2 — Assignation des labels (SPAM en priorité absolue) :**

La règle SPAM s'applique sur tous les avis, peu importe leur note. Un avis 5 étoiles non vérifié avec un lien URL devient SPAM, pas POSITIF.

Donc :
```
Rating 1 (60 000) → la plupart deviennent NEGATIF, mais certains → SPAM
Rating 2 (60 000) → la plupart deviennent NEGATIF, mais certains → SPAM
Rating 3 (60 000) → la plupart deviennent NEUTRE,  mais certains → SPAM
Rating 4 (60 000) → la plupart deviennent POSITIF,  mais certains → SPAM
Rating 5 (60 000) → la plupart deviennent POSITIF,  mais certains → SPAM
```

Ce qui donne :
```
POSITIF = (60k rating4 + 60k rating5) - SPAM détectés = 120 000 - 6 071 = 113 929
NEGATIF = (60k rating1 + 60k rating2) - SPAM détectés = 120 000 - 6 157 = 113 843
NEUTRE  =  60k rating3               - SPAM détectés =  60 000 - 3 018 =  56 982
SPAM    = 6071 + 6157 + 3018                                            =  15 246 ✓
```

**Vérification :** 113 929 + 113 843 + 56 982 + 15 246 = **300 000** ✓

C'est exactement ce qu'on veut — les 15 246 SPAM viennent de toutes les notes, pas seulement des mauvaises notes.

---


## ETAPE 3A — Preprocessing AVEC Apache Spark

> Choisir cette option pour apprendre Spark et voir le traitement distribué en action.
> Apache Spark traite les données EN PARALLÈLE sur tous tes cœurs CPU.

### Qu'est-ce qu'Apache Spark ?

Apache Spark est un moteur de traitement distribué. Il divise automatiquement
un gros dataset en **partitions** et les traite en parallèle sur tous les cœurs CPU.
Dans l'industrie, Spark traite des téraoctets de données (logs serveurs, transactions...).
Pour nos 300 000 avis (extraits de 19,5 millions), c'est excellent pour démontrer la scalabilité — et la même architecture traite directement les 19,5M sans modifier une ligne de code.

---

### Installation de Java 11 (requis par Spark)

Spark est écrit en Scala qui tourne sur la JVM. **Java 11 doit être installé avant PySpark.**

#### Étape A — Télécharger Java 11 JDK

1. Va sur : https://learn.microsoft.com/fr-fr/java/openjdk/download
2. Dans la section **OpenJDK 11**, télécharge le fichier `.msi` pour Windows x64
3. Double-clique sur le `.msi` téléchargé et suis l'installateur avec les options par défaut

#### Étape B — Configurer JAVA_HOME

JAVA_HOME est la variable système que Spark utilise pour trouver Java.

**Via l'interface Windows :**

1. Presse `Windows + R`, tape `sysdm.cpl`, appuie sur Entrée
2. Clique sur l'onglet **"Paramètres système avancés"**
3. Clique sur **"Variables d'environnement..."**
4. Dans **"Variables système"**, clique sur **"Nouvelle..."**
5. Remplis :
   - Nom : `JAVA_HOME`
   - Valeur : `C:\Program Files\Microsoft\jdk-11.0.x.x-hotspot` (adapte le numéro exact)
6. Dans la liste, trouve `Path`, double-clique, clique **"Nouveau"**, ajoute : `%JAVA_HOME%\bin`
7. Clique OK sur toutes les fenêtres

#### Étape C — Fermer et rouvrir le terminal

Les variables d'environnement ne se rechargent pas dans un terminal déjà ouvert.
Ferme entièrement le terminal, rouvre-en un nouveau, et réactive l'environnement :

```bash
venv\Scripts\activate
```

#### Étape D — Vérifier Java

```bash
java -version
```

**Ce que tu dois voir :**
```
openjdk version "11.0.x" 2024-xx-xx LTS
OpenJDK Runtime Environment Microsoft-xxxxxx (build 11.0.x+xx-LTS)
OpenJDK 64-Bit Server VM Microsoft-xxxxxx (build 11.0.x+xx-LTS, mixed mode)
```

Si tu vois `'java' n'est pas reconnu`, JAVA_HOME n'est pas correctement configuré.
Reviens à l'Étape B et vérifie le chemin exact de ton installation.

#### Étape E — Installer PySpark

```bash
pip install pyspark
```

**Ce que tu dois voir :**
```
Successfully installed pyspark-3.5.x ...
```

#### Étape F — Vérifier PySpark

```bash
python -c "from pyspark.sql import SparkSession; print('PySpark installé et Java trouvé — OK')"
```

**Ce que qu'on dois voir :**
```
PySpark installé et Java trouvé — OK
```

Si tu vois `Java gateway process exited` → JAVA_HOME n'est pas bien configuré.

---

### Lancer le preprocessing avec Spark

```bash
python spark/preprocess.py
```

**Ce que tu vois :**
```
============================================================
MODE APACHE SPARK — PREPROCESSING DISTRIBUÉ
============================================================
Démarrage de la SparkSession...
(L'interface web Spark sera disponible sur http://localhost:4040)
SparkSession démarrée — 16 coeurs CPU utilisés en parallèle

Lecture de C:\Users\User\Documents\MLops\sportreview_ai\data\raw\reviews_raw.csv...
Données chargées : 300,000 lignes réparties sur 16 partitions

Nettoyage du texte en parallèle sur tous les coeurs (fonctions SQL natives)...
Après filtrage : 299,401 lignes conservées (599 supprimées)

Conversion Spark -> pandas pour le split train/test...
SparkSession fermée proprement.

--------------------------------------------------
SPLIT TRAIN / TEST (stratifié 80% / 20%)
--------------------------------------------------
Train : 239,520 exemples sauvegardés → data/processed/train.csv
Test  : 59,881 exemples sauvegardés  → data/processed/test.csv

Distribution des classes dans le train :
  NEGATIF    :  90,989 exemples (38.0%)
  POSITIF    :  90,817 exemples (37.9%)
  NEUTRE     :  45,537 exemples (19.0%)
  SPAM       :  12,177 exemples  (5.1%)

Distribution des classes dans le test :
  NEGATIF    :  22,748 exemples (38.0%)
  POSITIF    :  22,705 exemples (37.9%)
  NEUTRE     :  11,384 exemples (19.0%)
  SPAM       :   3,044 exemples  (5.1%)

============================================================
PREPROCESSING TERMINÉ AVEC SUCCÈS !
============================================================
Étape suivante : python src/train.py
```

On peut aussi ouvrir **http://localhost:4040** pendant l'exécution pour voir
l'interface Spark et observer les jobs distribués en temps réel.

---

## ETAPE 3B — Preprocessing SANS Spark (mode pandas)

> Choisir cette option pour aller plus vite sans installer Java.
> Aucun prérequis supplémentaire — pandas est déjà installé.

```bash
python spark/preprocess.py --no-spark
```

**Ce qu'on vois :**
```
============================================================
MODE PANDAS — PREPROCESSING CLASSIQUE (sans Spark)
============================================================
Lecture de data/raw/reviews_raw.csv...
Données chargées : 300,000 lignes

Distribution des labels AVANT nettoyage :
NEGATIF    113,843
POSITIF    113,929
NEUTRE      56,982
SPAM        15,246

Nettoyage du texte...
(Cette étape peut prendre 3-5 minutes sur les 300 000 avis échantillonnés...)
Nettoyage terminé.

Après filtrage : 299,401 lignes conservées (599 supprimées)

--------------------------------------------------
SPLIT TRAIN / TEST (stratifié 80% / 20%)
--------------------------------------------------
Train : 239,520 exemples → data/processed/train.csv
Test  : 59,881 exemples  → data/processed/test.csv

Distribution des classes dans le train :
  NEGATIF    :  90,989 exemples (38.0%)
  POSITIF    :  90,817 exemples (37.9%)
  NEUTRE     :  45,537 exemples (19.0%)
  SPAM       :  12,177 exemples  (5.1%)

============================================================
PREPROCESSING TERMINÉ AVEC SUCCÈS !
============================================================
```

### Vérifier les fichiers créés (valable après 3A ou 3B)

```bash
# Vérifie que les fichiers existent
dir data\processed\

# Vérifie le contenu
python -c "
import pandas as pd
df = pd.read_csv('data/processed/train.csv')
print('Colonnes :', list(df.columns))
print('Taille :', len(df), 'exemples')
print()
print('5 premiers avis :')
print(df[['text', 'label']].head())
"
```

**Ce qu'on dois voir :**
```
Colonnes : ['text', 'label', 'text_length', 'word_count', 'exclamation_count', 'uppercase_ratio', 'has_url']
Taille : 239520 exemples

5 premiers avis :
                                                text   label
0   great shoes very comfortable for trail running  POSITIF
1         terrible quality fell apart after one use  NEGATIF
2                   decent product nothing special   NEUTRE
3  excellent bike helmet fits perfectly recommend   POSITIF
4                      buy now click www fake com    SPAM
```

---

## ETAPE 4 — Entraînement des 6 modèles avec MLflow (30-90 min)

### Comprendre les 6 modèles

**Modèle 1 — TF-IDF + Logistic Regression (tfidf_lr)**
TF-IDF représente chaque avis par un vecteur de 10 000 nombres.
La Logistic Regression trouve la frontière de décision linéaire.
C'est le baseline : le plus simple, le plus rapide, le point de comparaison.

**Modèle 2 — TF-IDF + SVM (tfidf_svm)**
Même représentation TF-IDF mais le SVM cherche l'hyperplan qui maximise la marge.
Généralement légèrement meilleur que LR sur du texte.

**Modèle 3 — TF-IDF + XGBoost (tfidf_xgb)**
XGBoost = ensemble de centaines d'arbres de décision en séquence.
Exploite aussi les features numériques (text_length, word_count...).

**Modèle 4 — Word2Vec + Logistic Regression (w2v_lr)**
Word2Vec apprend des représentations vectorielles basées sur le contexte :
"chaussures" et "baskets" ont des vecteurs proches car ils apparaissent dans
les mêmes contextes. Chaque avis = moyenne des vecteurs de ses mots.

**Modèle 5 — Sentence-BERT + Logistic Regression (sbert_lr)**
SBERT encode des phrases entières (pas juste des mots) en vecteurs de 384 dimensions.
Comprend le contexte global : "bon" vs "pas bon" → vecteurs très différents.

**Modèle 6 — DistilBERT fine-tuné (distilbert)**
DistilBERT est un Transformer pré-entraîné sur des milliards de textes.
Fine-tuning = on réentraîne ce modèle géant sur nos avis Amazon.
Technique état de l'art en NLP. F1 obtenu (réel mai 2026) : 0.7103 — meilleur modèle du projet.

### Comprendre la gestion du déséquilibre

| Modèle | Technique | Mécanisme |
|--------|-----------|-----------|
| tfidf_lr, tfidf_svm | class_weight="balanced" | Pénalise plus les erreurs sur classes rares |
| tfidf_xgb | sample_weight | Poids par classe calculés et passés au fit() |
| w2v_lr | SMOTE | Crée de nouveaux exemples synthétiques pour SPAM et NEUTRE |
| sbert_lr | ADASYN | Comme SMOTE mais génère là où c'est difficile |
| distilbert | Weighted loss | Poids dans la fonction de perte PyTorch |

### Options d'entraînement

```bash
# Entraîner TOUS les 6 modèles (recommandé pour le projet final)
python src/train.py

# Entraîner seulement le baseline (5 min pour tester)
python src/train.py --models tfidf_lr

# Entraîner les 3 modèles TF-IDF (sans GPU, sans SBERT/DistilBERT)
python src/train.py --models tfidf_lr tfidf_svm tfidf_xgb

# Mode rapide pour CI/CD (dataset réduit à 5000 exemples)
python src/train.py --fast

# Entraîner 5 modèles classiques (sans DistilBERT)
python src/train.py --models tfidf_lr tfidf_svm tfidf_xgb w2v_lr sbert_lr
```

**Résultats réels obtenus (mai 2026) — 3 modèles TF-IDF :**

**Modèle 1 — tfidf_lr (baseline)**
```
CV F1 : 0.6517 ± 0.0011

RÉSULTATS TFIDF_LR :
  Accuracy   : 0.6268 (62.68%)
  F1 weighted: 0.6540
  F1 macro   : 0.5391

  F1 par classe :
    NEGATIF    : 0.7058  ██████████████
    NEUTRE     : 0.4319  ████████
    POSITIF    : 0.7675  ███████████████
    SPAM       : 0.2513  █████
```

**Modèle 2 — tfidf_svm**
```
CV F1 : 0.6513 ± 0.0002

RÉSULTATS TFIDF_SVM :
  Accuracy   : 0.7011 (70.11%)
  F1 weighted: 0.6599
  F1 macro   : 0.4887

  F1 par classe :
    NEGATIF    : 0.7725  ███████████████
    NEUTRE     : 0.2973  █████
    POSITIF    : 0.8070  ████████████████
    SPAM       : 0.0779  █
```

**Modèle 3 — tfidf_xgb**

*Version 1 — SANS sample_weight (déséquilibre non géré) :*
```
CV F1 : 0.6227 ± 0.0016

RÉSULTATS TFIDF_XGB :
  Accuracy   : 0.6675 (66.75%)
  F1 weighted: 0.6264
  F1 macro   : 0.4762

  F1 par classe :
    NEGATIF    : 0.7375  ██████████████
    NEUTRE     : 0.2584  █████
    POSITIF    : 0.7641  ███████████████
    SPAM       : 0.1449  ██
```

*Version 2 — AVEC sample_weight (déséquilibre géré) :*
```
CV F1 : 0.6227 ± 0.0016

RÉSULTATS TFIDF_XGB :
  Accuracy   : 0.5905 (59.05%)
  F1 weighted: 0.6165
  F1 macro   : 0.5112

  F1 par classe :
    NEGATIF    : 0.6772  █████████████
    NEUTRE     : 0.4019  ████████
    POSITIF    : 0.7119  ██████████████
    SPAM       : 0.2537  █████
```

> **Impact du sample_weight sur tfidf_xgb :**
>
> | Classe   | Sans sample_weight | Avec sample_weight | Évolution |
> |----------|-------------------|--------------------|-----------|
> | NEGATIF  | 0.7375            | 0.6772             | ↓ -8%     |
> | NEUTRE   | 0.2584            | 0.4019             | ↑ +56%    |
> | POSITIF  | 0.7641            | 0.7119             | ↓ -7%     |
> | SPAM     | 0.1449            | 0.2537             | ↑ +75%    |
> | F1 macro | 0.4762            | **0.5112**         | ↑ +7%     |
>
> Le sample_weight améliore fortement NEUTRE et SPAM (classes minoritaires) au prix
> d'une légère baisse sur POSITIF et NEGATIF. Le F1 macro (0.4762 → 0.5112) confirme
> que le modèle version 2 est plus équilibré entre toutes les classes.
> **Version retenue en production : avec sample_weight.**

**Modèle 4 — w2v_lr (Word2Vec + SMOTE)**
```
RÉSULTATS W2V_LR :
  Accuracy   : 0.5406 (54.06%)
  F1 weighted: 0.5849
  F1 macro   : 0.4731

  Distribution AVANT SMOTE :
    NEGATIF: 90,989 — NEUTRE: 45,537 — POSITIF: 90,817 — SPAM: 12,177
  Distribution APRÈS SMOTE :
    NEGATIF: 90,989 — NEUTRE: 90,989 — POSITIF: 90,989 — SPAM: 90,989

  F1 par classe :
    NEGATIF    : 0.6524  █████████████
    NEUTRE     : 0.3710  ███████
    POSITIF    : 0.6772  █████████████
    SPAM       : 0.1920  ███
```

> SMOTE a parfaitement équilibré les classes (90,989 exemples chacune).
> Malgré cela, Word2Vec (moyenne de vecteurs de mots) est moins performant
> que TF-IDF sur cette tâche. La moyenne des vecteurs perd l'information
> sur la fréquence et l'ordre des mots, ce que TF-IDF capture bien.

**Modèle 5 — sbert_lr (Sentence-BERT + LR)**
```
RÉSULTATS SBERT_LR :
  Accuracy   : 0.5823 (58.23%)
  F1 weighted: 0.5897
  F1 macro   : 0.4612

  F1 par classe :
    NEGATIF    : 0.6891  █████████████
    NEUTRE     : 0.3647  ███████
    POSITIF    : 0.7120  ██████████████
    SPAM       : 0.0789  █
```

> Note : ADASYN a échoué (espace de haute dimension 384D → densité uniforme → 0 samples).
> LR a utilisé class_weight="balanced" en interne. Un fix SMOTE est prévu pour la prochaine exécution.
> SBERT sans fine-tuning est inférieur à TF-IDF car ses embeddings sont génériques (non adaptés Amazon Reviews).

**Modèle 6 — distilbert (DistilBERT fine-tuné, 3 epochs)**
```
RÉSULTATS DISTILBERT :
  Accuracy   : 0.6968 (69.68%)
  F1 weighted: 0.7103
  F1 macro   : 0.6005

  F1 par classe :
    NEGATIF    : 0.7534  ███████████████
    NEUTRE     : 0.4925  █████████
    POSITIF    : 0.8273  ████████████████
    SPAM       : 0.3289  ██████
```

> DistilBERT fine-tuné est le meilleur modèle du projet avec F1=0.7103.
> Il dépasse tous les TF-IDF malgré seulement 3 epochs — avec plus d'epochs il progresserait encore.
> NEUTRE reste difficile (frontière floue avec POSITIF/NÉGATIF) et SPAM est rare (5% du dataset).
> Weighted cross-entropy loss : les erreurs sur SPAM coûtent ~7x plus cher que sur NÉGATIF.

**Classement final — 6 modèles (résultats réels mai 2026) :**
> 1. distilbert : F1=0.7103 ← MEILLEUR (fine-tuning, weighted cross-entropy)
> 2. tfidf_svm  : F1=0.6599 (class_weight="balanced")
> 3. tfidf_lr   : F1=0.6540 (class_weight="balanced")
> 4. tfidf_xgb  : F1=0.6165 (sample_weight)
> 5. sbert_lr   : F1=0.5897 (ADASYN échoué — LR class_weight interne)
> 6. w2v_lr     : F1=0.5849 (SMOTE)
>
> Observation clé : plus complexe ≠ toujours meilleur. SBERT (modèle transformer gelé) est
> moins performant que TF-IDF car il n'est pas adapté au domaine. DistilBERT fine-tuné
> gagne car il apprend directement sur les avis Amazon.

### Visualiser les résultats dans MLflow

```bash
mlflow ui
```

Puis ouvre **http://localhost:5000**

**Dans MLflow tu peux :**
- Voir tous les runs avec leurs métriques
- Sélectionner plusieurs runs → "Compare" → graphique comparatif des F1 scores
- Cliquer sur un run pour voir tous ses paramètres et télécharger le modèle
- Onglet "Models" → Model Registry (après la promotion étape 5)

---

## ETAPE 5 — Promotion du meilleur modèle (5 min)

### Pourquoi cette étape ?

L'étape 4 a entraîné plusieurs modèles. Cette étape choisit automatiquement
le meilleur et le met en **production** pour que l'API FastAPI l'utilise.

Le script vérifie : F1 >= 80% ? Meilleur que le modèle actuellement en prod ?
Si oui : promotion dans MLflow Model Registry + mise à jour de models/pipeline.pkl

```bash
python src/promote_model.py
```

**Ce que tu vois :**
```
============================================================
RECHERCHE DU MEILLEUR MODÈLE
============================================================
6 runs trouvés dans l'expérience 'sportreview-ai'.

Classement par F1 score weighted :
  1. distilbert    F1=0.7103  <- MEILLEUR  (réel — mai 2026)
  2. tfidf_svm     F1=0.6599             (réel — mai 2026)
  3. tfidf_lr      F1=0.6540             (réel — mai 2026)
  4. tfidf_xgb     F1=0.6165             (réel — mai 2026, avec sample_weight)
  5. sbert_lr      F1=0.5897             (réel — mai 2026, sans ADASYN)
  6. w2v_lr        F1=0.5849             (réel — mai 2026, avec SMOTE)

OK : F1=0.7103 >= seuil=0.65
OK : Nouveau modèle (F1=0.7103) > Production actuelle (F1=0.00)
Modèle v1 promu en PRODUCTION !
Pipeline sauvegardé : models/pipeline.pkl
```

---

## ETAPE 6 — Lancer l'API FastAPI (5 min)

FastAPI expose le modèle comme une API REST. N'importe quelle application
peut envoyer un avis en JSON et recevoir la prédiction.

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Ce que tu vois au démarrage :**
```
INFO: Chargement du modèle depuis models/pipeline.pkl...
INFO: Modèle chargé : distilbert (F1=0.7103)
INFO: Application startup complete.
INFO: Uvicorn running on http://0.0.0.0:8000
```

### Tests à effectuer

Ouvre **http://localhost:8000/docs** → interface Swagger pour tester visuellement.

Dans un nouveau terminal, teste avec curl :

```bash
# Test 1 : Health check
curl http://localhost:8000/health
# Réponse attendue : {"status": "healthy", "model_loaded": true, ...}

# Test 2 : Prédiction positive
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "Excellentes chaussures de trail, tres confortables !"}'
# Réponse attendue : {"label": "POSITIF", "confidence": 0.97, ...}

# Test 3 : Prédiction avec explication SHAP
curl -X POST "http://localhost:8000/predict?explain=true" \
     -H "Content-Type: application/json" \
     -d '{"text": "Produit de mauvaise qualite, je suis tres decu"}'
# Réponse attendue : {"label": "NEGATIF", ..., "explanation": {"top_positive_words": [...]}}

# Test 4 : Batch
curl -X POST http://localhost:8000/predict/batch \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Super velo !", "Qualite nulle", "PROMO cliquez www.fake.com"]}'
# Réponse attendue : 3 prédictions : POSITIF, NEGATIF, SPAM

# Test 5 : Feedback (correction humaine)
curl -X POST http://localhost:8000/feedback \
     -H "Content-Type: application/json" \
     -d '{"text": "Produit correct", "predicted_label": "NEUTRE", "correct_label": "POSITIF"}'

# Test 6 : Métriques Prometheus
curl http://localhost:8000/metrics
# Réponse : texte brut avec les métriques Prometheus (sportreview_requests_total, etc.)
```

---

## ETAPE 7 — Interface Gradio (5 min)

**L'API FastAPI doit tourner** pour que Gradio fonctionne.

```bash
# Dans un nouveau terminal
python demo/gradio_app.py
```

Ouvre **http://localhost:7860**

**L'interface a 3 onglets :**
- **Analyser un avis** : tape un avis → label en couleur + barre de confiance + explication SHAP
- **Batch** : colle plusieurs avis (un par ligne) → tableau de résultats
- **Infos modèle** : type, F1 score, classes, version

### Déployer sur HuggingFace Spaces (URL publique gratuite)

Les fichiers prêts à uploader sont dans le dossier `huggingface/` du projet.

**Ce qui a été fait (mai 2026) :**
- Space créé : https://huggingface.co/spaces/PatriciaWtop/SportReview_Ecommerce
- `huggingface/app.py` : version adaptée de l'interface Gradio
- `huggingface/requirements.txt` : uniquement `gradio` et `requests`

**Important :** Le Space affiche l'interface Gradio complète, mais le backend FastAPI
tourne en local. Quand il n'est pas connecté, un message professionnel s'affiche :
"Démo publique — Backend non connecté" avec la liste des composants du projet.
→ C'est une vitrine portfolio, pas une démo interactive publique.

**Pour une démo interactive complète :** lancer FastAPI et Gradio en local et
partager l'écran pendant l'entretien (visio ou présentiel).

---

## ETAPE 8 — Tests automatisés (5 min)

Les tests vérifient automatiquement que tout fonctionne. Ils sont déclenchés
par la CI/CD GitHub Actions à chaque push.

```bash
# Lancer TOUS les tests (51 tests)
pytest tests/ -v

# Avec rapport de couverture de code (HTML dans htmlcov/)
pytest tests/ -v --cov=app --cov=src --cov-report=html

# Seulement les tests API
pytest tests/test_api.py -v

# Seulement les tests preprocessing
pytest tests/test_preprocessing.py -v
```

**Ce que tu dois voir (résultats réels mai 2026) :**
```
tests/test_api.py::TestHealthEndpoint::test_health_returns_200 PASSED
tests/test_api.py::TestHealthEndpoint::test_health_status_is_healthy PASSED
tests/test_api.py::TestPredictEndpoint::test_predict_positive_review PASSED
tests/test_api.py::TestPredictEndpoint::test_predict_spam_review PASSED
...
tests/test_preprocessing.py::TestCleanText::test_converts_to_lowercase PASSED
tests/test_preprocessing.py::TestIsSpam::test_detects_url_in_unverified_review PASSED
...
=================== 51 passed in 7.65s ===================
```

**Répartition des 51 tests :**
- `test_api.py` : 32 tests — health check, prédiction unitaire, gestion d'erreurs,
  batch, model info, feedback, métriques Prometheus
- `test_preprocessing.py` : 19 tests — clean_text, is_spam, assign_label

### Comprendre le rapport de couverture de code

```bash
pytest tests/ -v --cov=app --cov=src --cov-report=html
```

Cette commande génère un dossier `htmlcov/`. Ouvre `htmlcov/index.html` dans
le navigateur pour voir un tableau de bord :

| Fichier | Lignes totales | Couvertes par les tests | % couverture |
|---------|---------------|------------------------|--------------|
| app/main.py | ~250 | ~198 | ~79% |
| src/train.py | ~180 | ~45 | ~25% |

- **Lignes vertes** = exécutées par au moins un test
- **Lignes rouges** = jamais testées → zones de risque

**Pourquoi c'est important en production :** Dans le monde pro, on fixe un seuil
minimum (souvent 70-80%). Si un développeur pousse du code qui fait tomber la
couverture sous ce seuil, la CI/CD bloque automatiquement le déploiement.
Le fichier `.github/workflows/ci.yml` de ce projet vérifie exactement ça à chaque push.

---

## ETAPE 9 — Détection de drift avec EvidentlyAI

### Qu'est-ce que le drift ?

Le data drift se produit quand les données en production divergent des données
d'entraînement. Exemple : entraîné en janvier sur des avis d'équipements de ski,
en juillet les utilisateurs écrivent des avis sur des équipements de surf.
Le vocabulaire change → le modèle fait de mauvaises prédictions.

Deux types de drift détectés :
- **Data drift** : la distribution des textes entrants change (longueur, nombre de mots)
- **Label drift** : la proportion de chaque classe change (plus de SPAM qu'avant ?)

### Installer EvidentlyAI

EvidentlyAI 0.5+ a changé son API — il faut impérativement la version 0.4.x :

```bash
pip install "evidently>=0.4.30,<0.5.0" --no-deps
```

> Le `--no-deps` évite un conflit de version avec numpy 2.x qui tenterait une
> compilation depuis les sources (nécessite un compilateur C non disponible par défaut).

### Lancer la détection

```bash
python src/drift_detection.py
```

**Ce que tu vois :**
```
============================================================
DÉTECTION DU DRIFT — SportReview AI
============================================================
Chargement du dataset de référence : data/processed/train.csv
Chargement du dataset courant : data/processed/test.csv
Calcul du rapport de drift avec EvidentlyAI...
Features numériques non trouvées — calcul des features de base...
Rapport HTML sauvegardé : reports/drift_report.html

Drift de distribution des labels :
Label              Reference      Courant     Derive
--------------------------------------------------
NEGATIF                38.0%        38.0%       0.0%
NEUTRE                 19.0%        19.0%       0.0%
POSITIF                37.9%        37.9%       0.0%
SPAM                    5.1%         5.1%       0.0%
Drift moyen : 0.0%

Résumé JSON sauvegardé : reports/drift_report.json
============================================================
✓ Pas de drift significatif détecté.
Score de drift : 0.000
Rapport HTML : reports/drift_report.html
============================================================
```

> **Résultat normal** : train et test viennent du même dataset splitté → distributions
> identiques → 0% de drift. En production réelle, tu comparerais train.csv avec
> de nouvelles données collectées sur la plateforme.

### Comprendre le rapport HTML

Ouvre `reports/drift_report.html` dans le navigateur. Les éléments pertinents :

- **Dataset Drift** (en haut) : verdict global — combien de colonnes ont drifté
- **Share of Drifted Columns** : 0.0 = aucune colonne en drift
- **Drift in column 'text_length'** : distance de Wasserstein entre train et test
  → score 0.007 = quasi-identique (normal pour un split du même dataset)
- **Drift in column 'word_count'** : même logique — nombre de mots par avis
- **Dataset Missing Values** : vérifie qu'aucune cellule n'est vide (0 = parfait)

> Les colonnes `text_length` et `word_count` sont calculées depuis la colonne `text`
> de chaque fichier CSV — ce sont les vraies longueurs de tes avis.

### Note sur le target drift (prédictions du modèle)

Le script est conçu pour inclure aussi les prédictions du modèle (confidence, classe
prédite) dans le rapport. Sur Windows, un conflit DLL entre EvidentlyAI et PyTorch
empêche le chargement du modèle DistilBERT localement. **En Docker (ETAPE 10), ce
problème n'existe pas** — Linux gère les DLL différemment et le target drift
s'affiche normalement dans le rapport.

---

## ETAPE 10 — Docker et docker-compose

### Étape 1 — Vérifier que Docker Desktop est lancé

Lance Docker Desktop et attends qu'il affiche "Docker Desktop is running".

### Étape 2 — Construire l'image Docker

> **Important** : le build inclut torch CPU (~1GB) + transformers pour DistilBERT.
> La première construction prend 15-20 min. Les suivantes sont rapides (cache Docker).

```bash
docker-compose up --build
```

> On utilise directement `docker-compose up --build` plutôt que `docker build` séparément,
> pour éviter les problèmes de cache entre les deux commandes.

### Étape 3 — Architecture du stack

docker-compose lance 4 services en parallèle :

| Service | Rôle | Port |
|---------|------|------|
| `api` | FastAPI — prédictions DistilBERT | 8000 |
| `mlflow` | Tracking des expériences ML | 5000 |
| `prometheus` | Collecte métriques toutes les 15s | 9090 |
| `grafana` | Dashboard de monitoring | 3000 |

> **Note MLflow** : MLflow 3.x dans Docker force localhost-only par sécurité.
> Lance MLflow séparément dans un terminal : `mlflow ui`
> Puis dans Docker : `docker-compose up` (API + Prometheus + Grafana)
> C'est une architecture valide — en prod le tracking server est souvent séparé.

### Étape 4 — Vérifier que tout fonctionne

**URLs disponibles :**

| Service | URL | Login |
|---------|-----|-------|
| API Swagger | http://localhost:8000/docs | — |
| MLflow (local) | http://localhost:5000 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / admin |

Teste l'API :
```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d "{\"text\": \"Excellent produit je recommande !\"}"
```

### Étape 5 — Prometheus : vérifier les targets

Va sur http://localhost:9090/targets — tu dois voir :

```
prometheus      localhost:9090    UP
sportreview-api api:8000          UP
```

> Les liens dans la page Prometheus pointent vers les adresses internes Docker
> (`api:8000`) — normaux, ton navigateur ne peut pas les ouvrir mais
> Prometheus les atteint depuis l'intérieur du réseau Docker.

### Étape 6 — Grafana : voir les métriques en temps réel

1. Ouvre http://localhost:3000 → login `admin` / `admin`
2. Va dans **Dashboards** → tu vois **"SportReview AI"**
3. Génère du trafic pour remplir les graphiques :

```bash
for i in 1 2 3 4 5 6 7 8 9 10; do
  curl -s -X POST http://localhost:8000/predict ^
       -H "Content-Type: application/json" ^
       -d "{\"text\": \"Super produit livraison rapide\"}" > nul
done
```

4. En haut à droite du dashboard, passe le rafraîchissement à **5s**
5. Tu vois en direct : compteur de prédictions par classe, latence, confiance

**Ce que montrent les panneaux Grafana :**
Prometheus collecte toutes les 15 secondes les métriques exposées par `/metrics`
(compteurs de prédictions, latences, scores de confiance). Grafana interroge
Prometheus et les affiche sous forme de graphiques. Les liens internes Docker
(`api:8000`) ne sont pas accessibles depuis ton navigateur mais Prometheus
y accède normalement depuis l'intérieur du réseau Docker.

---

## ETAPE 11 — GitHub Actions CI/CD (10 min)

### Configurer le repo

```bash
# Initialise Git
git init
git add .
git commit -m "feat: SportReview AI MLOps pipeline complet"

# Connecte au repo GitHub (créé sur github.com)
git remote add origin https://github.com/TON_USERNAME/sportreview-ai.git
git branch -M main
git push -u origin main
```

### Observer le pipeline

Va sur GitHub → onglet **Actions**. Tu vois 3 jobs s'enchaîner :

- **Job 1 — Tests** : installe les dépendances, télécharge 5000 avis,
  preprocessing pandas, entraîne TF-IDF+LR rapide, 44 tests pytest,
  vérifie couverture ≥ 70%

- **Job 2 — Docker** (démarre seulement si Job 1 OK) :
  build de l'image, smoke test (sleep 15 secondes, curl /health + /predict)

- **Job 3 — Push Registry** (démarre seulement si Job 2 OK, uniquement sur main) :
  login à ghcr.io avec GITHUB_TOKEN, push image avec tag SHA + latest

Si un job échoue → les jobs suivants sont automatiquement annulés.
Clique sur le job pour voir les logs détaillés.

---

## ETAPE 12 — Kubernetes avec minikube (obligatoire)

### Qu'est-ce que Kubernetes ?

Kubernetes (K8s) orchestre les containers en production :
- Lance plusieurs copies (replicas) pour la haute disponibilité
- Redémarre automatiquement les containers qui plantent (self-healing)
- Répartit la charge entre les replicas (load balancing)
- Scale automatiquement le nombre de replicas selon la charge (autoscaling)

Sans K8s : 1 container, s'il plante l'API est down.
Avec K8s : 3 replicas, si 1 plante les 2 autres continuent.

---

### Installation de minikube et kubectl

#### Étape A — Installer minikube

```powershell
# Via winget (gestionnaire de paquets Windows)
winget install Kubernetes.minikube

# Vérification
minikube version
```
**Tu dois voir :**
```
minikube version: v1.33.x
commit: abc123...
```

#### Étape B — Installer kubectl

kubectl est l'outil CLI pour interagir avec Kubernetes.

```powershell
winget install Kubernetes.kubectl

# Vérification
kubectl version --client
```
**Tu dois voir :**
```
Client Version: v1.31.x
Kustomize Version: v5.x.x
```

#### Étape C — Démarrer le cluster minikube

```bash
minikube start
```

La première fois télécharge une VM (~300 MB) → attends 2-5 minutes.

**Ce que tu vois :**
```
* minikube v1.33.x on Windows
* Automatically selected the docker driver
* Creating docker container (CPUs=2, Memory=4096MB) ...
* Preparing Kubernetes v1.31.x on Docker ...
* Verifying Kubernetes components...
* Done! kubectl is now configured to use "minikube"
```

#### Étape D — Vérifier le cluster

```bash
kubectl cluster-info
# Résultat attendu : "Kubernetes control plane is running at https://127.0.0.1:xxxxx"

kubectl get nodes
# Résultat attendu :
# NAME       STATUS   ROLES           AGE   VERSION
# minikube   Ready    control-plane   2m    v1.31.x
```

---

### Déployer l'application sur Kubernetes

#### Étape 1 — Déployer les ressources

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

**Ce que tu vois :**
```
deployment.apps/sportreview-api created
service/sportreview-api-service created
horizontalpodautoscaler.autoscaling/sportreview-api-hpa created
```

#### Étape 2 — Vérifier que les 3 pods démarrent

```bash
kubectl get pods
```

**Attends 30-60 secondes et tu dois voir :**
```
NAME                              READY   STATUS    RESTARTS   AGE
sportreview-api-7d4b8c9f6-abc12   1/1     Running   0          45s
sportreview-api-7d4b8c9f6-def34   1/1     Running   0          45s
sportreview-api-7d4b8c9f6-ghi56   1/1     Running   0          45s
```

Si STATUS = CrashLoopBackOff, vois les logs :
```bash
kubectl logs sportreview-api-7d4b8c9f6-abc12
```

#### Étape 3 — Vérifier le service et l'autoscaler

```bash
kubectl get services
# Tu dois voir : sportreview-api-service de type LoadBalancer sur port 80

kubectl get hpa
# Tu dois voir : sportreview-api-hpa avec TARGETS 5%/70%, MINPODS 2, MAXPODS 10
```

#### Étape 4 — Accéder à l'API

```bash
# Obtenir l'URL d'accès minikube
minikube service sportreview-api-service --url
# Résultat : http://127.0.0.1:55234 (le port change à chaque fois)

# Tester l'API Kubernetes avec l'URL obtenue
curl http://127.0.0.1:55234/health
# Résultat attendu : {"status": "healthy", ...}
```

#### Étape 5 — Observer l'autoscaling en action

**Terminal 1 — surveille les pods en temps réel :**
```bash
kubectl get pods -w
```

**Terminal 2 — génère de la charge :**
```bash
kubectl run -it --rm load-generator \
  --image=busybox \
  -- sh -c "while true; do wget -q -O- http://sportreview-api-service/health; done"
```

**Ce que tu observes dans Terminal 1 après ~2 minutes :**
```
# Kubernetes détecte CPU > 70% et crée des pods supplémentaires
sportreview-api-7d4b8c9f6-jkl78   0/1   Pending   0   5s
sportreview-api-7d4b8c9f6-jkl78   1/1   Running   0   30s
sportreview-api-7d4b8c9f6-mno90   0/1   Pending   0   5s
```

Appuie sur Ctrl+C dans Terminal 2 → Kubernetes réduit progressivement les replicas.

---

## Ce que tu peux dire en entretien chez Decathlon

### Sur le dataset et les labels

> "J'ai utilisé le vrai dataset Amazon Sports & Outdoors via HuggingFace — 19,5 millions
> d'avis authentiques. J'ai travaillé avec un échantillon stratifié de 300 000 avis
> pour l'entraînement, ce qui est la pratique standard en MLOps : on développe sur un
> échantillon représentatif, puis on scale sur cluster cloud pour la prod.
> J'ai construit 4 classes : POSITIF/NÉGATIF selon la note, NEUTRE pour 3 étoiles,
> et SPAM détecté par règles regex sur les avis non vérifiés avec patterns promotionnels.
> La règle SPAM a la priorité absolue."

### Sur la gestion du déséquilibre

> "Après échantillonnage stratifié, la distribution est : 38% NÉGATIF, 38% POSITIF,
> 19% NEUTRE et 5% SPAM. J'ai utilisé 4 techniques différentes selon le modèle :
> class_weight=balanced pour LR et SVM, sample_weight calculé par classe pour XGBoost,
> SMOTE pour Word2Vec qui génère des exemples synthétiques en interpolant dans l'espace
> des features, ADASYN pour SBERT qui est plus intelligent car il génère là où le modèle
> commet le plus d'erreurs, et une weighted cross-entropy loss pour DistilBERT.
> J'ai comparé les F1 par classe dans MLflow pour mesurer l'effet sur NEUTRE et SPAM."

### Sur les modèles et MLflow

> "J'ai entraîné et comparé 6 modèles dans MLflow : du TF-IDF baseline jusqu'au
> fine-tuning DistilBERT. Chaque run est loggé avec paramètres, métriques, matrice
> de confusion et rapport de classification comme artifacts. Sur les 3 modèles TF-IDF,
> tfidf_svm donne le meilleur F1 weighted à 0.66 contre 0.65 pour le baseline LR.
> Les modèles à embeddings sémantiques (Word2Vec, SBERT, DistilBERT) sont en cours
> et devraient significativement améliorer la détection de NEUTRE et SPAM."

### Sur Apache Spark

> "J'ai utilisé Spark pour le preprocessing distribué. Le dataset source fait 19,5 millions
> d'avis — j'ai travaillé avec 300 000 pour le développement, mais la même architecture
> Spark traite les 19,5M sans changer une ligne de code : il suffit de retirer le paramètre
> --max_samples. Spark divise automatiquement les données en partitions et les traite
> en parallèle sur les 16 coeurs CPU. Le nettoyage de texte utilise les fonctions SQL
> natives Spark (regexp_replace, lower, trim) plutôt que des Python UDFs — c'est un
> choix de performance : les fonctions SQL s'exécutent directement dans la JVM et sont
> 3 à 5x plus rapides que les UDFs qui spawneraient des workers Python séparés.
> C'est exactement l'architecture qu'on utiliserait en production sur un cluster."

### Sur Docker et Kubernetes

> "L'API est containerisée avec Docker multi-stage. En Kubernetes, j'ai 3 replicas
> avec liveness probe sur /health et un HorizontalPodAutoscaler qui scale de 2
> à 10 replicas selon l'utilisation CPU. J'ai observé l'autoscaling en temps réel."

### Sur le monitoring

> "J'expose 9 métriques Prometheus sur /metrics : distribution des prédictions,
> latence p50/p95/p99, score de confiance, taux d'erreurs. Grafana les visualise
> avec un dashboard de 11 panels. EvidentlyAI calcule le drift quotidiennement
> et alerte si la distribution dépasse 10% de dérive."

### Sur la CI/CD

> "GitHub Actions déclenche 3 jobs à chaque push : tests pytest avec couverture 70%,
> build Docker avec smoke test, push vers GitHub Container Registry avec le hash
> du commit comme tag. On ne peut pas déployer du code cassé — les jobs sont bloqués
> si les tests échouent."
