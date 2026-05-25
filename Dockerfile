# ─────────────────────────────────────────────────────────────────────────────
# Dockerfile — SportReview AI
#
# Multi-stage build :
#   Stage 1 (builder) : installe les dépendances lourdes
#   Stage 2 (runtime) : image finale légère sans les outils de build
#
# Commandes :
#   docker build -t sportreview-ai:v1.0 .
#   docker run -p 8000:8000 sportreview-ai:v1.0
# ─────────────────────────────────────────────────────────────────────────────

# ── STAGE 1 : BUILDER ────────────────────────────────────────────────────────
# On utilise une image complète pour installer les dépendances
FROM python:3.11-slim AS builder

# Évite les questions interactives pendant l'installation des paquets système
ENV DEBIAN_FRONTEND=noninteractive

# Installe les outils système nécessaires pour compiler certaines librairies Python
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*   # nettoie le cache apt pour réduire la taille

# Définit le dossier de travail dans le container
WORKDIR /app

# Copie d'abord requirements.txt SEUL — optimisation du cache Docker :
# si requirements.txt ne change pas, Docker réutilise le cache de pip install
COPY requirements.txt .

# Installe les dépendances Python sans cache pip (réduit la taille de l'image)
# On n'installe PAS pyspark ni gradio dans l'image API (trop lourd)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        scikit-learn \
        pandas \
        numpy \
        xgboost \
        sentence-transformers \
        shap \
        mlflow \
        fastapi \
        "uvicorn[standard]" \
        pydantic \
        prometheus-client \
        gensim \
        "evidently>=0.4.30,<0.5.0" \
        requests \
        transformers && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu


# ── STAGE 2 : RUNTIME ────────────────────────────────────────────────────────
# Image finale : on repart d'une base propre et on copie seulement ce qu'il faut
FROM python:3.11-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Variables d'environnement pour optimiser Python dans Docker
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Copie les librairies installées depuis le stage builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copie le code de l'application
COPY app/ ./app/
COPY src/ ./src/
COPY models/ ./models/

# Expose le port sur lequel uvicorn va écouter
EXPOSE 8000

# ── HEALTHCHECK ──────────────────────────────────────────────────────────────
# Kubernetes et Docker appellent cet endpoint toutes les 30s pour vérifier
# que le container est sain. Si 3 échecs consécutifs → container redémarré.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# ── COMMANDE DE DÉMARRAGE ────────────────────────────────────────────────────
# Lance uvicorn sur 0.0.0.0 (obligatoire dans Docker pour accepter le trafic externe)
# --workers 2 : 2 processus parallèles pour gérer plus de requêtes simultanées
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
