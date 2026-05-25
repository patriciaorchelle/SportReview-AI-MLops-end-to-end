# -*- coding: utf-8 -*-
"""
schemas.py — Modèles de données Pydantic pour l'API FastAPI

Pydantic valide automatiquement les données d'entrée et de sortie de l'API.
Si les données ne correspondent pas au schéma, FastAPI retourne une erreur 422
avec un message clair — sans qu'on ait à écrire de code de validation.

Ces schémas servent aussi à générer automatiquement la documentation
Swagger UI accessible sur /docs.
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# SCHÉMAS D'ENTRÉE (requêtes envoyées à l'API)
# ─────────────────────────────────────────────────────────────────────────────

class ReviewInput(BaseModel):
    """
    Schéma d'entrée pour une prédiction sur un seul avis.

    Field() permet de définir des contraintes et d'ajouter de la documentation
    qui apparaît automatiquement dans Swagger UI.
    """
    text: str = Field(
        ...,                                    # ... = champ obligatoire
        min_length=5,                           # minimum 5 caractères
        max_length=5000,                        # maximum 5000 caractères
        description="Texte de l'avis produit",
        examples=["Excellent produit, tres bonne qualite, je recommande vivement !"]
    )

    # Optionnel : informations sur l'achat pour améliorer la détection spam
    verified_purchase: Optional[bool] = Field(
        default=True,
        description="L'achat a-t-il été vérifié ? (améliore la détection spam)"
    )

    # Validator personnalisé : rejette les textes vides ou avec seulement des espaces
    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v):
        if not v.strip():
            raise ValueError("Le texte ne peut pas être vide ou ne contenir que des espaces")
        return v.strip()  # supprime les espaces en début/fin

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "Super chaussures de running, confortables et durables !",
                "verified_purchase": True
            }
        }
    }


class BatchInput(BaseModel):
    """
    Schéma d'entrée pour des prédictions en lot (batch).
    Permet d'envoyer jusqu'à 500 avis en une seule requête.
    """
    texts: List[str] = Field(
        ...,
        min_length=1,                           # au moins 1 texte
        max_length=500,                         # maximum 500 textes par batch
        description="Liste de textes à classifier"
    )

    @field_validator("texts")
    @classmethod
    def texts_must_not_be_empty(cls, v):
        if not v:
            raise ValueError("La liste de textes est vide")
        # Filtre les textes vides
        cleaned = [t.strip() for t in v if t and t.strip()]
        if not cleaned:
            raise ValueError("Tous les textes sont vides après nettoyage")
        return cleaned

    model_config = {
        "json_schema_extra": {
            "example": {
                "texts": [
                    "Excellent produit, je recommande !",
                    "Tres decu, qualite mediocre",
                    "PROMO INCROYABLE cliquez maintenant !!!",
                    "Produit correct, rien d exceptionnel"
                ]
            }
        }
    }


class FeedbackInput(BaseModel):
    """
    Schéma pour signaler une mauvaise prédiction.

    Cet endpoint permet de collecter des corrections humaines
    pour améliorer le modèle lors du prochain ré-entraînement.
    C'est le début d'une boucle de feedback humain (Human-in-the-Loop).
    """
    text: str = Field(..., description="Le texte de l'avis")
    predicted_label: str = Field(..., description="Label prédit par le modèle")
    correct_label: str = Field(..., description="Label correct selon l'utilisateur")
    comment: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Commentaire optionnel sur la correction"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "Super produit !",
                "predicted_label": "NEUTRE",
                "correct_label": "POSITIF",
                "comment": "Le modèle a sous-estimé la positivité de cet avis"
            }
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# SCHÉMAS DE SORTIE (réponses retournées par l'API)
# ─────────────────────────────────────────────────────────────────────────────

class SHAPExplanation(BaseModel):
    """
    Explication SHAP de la prédiction.

    SHAP (SHapley Additive exPlanations) indique quels mots
    ont le plus contribué à la prédiction, et dans quel sens.
    Valeur positive = contribue vers ce label
    Valeur négative = contribue contre ce label
    """
    top_positive_words: Dict[str, float] = Field(
        description="Mots qui poussent vers le label prédit (+ = favorable)"
    )
    top_negative_words: Dict[str, float] = Field(
        description="Mots qui vont contre le label prédit (- = défavorable)"
    )


class PredictionOutput(BaseModel):
    """
    Schéma de sortie pour une prédiction.

    Contient le label, la confiance, les probabilités de toutes les classes,
    et optionnellement une explication SHAP.
    """
    text: str = Field(description="Texte original de l'avis")
    label: str = Field(description="Classe prédite : POSITIF, NEGATIF, NEUTRE ou SPAM")
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Score de confiance entre 0 et 1"
    )
    probabilities: Dict[str, float] = Field(
        description="Probabilités pour chaque classe (somme = 1.0)"
    )
    is_reliable: bool = Field(
        description="True si la confiance dépasse le seuil de fiabilité (60%)"
    )
    explanation: Optional[SHAPExplanation] = Field(
        default=None,
        description="Explication SHAP des mots les plus importants (si demandé)"
    )
    timestamp: str = Field(description="Horodatage de la prédiction (UTC)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "Super chaussures de running !",
                "label": "POSITIF",
                "confidence": 0.924,
                "probabilities": {
                    "POSITIF": 0.924,
                    "NEGATIF": 0.032,
                    "NEUTRE": 0.028,
                    "SPAM": 0.016
                },
                "is_reliable": True,
                "explanation": None,
                "timestamp": "2026-05-23T14:30:00"
            }
        }
    }


class BatchOutput(BaseModel):
    """Schéma de sortie pour une prédiction en lot."""
    predictions: List[PredictionOutput] = Field(
        description="Liste des prédictions"
    )
    count: int = Field(description="Nombre total de prédictions")
    processing_time_ms: float = Field(
        description="Temps de traitement en millisecondes"
    )


class ModelInfoOutput(BaseModel):
    """Métadonnées du modèle actuellement chargé."""
    model_type: str = Field(description="Type de modèle (ex: TF-IDF + Logistic Regression)")
    classes: List[str] = Field(description="Liste des classes supportées")
    version: Optional[str] = Field(default=None, description="Version du modèle")
    f1_score: Optional[float] = Field(default=None, description="F1 score sur le test set")
    run_id: Optional[str] = Field(default=None, description="ID du run MLflow")
    loaded_at: str = Field(description="Date/heure de chargement du modèle")


class HealthOutput(BaseModel):
    """Schéma de sortie pour le health check."""
    status: str = Field(description="'healthy' ou 'unhealthy'")
    model_loaded: bool = Field(description="True si le modèle est bien chargé en mémoire")
    classes: List[str] = Field(description="Classes supportées par le modèle")
    uptime_seconds: float = Field(description="Temps de fonctionnement du serveur en secondes")
    timestamp: str = Field(description="Horodatage UTC")


class FeedbackOutput(BaseModel):
    """Confirmation d'enregistrement du feedback."""
    message: str
    feedback_id: str
    timestamp: str


class DriftOutput(BaseModel):
    """Résultats du dernier rapport de drift."""
    drift_score: float = Field(description="Score de drift global (0=aucun drift, 1=drift total)")
    drift_detected: bool = Field(description="True si le drift dépasse le seuil")
    threshold: float = Field(description="Seuil de drift configuré")
    label_distribution: Optional[dict] = Field(default=None)
    generated_at: Optional[str] = Field(default=None)
    report_available: bool = Field(description="True si le rapport HTML est disponible")
