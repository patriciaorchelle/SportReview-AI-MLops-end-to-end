# -*- coding: utf-8 -*-
"""
test_preprocessing.py — Tests du pipeline de preprocessing

Teste les fonctions de nettoyage du texte et de détection de spam.
Ces tests n'ont pas besoin du modèle ML — ils testent seulement
les fonctions Python pures de preprocessing.

Lancer :
    pytest tests/test_preprocessing.py -v
"""

import sys
import os
import pandas as pd
import pytest

# Ajoute le dossier racine au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spark.preprocess import clean_text, extract_features, is_spam
from data.load_data import assign_label


# ─────────────────────────────────────────────────────────────────────────────
# TESTS CLEAN_TEXT
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanText:
    """Tests de la fonction clean_text."""

    def test_converts_to_lowercase(self):
        """Le texte doit être converti en minuscules."""
        result = clean_text("EXCELLENT PRODUIT")
        assert result == "excellent produit"

    def test_removes_html_tags(self):
        """Les balises HTML doivent être supprimées."""
        result = clean_text("Bon produit <br/> je recommande <p>vraiment</p>")
        assert "<br" not in result
        assert "<p>" not in result
        assert "bon produit" in result

    def test_removes_urls(self):
        """Les URLs doivent être supprimées."""
        result = clean_text("Achetez sur http://fake-shop.com maintenant !")
        assert "http" not in result
        assert "fake-shop" not in result

    def test_removes_special_characters(self):
        """Les caractères spéciaux doivent être supprimés."""
        result = clean_text("Produit @#$% excellent !!!")
        assert "@" not in result
        assert "#" not in result
        assert "$" not in result

    def test_reduces_multiple_spaces(self):
        """Les espaces multiples doivent être réduits à un seul."""
        result = clean_text("Bon   produit    excellent")
        assert "  " not in result

    def test_handles_empty_string(self):
        """Un texte vide doit retourner une chaîne vide."""
        result = clean_text("")
        assert result == ""

    def test_handles_none(self):
        """None doit retourner une chaîne vide."""
        result = clean_text(None)
        assert result == ""

    def test_strips_leading_trailing_spaces(self):
        """Les espaces en début/fin doivent être supprimés."""
        result = clean_text("   bon produit   ")
        assert result == "bon produit"

    def test_normalizes_accents(self):
        """Les accents doivent être normalisés (é→e, etc.)."""
        result = clean_text("Très déçu, qualité médiocre")
        # Après normalisation, les accents sont supprimés
        assert "é" not in result
        assert "è" not in result


# ─────────────────────────────────────────────────────────────────────────────
# TESTS IS_SPAM
# ─────────────────────────────────────────────────────────────────────────────

class TestIsSpam:
    """Tests de la détection de spam."""

    def test_detects_url_in_unverified_review(self):
        """Un avis non vérifié avec URL doit être détecté comme spam."""
        row = pd.Series({
            "text": "Achetez sur www.promo-fake.com maintenant !",
            "verified_purchase": False
        })
        assert is_spam(row) is True

    def test_legitimate_review_not_spam(self):
        """Un avis vérifié normal ne doit pas être classé spam."""
        row = pd.Series({
            "text": "Excellent produit, je suis très satisfait de mon achat.",
            "verified_purchase": True
        })
        assert is_spam(row) is False

    def test_verified_with_url_not_spam(self):
        """Un avis vérifié avec URL peut ne pas être spam (ex: blog)."""
        row = pd.Series({
            "text": "J ai fait un test complet sur http://mon-blog.com",
            "verified_purchase": True
        })
        # Achat vérifié → pas automatiquement spam
        # (le modèle ML décidera ensuite)
        assert is_spam(row) is False

    def test_short_unverified_review_is_spam(self):
        """Un avis court et non vérifié est probablement du spam."""
        row = pd.Series({
            "text": "Super !!!",
            "verified_purchase": False
        })
        assert is_spam(row) is True


# ─────────────────────────────────────────────────────────────────────────────
# TESTS ASSIGN_LABEL
# ─────────────────────────────────────────────────────────────────────────────

class TestAssignLabel:
    """Tests de la fonction assign_label."""

    def test_5_stars_is_positif(self):
        """Une note de 5 étoiles doit donner POSITIF."""
        row = pd.Series({"text": "Bon produit", "rating": 5, "verified_purchase": True})
        assert assign_label(row) == "POSITIF"

    def test_4_stars_is_positif(self):
        """Une note de 4 étoiles doit donner POSITIF."""
        row = pd.Series({"text": "Bien", "rating": 4, "verified_purchase": True})
        assert assign_label(row) == "POSITIF"

    def test_3_stars_is_neutre(self):
        """Une note de 3 étoiles doit donner NEUTRE."""
        row = pd.Series({"text": "Correct", "rating": 3, "verified_purchase": True})
        assert assign_label(row) == "NEUTRE"

    def test_2_stars_is_negatif(self):
        """Une note de 2 étoiles doit donner NEGATIF."""
        row = pd.Series({"text": "Decevant", "rating": 2, "verified_purchase": True})
        assert assign_label(row) == "NEGATIF"

    def test_1_star_is_negatif(self):
        """Une note de 1 étoile doit donner NEGATIF."""
        row = pd.Series({"text": "Terrible", "rating": 1, "verified_purchase": True})
        assert assign_label(row) == "NEGATIF"

    def test_spam_overrides_rating(self):
        """Le spam doit être détecté même si la note est 5 étoiles."""
        row = pd.Series({
            "text": "PROMO cliquez sur www.fake.com maintenant !",
            "rating": 5,
            "verified_purchase": False
        })
        assert assign_label(row) == "SPAM"
