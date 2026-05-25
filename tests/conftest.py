# -*- coding: utf-8 -*-
"""
conftest.py — Configuration pytest pour tous les tests

Ce fichier est chargé automatiquement par pytest avant d'exécuter les tests.
Il configure le sys.path pour que les imports fonctionnent correctement
depuis le dossier tests/.
"""

import sys
import os

# Ajoute le dossier racine du projet au path Python
# → permet d'écrire "from app.main import app" dans les tests
# → sans ça, pytest ne saurait pas où trouver les modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
