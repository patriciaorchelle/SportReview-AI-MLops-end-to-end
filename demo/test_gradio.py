# -*- coding: utf-8 -*-
"""
gradio_app.py — Interface de démonstration Gradio

Cette interface permet à n'importe qui (recruteurs, équipes non-techniques)
de tester le modèle sans avoir à écrire du code.

Déploiement local :
    python demo/gradio_app.py

Déploiement public sur HuggingFace Spaces :
    1. Crée un Space sur https://huggingface.co/spaces
    2. Choisis "Gradio" comme type
    3. Upload ce fichier comme app.py
    4. L'URL publique sera : https://huggingface.co/spaces/TON_USERNAME/sportreview-ai

L'URL publique peut être partagée avec les recruteurs avant l'entretien !
"""

import os
import sys
import json
import requests

# Ajoute le dossier racine au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import gradio as gr
except ImportError:
    raise ImportError("Gradio non installé. Lance : pip install gradio")

# URL de l'API FastAPI
# En local : http://localhost:8000
# En production : URL du serveur
API_URL = os.getenv("API_URL", "http://localhost:8000")


# ─────────────────────────────────────────────────────────────────────────────
# COULEURS PAR LABEL (pour l'affichage visuel)
# ─────────────────────────────────────────────────────────────────────────────

LABEL_COLORS = {
    "POSITIF": "#22c55e",   # vert
    "NEGATIF": "#ef4444",   # rouge
    "NEUTRE":  "#f59e0b",   # orange
    "SPAM":    "#8b5cf6",   # violet
}

LABEL_EMOJIS = {
    "POSITIF": "✅",
    "NEGATIF": "❌",
    "NEUTRE":  "⚖️",
    "SPAM":    "🚫",
}


# ─────────────────────────────────────────────────────────────────────────────
# FONCTION DE PRÉDICTION — appelée par Gradio à chaque soumission
# ─────────────────────────────────────────────────────────────────────────────

def predict_review(text: str, include_explanation: bool = True) -> tuple:
    """
    Appelle l'API FastAPI et formate le résultat pour Gradio.

    Args:
        text               : texte de l'avis saisi par l'utilisateur
        include_explanation: si True, demande l'explication SHAP

    Returns:
        tuple : (label_html, confidence_bar, probabilities_html, explanation_html)
    """
    if not text or not text.strip():
        return "⚠️ Veuillez saisir un texte", 0.0, "", ""

    try:
        # Appel à l'API FastAPI
        response = requests.post(
            f"{API_URL}/predict",
            params={"explain": include_explanation},
            json={"text": text, "verified_purchase": True},
            timeout=120
        )

        if response.status_code != 200:
            return f"❌ Erreur API : {response.status_code}", 0.0, "", ""

        data = response.json()
        label = data["label"]
        confidence = data["confidence"]
        probabilities = data["probabilities"]
        is_reliable = data["is_reliable"]
        explanation = data.get("explanation")

        # ── Formatage du label ─────────────────────────────────────────────
        emoji = LABEL_EMOJIS.get(label, "")
        reliability_note = "" if is_reliable else " (confiance faible)"
        label_display = f"{emoji} **{label}**{reliability_note}"

        # ── Formatage des probabilités ────────────────────────────────────
        proba_lines = []
        for cls, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            bar_width = int(prob * 100)
            color = LABEL_COLORS.get(cls, "#6b7280")
            emoji_cls = LABEL_EMOJIS.get(cls, "")
            proba_lines.append(
                f"{emoji_cls} {cls}: **{prob*100:.1f}%**"
            )
        probabilities_text = "\n".join(proba_lines)

        # ── Formatage de l'explication SHAP ──────────────────────────────
        explanation_text = ""
        if explanation and (explanation.get("top_positive_words") or explanation.get("top_negative_words")):
            pos_words = explanation.get("top_positive_words", {})
            neg_words = explanation.get("top_negative_words", {})

            if pos_words:
                pos_list = ", ".join([f"**{w}**" for w in pos_words.keys()])
                explanation_text += f"✅ Mots qui poussent vers {label} : {pos_list}\n\n"

            if neg_words:
                neg_list = ", ".join([f"**{w}**" for w in neg_words.keys()])
                explanation_text += f"❌ Mots qui vont contre {label} : {neg_list}"

        if not explanation_text:
            explanation_text = "Explication non disponible pour ce type de modèle."

        bar = "█" * int(confidence * 20) + "░" * (20 - int(confidence * 20))
        confidence_text = f"**Score de confiance : {confidence*100:.1f}%**\n\n`{bar}`"
        return label_display, confidence_text, probabilities_text, explanation_text

    except requests.exceptions.ConnectionError:
        return (
            "❌ API non accessible",
            "",
            "Lance l'API avec : uvicorn app.main:app --reload",
            ""
        )
    except Exception as e:
        return f"❌ Erreur : {str(e)}", "", "", ""


def predict_batch_demo(texts: str) -> str:
    """
    Prédit plusieurs avis à la fois (un par ligne).

    Args:
        texts : textes séparés par des sauts de ligne

    Returns:
        Résultats formatés en tableau markdown
    """
    lines = [t.strip() for t in texts.split("\n") if t.strip()]

    if not lines:
        return "⚠️ Aucun texte saisi"

    if len(lines) > 20:
        return "⚠️ Maximum 20 avis en démo (API supporte 500)"

    try:
        response = requests.post(
            f"{API_URL}/predict/batch",
            json={"texts": lines},
            timeout=60
        )

        if response.status_code != 200:
            return f"❌ Erreur API : {response.status_code}"

        data = response.json()
        predictions = data["predictions"]

        # Formate en tableau markdown
        result = f"**{data['count']} avis analysés en {data['processing_time_ms']:.0f}ms**\n\n"
        result += "| Avis | Label | Confiance |\n"
        result += "|------|-------|----------|\n"

        for pred in predictions:
            text_preview = pred["text"][:60] + "..." if len(pred["text"]) > 60 else pred["text"]
            emoji = LABEL_EMOJIS.get(pred["label"], "")
            result += f"| {text_preview} | {emoji} {pred['label']} | {pred['confidence']*100:.0f}% |\n"

        return result

    except requests.exceptions.ConnectionError:
        return "❌ API non accessible. Lance : uvicorn app.main:app --reload"
    except Exception as e:
        return f"❌ Erreur : {str(e)}"


def get_model_info() -> str:
    """Récupère et formate les infos du modèle en production."""
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=5)
        if response.status_code == 200:
            info = response.json()
            return (
                f"**Type** : {info['model_type']}\n\n"
                f"**Classes** : {', '.join(info['classes'])}\n\n"
                f"**F1 Score** : {info.get('f1_score', 'N/A')}\n\n"
                f"**Version** : {info.get('version', 'N/A')}\n\n"
                f"**Chargé le** : {info['loaded_at']}"
            )
    except Exception:
        pass
    return "API non accessible"


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRUCTION DE L'INTERFACE GRADIO
# ─────────────────────────────────────────────────────────────────────────────

# Exemples d'avis pour aider l'utilisateur à tester
EXAMPLES = [
    ["Excellentes chaussures de trail, confortables et très durables !", True],
    ["Très déçu, la fermeture éclair a lâché après 2 semaines d'utilisation.", True],
    ["Produit correct, rien d'exceptionnel mais fait le travail.", True],
    ["PROMO INCROYABLE!! Achetez maintenant sur notre site clickici.com !!!!", False],
    ["The running shoes are amazing, best purchase ever! Highly recommend.", True],
    ["Poor quality, the sole came off after first use. Very disappointed.", True],
]

# Interface principale
with gr.Blocks(
    title="SportReview AI — Classificateur d'Avis Sportifs",
) as demo:

    # ── En-tête ──────────────────────────────────────────────────────────────
    gr.Markdown("""
    # 🏃 SportReview AI
    ### Classificateur d'avis produits sportifs

    Ce modèle classifie automatiquement les avis en **POSITIF**, **NÉGATIF**, **NEUTRE** ou **SPAM**.
    Entraîné sur 300 000 avis Amazon Sports & Outdoors avec MLOps complet (MLflow, Docker, Kubernetes).

    ---
    """)

    # ── Onglet 1 : Prédiction unitaire ───────────────────────────────────────
    with gr.Tab("🔍 Analyser un avis"):
        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Texte de l'avis",
                    placeholder="Entrez votre avis produit ici...",
                    lines=4,
                    max_lines=10,
                )
                explain_checkbox = gr.Checkbox(
                    label="Inclure l'explication SHAP",
                    value=False,
                )
                predict_btn = gr.Button(
                    "🔍 Analyser",
                    variant="primary",
                    size="lg",
                )
            with gr.Column(scale=1):
                label_output = gr.Markdown(value="*Résultat ici*")
                # gr.Slider(interactive=False) supprimé — capture les clics dans Gradio 6.14
                confidence_output = gr.Markdown(value="")

        with gr.Row():
            probabilities_output = gr.Markdown(value="")
            explanation_output = gr.Markdown(value="")

        gr.Markdown("**Exemples rapides :**")
        with gr.Row():
            ex1 = gr.Button("✅ Avis positif", size="sm")
            ex2 = gr.Button("❌ Avis négatif", size="sm")
            ex3 = gr.Button("⚖️ Avis neutre", size="sm")
            ex4 = gr.Button("🚫 Spam", size="sm")

        ex1.click(fn=lambda: "Excellentes chaussures de trail, confortables et très durables !",
                  outputs=text_input)
        ex2.click(fn=lambda: "Très déçu, la fermeture éclair a lâché après 2 semaines.",
                  outputs=text_input)
        ex3.click(fn=lambda: "Produit correct, rien d'exceptionnel mais fait le travail.",
                  outputs=text_input)
        ex4.click(fn=lambda: "PROMO INCROYABLE!! Achetez maintenant sur clickici.com !!!!",
                  outputs=text_input)

        predict_btn.click(
            fn=predict_review,
            inputs=[text_input, explain_checkbox],
            outputs=[label_output, confidence_output, probabilities_output, explanation_output],
        )

    # ── Onglet 2 : Batch ─────────────────────────────────────────────────────
    with gr.Tab("📋 Analyser plusieurs avis"):
        gr.Markdown("Entrez un avis par ligne (maximum 20 en démo, 500 via l'API).")
        batch_input = gr.Textbox(
            label="Avis (un par ligne)",
            placeholder="Super produit !\nTrès déçu de la qualité\nSPAM cliquez ici !!!",
            lines=8,
        )
        batch_btn = gr.Button("Analyser tout", variant="primary")
        batch_output = gr.Markdown(value="")

        batch_btn.click(
            fn=predict_batch_demo,
            inputs=batch_input,
            outputs=batch_output,
        )

    # ── Onglet 3 : Infos du modèle ───────────────────────────────────────────
    with gr.Tab("⚙️ Modèle en production"):
        refresh_btn = gr.Button("🔄 Actualiser les infos")
        model_info_output = gr.Markdown(value=get_model_info())

        refresh_btn.click(
            fn=get_model_info,
            inputs=[],
            outputs=model_info_output,
        )

    # ── Pied de page ─────────────────────────────────────────────────────────
    gr.Markdown("""
    ---
    **Stack technique** : Python · Scikit-learn · MLflow · FastAPI · Docker · Kubernetes · GitHub Actions

    **Dataset** : Amazon Sports & Outdoors Reviews (HuggingFace)

    [GitHub](https://github.com/TON_USERNAME/sportreview-ai) | [API Docs](http://localhost:8000/docs)
    """)


# ─────────────────────────────────────────────────────────────────────────────
# DÉMARRAGE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Démarrage de l'interface Gradio...")
    print("Interface accessible sur : http://localhost:7860")
    print("\nPour déployer sur HuggingFace Spaces :")
    print("  1. Crée un Space sur https://huggingface.co/spaces")
    print("  2. Upload ce fichier comme app.py")
    print("  3. Ajoute requirements.txt avec : gradio requests")

    demo.launch(
        server_name="0.0.0.0",   # accessible depuis le réseau local
        server_port=7860,         # port standard Gradio
        share=False,              # True = génère un lien public temporaire
        show_error=True,          # affiche les erreurs dans l'interface
        theme=gr.themes.Soft(),   # thème (Gradio 6.14 : dans launch())
    )
