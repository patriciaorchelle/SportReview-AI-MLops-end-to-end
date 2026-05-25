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

def predict_review(text: str, include_explanation: bool = False) -> tuple:
    """
    Appelle l'API FastAPI et formate le résultat en HTML riche pour Gradio.
    Retourne : (result_html, explanation_html)
    """
    if not text or not text.strip():
        return "<p style='color:#f59e0b; padding:16px;'>⚠️ Veuillez saisir un texte.</p>", ""

    try:
        response = requests.post(
            f"{API_URL}/predict",
            params={"explain": include_explanation},
            json={"text": text, "verified_purchase": True},
            timeout=120
        )

        if response.status_code != 200:
            return f"<p style='color:#ef4444; padding:16px;'>❌ Erreur API : {response.status_code}</p>", ""

        data        = response.json()
        label       = data["label"]
        confidence  = data["confidence"]
        probabilities = data["probabilities"]
        is_reliable = data["is_reliable"]
        explanation = data.get("explanation")

        color   = LABEL_COLORS.get(label, "#6b7280")
        emoji   = LABEL_EMOJIS.get(label, "")
        warning = "" if is_reliable else "<span style='color:#f59e0b; font-size:0.85rem;'> ⚠️ confiance faible</span>"

        # ── Barre de confiance HTML ───────────────────────────────────────
        conf_pct = confidence * 100
        conf_bar = f"""
        <div style="margin: 12px 0;">
          <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
            <span style="font-size:0.9rem; color:#64748b; font-weight:500;">Score de confiance</span>
            <span style="font-size:0.95rem; font-weight:700; color:{color};">{conf_pct:.1f}%</span>
          </div>
          <div style="background:#e2e8f0; border-radius:999px; height:10px; overflow:hidden;">
            <div style="background:{color}; width:{conf_pct:.1f}%; height:100%; border-radius:999px;"></div>
          </div>
        </div>"""

        # ── Barres de probabilités HTML ───────────────────────────────────
        proba_rows = ""
        for cls, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            c     = LABEL_COLORS.get(cls, "#6b7280")
            em    = LABEL_EMOJIS.get(cls, "")
            pct   = prob * 100
            proba_rows += f"""
            <div style="margin:6px 0;">
              <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
                <span style="font-size:0.88rem;">{em} {cls}</span>
                <span style="font-size:0.88rem; font-weight:600; color:{c};">{pct:.1f}%</span>
              </div>
              <div style="background:#f1f5f9; border-radius:999px; height:8px; overflow:hidden;">
                <div style="background:{c}; width:{pct:.1f}%; height:100%; border-radius:999px;"></div>
              </div>
            </div>"""

        result_html = f"""
        <div style="border:2px solid {color}; border-radius:14px; padding:22px 24px;
                    background:linear-gradient(135deg, {color}10, {color}05); margin-top:8px;">
          <div style="font-size:2rem; font-weight:800; color:{color}; margin-bottom:4px;">
            {emoji} {label}{warning}
          </div>
          {conf_bar}
          <hr style="border:none; border-top:1px solid #e2e8f0; margin:14px 0;">
          <div style="font-size:0.9rem; color:#64748b; font-weight:600; margin-bottom:8px;">
            PROBABILITÉS PAR CLASSE
          </div>
          {proba_rows}
        </div>"""

        # ── Explication SHAP ──────────────────────────────────────────────
        explanation_html = ""
        if explanation and (explanation.get("top_positive_words") or explanation.get("top_negative_words")):
            pos_words = explanation.get("top_positive_words", {})
            neg_words = explanation.get("top_negative_words", {})
            if pos_words:
                words = ", ".join([f"<strong>{w}</strong>" for w in pos_words.keys()])
                explanation_html += f"<p>✅ <em>Mots en faveur de {label} :</em> {words}</p>"
            if neg_words:
                words = ", ".join([f"<strong>{w}</strong>" for w in neg_words.keys()])
                explanation_html += f"<p>❌ <em>Mots contre {label} :</em> {words}</p>"

        return result_html, explanation_html

    except requests.exceptions.ConnectionError:
        return "<p style='color:#ef4444; padding:16px;'>❌ API non accessible — lance : <code>uvicorn app.main:app --reload</code></p>", ""
    except Exception as e:
        return f"<p style='color:#ef4444; padding:16px;'>❌ Erreur : {str(e)}</p>", ""


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

CSS = """
.gradio-container { max-width: 1100px !important; margin: auto; }

#header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
    border-radius: 12px; padding: 28px 32px; margin-bottom: 20px;
    color: white;
}
#header h1 { color: white !important; font-size: 2rem; margin: 0 0 6px 0; }
#header p  { color: #bfdbfe !important; margin: 0; font-size: 1rem; }

.result-box {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 16px; min-height: 60px;
}
.result-label { font-size: 1.4rem; font-weight: 700; }

#analyze-btn { font-size: 1.1rem !important; height: 50px !important; }
#batch-btn   { font-size: 1.05rem !important; }

.example-btn { font-size: 0.85rem !important; }

footer { display: none !important; }
"""

with gr.Blocks(title="SportReview AI", css=CSS) as demo:

    # ── En-tête ──────────────────────────────────────────────────────────────
    gr.HTML("""
    <div id="header">
      <h1>🏃 SportReview AI</h1>
      <p>Classificateur d'avis produits sportifs · DistilBERT fine-tuné · 300 000 avis Amazon</p>
      <p style="margin-top:8px; font-size:0.85rem; color:#93c5fd;">
        Stack : Python · MLflow · FastAPI · Docker · Kubernetes · GitHub Actions
      </p>
    </div>
    """)

    # ── Onglet 1 : Prédiction unitaire ───────────────────────────────────────
    with gr.Tab("🔍 Analyser un avis"):

        with gr.Row(equal_height=False):
            # ── Colonne gauche : saisie ──────────────────────────────────
            with gr.Column(scale=3):
                gr.Markdown("### ✍️ Votre avis")
                text_input = gr.Textbox(
                    label="",
                    placeholder="Entrez votre avis produit ici...\nEx : Excellentes chaussures, très confortables !",
                    lines=6,
                    max_lines=14,
                )
                explain_checkbox = gr.Checkbox(
                    label="Inclure l'explication SHAP (mots influents)",
                    value=False,
                )
                predict_btn = gr.Button(
                    "🔍 Analyser l'avis",
                    variant="primary",
                    size="lg",
                    elem_id="analyze-btn",
                )

            # ── Colonne droite : résultat ────────────────────────────────
            with gr.Column(scale=3):
                gr.Markdown("### 📊 Résultat de l'analyse")
                result_output = gr.HTML(
                    value="<div style='padding:24px; border:2px dashed #e2e8f0; border-radius:14px; "
                          "color:#94a3b8; text-align:center; font-size:1rem;'>"
                          "🔍 Le résultat apparaîtra ici après analyse</div>"
                )
                explanation_output = gr.HTML(value="")

        gr.Markdown("---\n**🧪 Exemples — cliquez pour pré-remplir puis analysez :**")
        with gr.Row():
            ex1 = gr.Button("✅ Positif",  size="sm", elem_classes="example-btn")
            ex2 = gr.Button("❌ Négatif",  size="sm", elem_classes="example-btn")
            ex3 = gr.Button("⚖️ Neutre",   size="sm", elem_classes="example-btn")
            ex4 = gr.Button("🚫 Spam",     size="sm", elem_classes="example-btn")
            ex5 = gr.Button("🇬🇧 English", size="sm", elem_classes="example-btn")

        ex1.click(fn=lambda: "Excellentes chaussures de trail, confortables et très durables ! Je recommande vivement.", outputs=text_input)
        ex2.click(fn=lambda: "Très déçu, la fermeture éclair a lâché après 2 semaines. Qualité médiocre.", outputs=text_input)
        ex3.click(fn=lambda: "Produit correct, rien d'exceptionnel mais fait le travail au quotidien.", outputs=text_input)
        ex4.click(fn=lambda: "PROMO INCROYABLE!! Achetez maintenant sur notre site clickici.com !!!!!", outputs=text_input)
        ex5.click(fn=lambda: "The running shoes are amazing, best purchase ever! Highly recommend.", outputs=text_input)

        predict_btn.click(
            fn=predict_review,
            inputs=[text_input, explain_checkbox],
            outputs=[result_output, explanation_output],
            show_progress="full",
        )

    # ── Onglet 2 : Batch ─────────────────────────────────────────────────────
    with gr.Tab("📋 Analyser plusieurs avis"):
        gr.Markdown("""
        Entrez **un avis par ligne** et cliquez sur Analyser.
        Maximum 20 avis en démo — l'API supporte jusqu'à 500.
        """)
        batch_input = gr.Textbox(
            label="📝 Avis (un par ligne)",
            placeholder="Super produit !\nTrès déçu de la qualité\nSPAM cliquez ici !!!",
            lines=10,
        )
        batch_btn = gr.Button("📋 Analyser tous les avis", variant="primary", elem_id="batch-btn")
        batch_output = gr.Markdown(value="*Les résultats apparaîtront ici...*")

        batch_btn.click(
            fn=predict_batch_demo,
            inputs=batch_input,
            outputs=batch_output,
            show_progress="full",
        )

    # ── Onglet 3 : Infos du modèle ───────────────────────────────────────────
    with gr.Tab("⚙️ Modèle en production"):
        gr.Markdown("### Informations sur le modèle actuellement en production")
        with gr.Row():
            refresh_btn = gr.Button("🔄 Actualiser", variant="secondary")
        model_info_output = gr.Markdown(value=get_model_info(), elem_classes="result-box")

        refresh_btn.click(
            fn=get_model_info,
            inputs=[],
            outputs=model_info_output,
            show_progress="full",
        )

    # ── Pied de page ─────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="text-align:center; margin-top:20px; color:#64748b; font-size:0.85rem; border-top:1px solid #e2e8f0; padding-top:16px;">
      <strong>SportReview AI</strong> · Patricia WELEHELA · Master 2 Machine Learning &amp; Data Science ·
      <a href="http://localhost:8000/docs" target="_blank" style="color:#2563eb;">API Docs</a>
    </div>
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
