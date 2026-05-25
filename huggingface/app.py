# -*- coding: utf-8 -*-
"""
app.py — SportReview AI · Interface Gradio (HuggingFace Spaces)

Classificateur d'avis produits sportifs.
Modèle : DistilBERT fine-tuné sur 300 000 avis Amazon.
Backend : FastAPI (déployé séparément) — configurable via API_URL.

Projet MLOps complet :
  Python · MLflow · FastAPI · Gradio · Docker · Kubernetes · GitHub Actions

Auteur : Patricia WELEHELA — Master 2 Machine Learning & Data Science
"""

import os
import requests

try:
    import gradio as gr
except ImportError:
    raise ImportError("Gradio non installé. Lance : pip install gradio")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# URL du backend FastAPI — configurable via variable d'environnement
# Sur HuggingFace Spaces : Settings → Variables → API_URL = https://votre-api.com
API_URL = os.getenv("API_URL", "http://localhost:8000")

LABEL_COLORS = {
    "POSITIF": "#22c55e",
    "NEGATIF": "#ef4444",
    "NEUTRE":  "#f59e0b",
    "SPAM":    "#8b5cf6",
}

LABEL_EMOJIS = {
    "POSITIF": "✅",
    "NEGATIF": "❌",
    "NEUTRE":  "⚖️",
    "SPAM":    "🚫",
}


# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _backend_unavailable_html():
    """HTML affiché quand le backend FastAPI n'est pas accessible."""
    return """
    <div style="border:2px solid #2563eb; border-radius:14px; padding:28px 28px;
                background:linear-gradient(135deg, #eff6ff 0%, #f0f9ff 100%); margin-top:8px;">

      <div style="font-size:1.4rem; font-weight:800; color:#1e3a5f; margin-bottom:16px;">
        🔗 Démo publique — Backend non connecté
      </div>

      <p style="color:#374151; font-size:0.95rem; margin-bottom:16px; line-height:1.6;">
        Cette interface est déployée sur HuggingFace Spaces.<br>
        Le backend FastAPI tourne <strong>en local</strong> pour la démonstration complète.
      </p>

      <div style="background:#1e3a5f; color:#e2e8f0; border-radius:10px;
                  padding:16px 18px; font-family:monospace; font-size:0.85rem;
                  line-height:2; margin-bottom:16px;">
        <span style="color:#60a5fa;"># Lancer le projet en local</span><br>
        uvicorn app.main:app --reload --host 0.0.0.0 --port 8000<br>
        python demo/gradio_app.py
      </div>

      <div style="background:#f0fdf4; border:1px solid #86efac; border-radius:10px;
                  padding:14px 16px; margin-bottom:16px;">
        <p style="margin:0; font-size:0.9rem; color:#166534; font-weight:600;">
          ✅ Ce projet MLOps comprend :
        </p>
        <ul style="margin:8px 0 0 0; padding-left:20px; color:#166534; font-size:0.88rem; line-height:1.8;">
          <li>6 modèles entraînés et comparés via MLflow</li>
          <li>DistilBERT fine-tuné — F1 Weighted : <strong>0.7103</strong></li>
          <li>Pipeline de promotion automatique en production</li>
          <li>API REST FastAPI avec prédiction unitaire et batch</li>
          <li>Docker · Kubernetes · CI/CD GitHub Actions</li>
        </ul>
      </div>

      <p style="margin:0; font-size:0.85rem; color:#6b7280;">
        📧 Démonstration live disponible sur demande :
        <a href="mailto:patriciawelehela@gmail.com"
           style="color:#2563eb; font-weight:600;">patriciawelehela@gmail.com</a>
      </p>
    </div>
    """


def predict_review(text: str, include_explanation: bool = False) -> tuple:
    """
    Appelle le backend FastAPI et formate le résultat en HTML.
    Retourne : (result_html, explanation_html)
    """
    if not text or not text.strip():
        return (
            "<p style='color:#f59e0b; padding:16px; font-size:0.95rem;'>"
            "⚠️ Veuillez saisir un texte avant d'analyser.</p>",
            ""
        )

    try:
        response = requests.post(
            f"{API_URL}/predict",
            params={"explain": include_explanation},
            json={"text": text, "verified_purchase": True},
            timeout=120,
        )

        if response.status_code != 200:
            return (
                f"<p style='color:#ef4444; padding:16px;'>❌ Erreur API {response.status_code}</p>",
                ""
            )

        data          = response.json()
        label         = data["label"]
        confidence    = data["confidence"]
        probabilities = data["probabilities"]
        is_reliable   = data["is_reliable"]
        explanation   = data.get("explanation")

        color   = LABEL_COLORS.get(label, "#6b7280")
        emoji   = LABEL_EMOJIS.get(label, "")
        warning = (
            "" if is_reliable
            else "<span style='color:#f59e0b; font-size:0.82rem; font-weight:500;'>"
                 " ⚠️ confiance faible</span>"
        )

        # Barre de confiance
        conf_pct = confidence * 100
        conf_bar = f"""
        <div style="margin:14px 0 10px 0;">
          <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
            <span style="font-size:0.88rem; color:#64748b; font-weight:500;">Score de confiance</span>
            <span style="font-size:0.92rem; font-weight:700; color:{color};">{conf_pct:.1f}%</span>
          </div>
          <div style="background:#e2e8f0; border-radius:999px; height:10px; overflow:hidden;">
            <div style="background:{color}; width:{conf_pct:.1f}%; height:100%;
                        border-radius:999px; transition:width 0.4s ease;"></div>
          </div>
        </div>"""

        # Barres par classe
        proba_rows = ""
        for cls, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
            c   = LABEL_COLORS.get(cls, "#6b7280")
            em  = LABEL_EMOJIS.get(cls, "")
            pct = prob * 100
            proba_rows += f"""
            <div style="margin:7px 0;">
              <div style="display:flex; justify-content:space-between; margin-bottom:3px;">
                <span style="font-size:0.87rem; color:#374151;">{em} {cls}</span>
                <span style="font-size:0.87rem; font-weight:600; color:{c};">{pct:.1f}%</span>
              </div>
              <div style="background:#f1f5f9; border-radius:999px; height:7px; overflow:hidden;">
                <div style="background:{c}; width:{pct:.1f}%; height:100%; border-radius:999px;"></div>
              </div>
            </div>"""

        result_html = f"""
        <div style="border:2px solid {color}; border-radius:14px; padding:24px 26px;
                    background:linear-gradient(135deg, {color}18, {color}06); margin-top:8px;">
          <div style="font-size:2rem; font-weight:800; color:{color}; margin-bottom:2px;">
            {emoji} {label}{warning}
          </div>
          {conf_bar}
          <hr style="border:none; border-top:1px solid #e2e8f0; margin:16px 0 12px 0;">
          <div style="font-size:0.82rem; color:#94a3b8; font-weight:700;
                      letter-spacing:0.06em; margin-bottom:10px;">
            PROBABILITÉS PAR CLASSE
          </div>
          {proba_rows}
        </div>"""

        # Explication SHAP
        explanation_html = ""
        if explanation and (
            explanation.get("top_positive_words") or explanation.get("top_negative_words")
        ):
            pos_words = explanation.get("top_positive_words", {})
            neg_words = explanation.get("top_negative_words", {})
            explanation_html = """
            <div style="border:1px solid #e2e8f0; border-radius:12px;
                        padding:18px 20px; margin-top:12px; background:#fafafa;">
              <div style="font-size:0.82rem; color:#94a3b8; font-weight:700;
                          letter-spacing:0.06em; margin-bottom:10px;">
                ANALYSE SHAP — MOTS INFLUENTS
              </div>"""
            if pos_words:
                words = " · ".join(
                    [f"<strong style='color:#22c55e;'>{w}</strong>" for w in pos_words.keys()]
                )
                explanation_html += f"<p style='font-size:0.88rem; margin:4px 0;'>✅ Pour {label} : {words}</p>"
            if neg_words:
                words = " · ".join(
                    [f"<strong style='color:#ef4444;'>{w}</strong>" for w in neg_words.keys()]
                )
                explanation_html += f"<p style='font-size:0.88rem; margin:4px 0;'>❌ Contre {label} : {words}</p>"
            explanation_html += "</div>"

        return result_html, explanation_html

    except requests.exceptions.ConnectionError:
        return _backend_unavailable_html(), ""
    except requests.exceptions.Timeout:
        return (
            "<p style='color:#f59e0b; padding:16px;'>"
            "⏱️ Délai dépassé — le modèle DistilBERT peut prendre jusqu'à 2 min au premier appel.</p>",
            ""
        )
    except Exception as e:
        return f"<p style='color:#ef4444; padding:16px;'>❌ Erreur : {str(e)}</p>", ""


def predict_batch_demo(texts: str) -> str:
    """Prédit plusieurs avis (un par ligne), retourne un tableau Markdown."""
    lines = [t.strip() for t in texts.split("\n") if t.strip()]

    if not lines:
        return "⚠️ Aucun texte saisi."
    if len(lines) > 20:
        return "⚠️ Maximum 20 avis en démo (l'API supporte jusqu'à 500)."

    try:
        response = requests.post(
            f"{API_URL}/predict/batch",
            json={"texts": lines},
            timeout=120,
        )

        if response.status_code != 200:
            return f"❌ Erreur API : {response.status_code}"

        data        = response.json()
        predictions = data["predictions"]

        result  = f"**{data['count']} avis analysés en {data['processing_time_ms']:.0f} ms**\n\n"
        result += "| Avis | Label | Confiance |\n|------|-------|-----------|\n"
        for pred in predictions:
            preview = pred["text"][:65] + "…" if len(pred["text"]) > 65 else pred["text"]
            emoji   = LABEL_EMOJIS.get(pred["label"], "")
            result += f"| {preview} | {emoji} {pred['label']} | {pred['confidence']*100:.0f}% |\n"

        return result

    except requests.exceptions.ConnectionError:
        return (
            "🔗 **Backend non connecté en démo publique.**\n\n"
            "La démonstration complète est disponible en local.\n"
            "📧 patriciawelehela@gmail.com"
        )
    except Exception as e:
        return f"❌ Erreur : {str(e)}"


def get_model_info() -> str:
    """Récupère les infos du modèle en production depuis le backend."""
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
    return (
        "**Modèle en production (local)** : DistilBERT fine-tuné\n\n"
        "**F1 Weighted** : 0.7103\n\n"
        "**Classes** : NEGATIF · NEUTRE · POSITIF · SPAM\n\n"
        "**Données** : 300 000 avis Amazon Sport\n\n"
        "*Backend non connecté en démo publique — infos statiques affichées.*"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
.gradio-container { max-width: 1100px !important; margin: auto; }

#header {
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
    border-radius: 14px; padding: 30px 34px; margin-bottom: 22px;
}
#header h1 { color: white !important; font-size: 2rem; margin: 0 0 6px 0; font-weight: 800; }
#header p  { color: #bfdbfe !important; margin: 4px 0 0 0; font-size: 0.95rem; }
#header .stack { color: #93c5fd !important; font-size: 0.82rem; margin-top: 10px !important; }

#analyze-btn { font-size: 1.05rem !important; height: 48px !important; }
#batch-btn   { font-size: 1.00rem !important; }

.example-btn button { font-size: 0.83rem !important; }

footer { display: none !important; }
"""


# ─────────────────────────────────────────────────────────────────────────────
# INTERFACE
# ─────────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="SportReview AI", css=CSS, theme=gr.themes.Soft()) as demo:

    # En-tête
    gr.HTML("""
    <div id="header">
      <h1>🏃 SportReview AI</h1>
      <p>Classificateur d'avis produits sportifs · DistilBERT fine-tuné · 300 000 avis Amazon</p>
      <p class="stack">
        Stack MLOps : Python · Scikit-learn · HuggingFace · MLflow · FastAPI · Docker · Kubernetes · GitHub Actions
      </p>
    </div>
    """)

    # ── Onglet 1 : Prédiction unitaire ───────────────────────────────────────
    with gr.Tab("🔍 Analyser un avis"):
        with gr.Row(equal_height=False):

            with gr.Column(scale=5):
                gr.Markdown("### ✍️ Votre avis")
                text_input = gr.Textbox(
                    label="",
                    placeholder=(
                        "Entrez votre avis produit ici…\n"
                        "Ex : Excellentes chaussures, très confortables !"
                    ),
                    lines=7,
                    max_lines=16,
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

            with gr.Column(scale=5):
                gr.Markdown("### 📊 Résultat")
                result_output = gr.HTML(
                    value=(
                        "<div style='padding:28px; border:2px dashed #e2e8f0; border-radius:14px;"
                        " color:#94a3b8; text-align:center; font-size:0.95rem;'>"
                        "🔍 Le résultat apparaîtra ici après analyse</div>"
                    )
                )
                explanation_output = gr.HTML(value="")

        gr.Markdown("---\n**🧪 Exemples — cliquez pour pré-remplir :**")
        with gr.Row(elem_classes="example-btn"):
            ex1 = gr.Button("✅ Positif",   size="sm")
            ex2 = gr.Button("❌ Négatif",   size="sm")
            ex3 = gr.Button("⚖️ Neutre",    size="sm")
            ex4 = gr.Button("🚫 Spam",      size="sm")
            ex5 = gr.Button("🇬🇧 English",  size="sm")

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
        Entrez **un avis par ligne** puis cliquez sur Analyser.
        Maximum 20 avis en démo — l'API supporte jusqu'à 500.
        """)
        batch_input = gr.Textbox(
            label="📝 Avis (un par ligne)",
            placeholder=(
                "Super produit !\n"
                "Très déçu de la qualité\n"
                "SPAM cliquez ici !!!"
            ),
            lines=10,
        )
        batch_btn = gr.Button(
            "📋 Analyser tous les avis",
            variant="primary",
            elem_id="batch-btn",
        )
        batch_output = gr.Markdown(value="*Les résultats apparaîtront ici…*")

        batch_btn.click(
            fn=predict_batch_demo,
            inputs=batch_input,
            outputs=batch_output,
            show_progress="full",
        )

    # ── Onglet 3 : Modèle en production ──────────────────────────────────────
    with gr.Tab("⚙️ Modèle en production"):
        gr.Markdown("### Informations sur le modèle actuellement en production")
        with gr.Row():
            refresh_btn = gr.Button("🔄 Actualiser", variant="secondary")
        model_info_output = gr.Markdown(value=get_model_info())

        refresh_btn.click(
            fn=get_model_info,
            inputs=[],
            outputs=model_info_output,
            show_progress="full",
        )

    # Pied de page
    gr.HTML("""
    <div style="text-align:center; margin-top:24px; padding-top:16px;
                border-top:1px solid #e2e8f0; color:#94a3b8; font-size:0.83rem;">
      <strong style="color:#374151;">SportReview AI</strong> ·
      Patricia WELEHELA · Master 2 Machine Learning &amp; Data Science ·
      <a href="http://localhost:8000/docs" target="_blank"
         style="color:#2563eb; text-decoration:none;">📄 API Docs</a>
    </div>
    """)


# ─────────────────────────────────────────────────────────────────────────────
# DÉMARRAGE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  SportReview AI — Interface Gradio")
    print("=" * 55)
    print(f"  Backend API : {API_URL}")
    print("  Interface   : http://localhost:7860")
    print("=" * 55)

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
