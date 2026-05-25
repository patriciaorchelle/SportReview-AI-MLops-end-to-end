"""
distilbert_wrapper.py
=====================
Wrapper qui fait ressembler DistilBERT à un pipeline sklearn.

Pourquoi ce wrapper ?
→ FastAPI charge les modèles avec pickle.load() et appelle .predict()
→ Les pipelines sklearn ont cette interface nativement
→ DistilBERT (HuggingFace) n'a pas cette interface
→ Ce wrapper implémente .predict() et .predict_proba() pour DistilBERT
→ Le wrapper est picklable car il stocke seulement le chemin du modèle
→ Le modèle HuggingFace est chargé en mémoire au premier appel (lazy loading)

C'est un pattern standard en MLOps appelé "Model Wrapper" ou "Adapter Pattern".

Usage :
    wrapper = DistilBertWrapper(model_dir="models/distilbert_finetuned")
    wrapper.predict(["Super produit !", "Nul ce truc"])
    # → ["POSITIF", "NEGATIF"]
"""

import os
import numpy as np


CLASSES = ["NEGATIF", "NEUTRE", "POSITIF", "SPAM"]


class DistilBertWrapper:
    """
    Wrapper sklearn-compatible pour DistilBERT fine-tuné.

    Implémente l'interface sklearn (predict / predict_proba) pour pouvoir
    être utilisé dans FastAPI exactement comme un Pipeline sklearn.

    Le modèle HuggingFace est chargé lazily (au premier appel) pour éviter
    de bloquer le démarrage de l'API si torch n'est pas disponible.
    """

    def __init__(self, model_dir: str, max_length: int = 128, batch_size: int = 32):
        """
        Args:
            model_dir  : chemin vers le dossier du modèle DistilBERT sauvegardé
                         (contient config.json, pytorch_model.bin, tokenizer files...)
            max_length : longueur maximale des séquences en tokens
            batch_size : taille des batches pour l'inférence
        """
        self.model_dir  = model_dir
        self.max_length = max_length
        self.batch_size = batch_size
        self.classes_   = CLASSES  # attribut sklearn standard

        # Le modèle et le tokenizer sont chargés lazily
        self._model     = None
        self._tokenizer = None
        self._device    = None

    def _load_model(self):
        """Charge le modèle HuggingFace en mémoire (une seule fois)."""
        if self._model is not None:
            return  # déjà chargé

        import os
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        try:
            import torch
        except ImportError:
            raise ImportError("torch est requis pour DistilBERT. Lance : pip install torch transformers")

        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

        if not os.path.exists(self.model_dir):
            # Fallback : le pickle a été créé sur Windows avec un chemin absolu,
            # mais on tourne sous Linux/Docker → on cherche dans les chemins standards
            fallbacks = [
                "/app/models/distilbert_finetuned",
                os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "models", "distilbert_finetuned"),
            ]
            resolved = next((p for p in fallbacks if os.path.exists(p)), None)
            if resolved:
                self.model_dir = os.path.abspath(resolved)
            else:
                raise FileNotFoundError(
                    f"Modèle DistilBERT introuvable : {self.model_dir}\n"
                    "Lance d'abord : python src/train.py --models distilbert"
                )

        self._tokenizer = DistilBertTokenizer.from_pretrained(self.model_dir)
        self._model     = DistilBertForSequenceClassification.from_pretrained(self.model_dir)
        self._device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model     = self._model.to(self._device)
        self._model.eval()

        print(f"DistilBERT chargé depuis {self.model_dir} sur {self._device}")

    def predict_proba(self, texts):
        """
        Retourne les probabilités pour chaque classe.

        Args:
            texts : liste de strings (les avis à classer)

        Returns:
            np.array de shape (n_texts, 4) — probabilités pour chaque classe
        """
        import torch
        from torch.utils.data import DataLoader, Dataset

        self._load_model()

        class SimpleDataset(Dataset):
            def __init__(self, encodings):
                self.encodings = encodings
            def __len__(self):
                return self.encodings["input_ids"].shape[0]
            def __getitem__(self, idx):
                return {k: v[idx] for k, v in self.encodings.items()}

        # Tokenisation
        encodings = self._tokenizer(
            list(texts),
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        dataset = SimpleDataset(encodings)
        loader  = DataLoader(dataset, batch_size=self.batch_size)

        all_probs = []
        with torch.no_grad():
            for batch in loader:
                input_ids      = batch["input_ids"].to(self._device)
                attention_mask = batch["attention_mask"].to(self._device)
                outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
                probs   = torch.softmax(outputs.logits, dim=1).cpu().numpy()
                all_probs.append(probs)

        return np.vstack(all_probs)

    def predict(self, texts):
        """
        Retourne les labels prédits.

        Args:
            texts : liste de strings

        Returns:
            np.array de strings (labels : "NEGATIF", "NEUTRE", "POSITIF", "SPAM")
        """
        probs   = self.predict_proba(texts)
        indices = np.argmax(probs, axis=1)
        return np.array([CLASSES[i] for i in indices])

    def __getstate__(self):
        """Pickle : on ne sérialise PAS le modèle torch (trop lourd, non portable)."""
        state = self.__dict__.copy()
        state["_model"]     = None
        state["_tokenizer"] = None
        state["_device"]    = None
        return state

    def __setstate__(self, state):
        """Unpickle : recharge le modèle lazily au premier appel."""
        self.__dict__.update(state)
        self._model     = None
        self._tokenizer = None
        self._device    = None

    def __repr__(self):
        return (f"DistilBertWrapper(model_dir='{self.model_dir}', "
                f"max_length={self.max_length}, batch_size={self.batch_size})")
