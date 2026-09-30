"""Load the saved model and predict emotions."""
import json
import os

import joblib
import numpy as np

from text_preprocessing import clean_text
from train_model import MODEL_DIR


class EmotionPredictor:
    def __init__(self, model_dir: str = MODEL_DIR):
        self.vectorizer = joblib.load(os.path.join(model_dir, "vectorizer.joblib"))
        self.model = joblib.load(os.path.join(model_dir, "model.joblib"))
        with open(os.path.join(model_dir, "metrics.json")) as f:
            self.metrics = json.load(f)

    def predict(self, text: str):
        """Return (label, {label: probability})."""
        X = self.vectorizer.transform([clean_text(text)])
        probs = self.model.predict_proba(X)[0]
        classes = [str(c) for c in self.model.classes_]
        scores = {c: float(p) for c, p in zip(classes, probs)}
        return classes[int(np.argmax(probs))], scores
