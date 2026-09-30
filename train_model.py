"""Train the emotion classifiers and save the best one.

Usage:
    python train_model.py --data train.txt
train.txt format: one sample per line ->  "some text;emotion"
"""
import argparse
import json
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from text_preprocessing import clean_text

MODEL_DIR = "models"


def load_data(path_or_buffer) -> pd.DataFrame:
    df = pd.read_csv(path_or_buffer, sep=";", header=None, names=["text", "emotion"])
    return df.dropna().reset_index(drop=True)


def train(df: pd.DataFrame, model_dir: str = MODEL_DIR, progress=None) -> dict:
    """Train 3 models (as in the notebook), save the best pipeline + metrics."""
    log = progress or (lambda msg: None)
    os.makedirs(model_dir, exist_ok=True)

    log("Cleaning text...")
    df = df.copy()
    df["clean"] = df["text"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["emotion"], test_size=0.20, random_state=42
    )

    candidates = {}

    log("Training Naive Bayes + Bag of Words...")
    bow = CountVectorizer()
    Xtr = bow.fit_transform(X_train)
    nb = MultinomialNB().fit(Xtr, y_train)
    candidates["Naive Bayes (BoW)"] = (bow, nb)

    log("Training Naive Bayes + TF-IDF...")
    tfidf_nb = TfidfVectorizer()
    Xtr = tfidf_nb.fit_transform(X_train)
    nb2 = MultinomialNB().fit(Xtr, y_train)
    candidates["Naive Bayes (TF-IDF)"] = (tfidf_nb, nb2)

    log("Training Logistic Regression + TF-IDF...")
    tfidf_lr = TfidfVectorizer()
    Xtr = tfidf_lr.fit_transform(X_train)
    lr = LogisticRegression(max_iter=1000).fit(Xtr, y_train)
    candidates["Logistic Regression (TF-IDF)"] = (tfidf_lr, lr)

    results, best_name, best_acc = {}, None, -1
    for name, (vec, model) in candidates.items():
        pred = model.predict(vec.transform(X_test))
        acc = accuracy_score(y_test, pred)
        results[name] = acc
        if acc > best_acc:
            best_name, best_acc = name, acc

    vec, model = candidates[best_name]
    pred = model.predict(vec.transform(X_test))
    labels = sorted(df["emotion"].unique())

    joblib.dump(vec, os.path.join(model_dir, "vectorizer.joblib"))
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))

    metrics = {
        "best_model": best_name,
        "accuracies": results,
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_test, pred, labels=labels).tolist(),
        "report": classification_report(y_test, pred, output_dict=True, zero_division=0),
        "class_counts": df["emotion"].value_counts().to_dict(),
        "n_samples": int(len(df)),
    }
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    log(f"Done. Best model: {best_name} ({best_acc:.4f})")
    return metrics


def model_exists(model_dir: str = MODEL_DIR) -> bool:
    return all(
        os.path.exists(os.path.join(model_dir, f))
        for f in ("vectorizer.joblib", "model.joblib", "metrics.json")
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="train.txt")
    ap.add_argument("--out", default=MODEL_DIR)
    args = ap.parse_args()
    m = train(load_data(args.data), args.out, progress=print)
    for k, v in m["accuracies"].items():
        print(f"{k:32s} {v:.4f}")
