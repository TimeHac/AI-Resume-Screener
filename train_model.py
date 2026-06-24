"""
train_model.py  –  Train and save the resume classification model.
Run: python train_model.py
"""

import os
import sys
import pickle
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ── Text cleaning ──────────────────────────────────────────────────────────

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s\+\#]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ── Load / generate dataset ────────────────────────────────────────────────

def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), "resumes.csv")
    if not os.path.exists(csv_path):
        print("Generating dataset...")
        os.chdir(os.path.dirname(__file__))
        from data.generate_dataset import generate_dataset
        df = generate_dataset(500)
    else:
        df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} resumes.")
    print(df["category"].value_counts())
    return df

# ── Train ──────────────────────────────────────────────────────────────────

def train(df):
    df["clean"] = df["resume"].apply(clean_text)

    X = df["clean"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline: TF-IDF → SVM (best accuracy) and LR (interpretable)
    pipelines = {
        "SVM": Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=15000,
                sublinear_tf=True,
                min_df=2,
            )),
            ("clf", SVC(kernel="linear", C=1.0, probability=True, random_state=42)),
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=15000,
                sublinear_tf=True,
                min_df=2,
            )),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ]),
    }

    best_pipeline = None
    best_score = 0
    best_name = ""

    print("\n── Model Evaluation ──────────────────────────────")
    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"\n{name}  →  Test Accuracy: {acc*100:.2f}%")
        print(classification_report(y_test, preds))
        if acc > best_score:
            best_score = acc
            best_pipeline = pipe
            best_name = name

    print(f"\n✓ Best model: {best_name} ({best_score*100:.2f}%)")

    # Cross-validation on best
    cv_scores = cross_val_score(best_pipeline, X, y, cv=5, scoring="accuracy")
    print(f"  5-Fold CV: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    return best_pipeline, best_name, best_score

# ── Save ───────────────────────────────────────────────────────────────────

def save_model(pipeline, model_dir="model"):
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "resume_classifier.pkl")
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\n✓ Model saved to {path}")

# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Generate dataset first
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    csv_path = os.path.join(data_dir, "resumes.csv")

    if not os.path.exists(csv_path):
        print("Generating synthetic training data...")
        original_dir = os.getcwd()
        os.chdir(data_dir)
        import subprocess
        subprocess.run([sys.executable, "generate_dataset.py"], check=True)
        os.chdir(original_dir)

    df = pd.read_csv(csv_path)
    df["clean"] = df["resume"].apply(clean_text)

    X = df["clean"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipelines = {
        "SVM": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=15000, sublinear_tf=True, min_df=2)),
            ("clf", SVC(kernel="linear", C=1.0, probability=True, random_state=42)),
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=15000, sublinear_tf=True, min_df=2)),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
        ]),
    }

    best_pipeline = None
    best_score = 0
    best_name = ""

    print("\n── Model Evaluation ──────────────────────────────")
    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"\n{name}  →  Test Accuracy: {acc*100:.2f}%")
        print(classification_report(y_test, preds))
        if acc > best_score:
            best_score = acc
            best_pipeline = pipe
            best_name = name

    print(f"\n✓ Best model: {best_name} ({best_score*100:.2f}%)")
    cv_scores = cross_val_score(best_pipeline, X, y, cv=5, scoring="accuracy")
    print(f"  5-Fold CV: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    save_model(best_pipeline)
    print("\nTraining complete. Run `python app.py` to start the Flask server.")
