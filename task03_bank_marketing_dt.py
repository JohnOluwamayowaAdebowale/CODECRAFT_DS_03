import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

DATA_PATH = "bank.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_bank_data(path):
    # UCI CSV is usually semicolon-separated
    df = pd.read_csv(path, sep=";")
    # Convert target
    if "y" in df.columns:
        df["y"] = (df["y"].astype(str).str.lower().str.strip() == "yes").astype(int)
    return df

def preprocess(df):
    df = df.copy()
    # One-hot encode categoricals
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if "y" in cat_cols:
        cat_cols.remove("y")
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    # Drop rows with any remaining NA
    df = df.dropna()
    return df

def plot_feature_importance(importances, feature_names, fname):
    idx = np.argsort(importances)[::-1][:20]
    plt.figure()
    plt.barh(np.array(feature_names)[idx][::-1], importances[idx][::-1])
    plt.title("Top 20 Feature Importances (Decision Tree)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, fname))
    plt.close()

def main():
    df = load_bank_data(DATA_PATH)
    df_prep = preprocess(df)
    X = df_prep.drop(columns=["y"])
    y = df_prep["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = DecisionTreeClassifier(
        max_depth=6, min_samples_split=20, min_samples_leaf=10, random_state=42
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # Save metrics
    cr = classification_report(y_test, y_pred, digits=3)
    cm = confusion_matrix(y_test, y_pred)

    with open(os.path.join(OUTPUT_DIR, "metrics.txt"), "w") as f:
        f.write("Classification Report\n")
        f.write(cr + "\n\n")
        f.write("Confusion Matrix\n")
        f.write(str(cm) + "\n")

    # Feature importance plot
    plot_feature_importance(clf.feature_importances_, X.columns, "feature_importance.png")

    print("Training complete. Metrics saved to outputs/metrics.txt")

if __name__ == "__main__":
    main()
