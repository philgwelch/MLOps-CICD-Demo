from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

OUTPUT_PATH = Path("outputs/")


def load_and_clean_data() -> pd.DataFrame:
    """Loads penguins dataset and cleans it."""
    df = sns.load_dataset("penguins")
    df.dropna(inplace=True)
    df = pd.get_dummies(df, columns=["island", "sex"], drop_first=True)
    return df


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """Trains the Random Forest model."""
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf


def write_metrics(path: Path, **kwargs) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "metrics.txt", "w") as f:
        f.write(f"Accuracy: {kwargs['accuracy']:.2f}\n")


def save_confustion_matrix(path: Path, cm: NDArray, labels) -> None:
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    plt.figure(figsize=(8, 6))
    disp.plot(cmap="Blues")
    plt.savefig(path / "confusion_matrix.png", dpi=120)


def main() -> None:
    # 1. Prepare Data
    print("Loading data...")
    df = load_and_clean_data()

    X = df.drop(columns=["species"])
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 2. Train
    print("Training model...")
    clf = train_model(X_train, y_train)

    # 3. Evaluate & Save (same as before...)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    write_metrics(OUTPUT_PATH, accuracy=accuracy)

    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    save_confustion_matrix(OUTPUT_PATH, cm, clf.classes_)


if __name__ == "__main__":
    main()
