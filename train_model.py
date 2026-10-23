
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ids_model.joblib"
DATA_PATH = BASE_DIR / "demo_network_traffic.csv"

FEATURES = [
    "duration",
    "packet_count",
    "bytes_per_second",
    "destination_port",
    "failed_connections",
]

def generate_training_data(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    rows = []

    # Normal traffic
    for _ in range(n // 2):
        rows.append([
            rng.exponential(8),
            rng.integers(5, 250),
            rng.uniform(100, 150000),
            rng.choice([80, 443, 53, 22, 123, 8080]),
            rng.choice([0, 0, 0, 0, 1]),
            "Normal",
        ])

    # Simulated suspicious traffic patterns
    for _ in range(n // 2):
        attack_type = rng.choice(["DoS-like", "Brute-force-like", "Scan-like"])

        if attack_type == "DoS-like":
            row = [
                rng.uniform(0.01, 2),
                rng.integers(500, 5000),
                rng.uniform(500000, 5000000),
                rng.choice([80, 443, 8080]),
                rng.integers(0, 4),
                "Suspicious",
            ]
        elif attack_type == "Brute-force-like":
            row = [
                rng.uniform(0.1, 5),
                rng.integers(2, 40),
                rng.uniform(100, 50000),
                rng.choice([21, 22, 3389]),
                rng.integers(5, 30),
                "Suspicious",
            ]
        else:
            row = [
                rng.uniform(0.01, 1),
                rng.integers(1, 20),
                rng.uniform(50, 20000),
                rng.integers(1, 65536),
                rng.integers(0, 3),
                "Suspicious",
            ]

        rows.append(row)

    columns = FEATURES + ["label"]
    data = pd.DataFrame(rows, columns=columns)
    return data.sample(frac=1, random_state=seed).reset_index(drop=True)


def main():
    data = generate_training_data()
    X = data[FEATURES]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    joblib.dump({"model": model, "features": FEATURES}, MODEL_PATH)
    data.to_csv(DATA_PATH, index=False)

    print("Model training completed!")
    print(f"Model saved: {MODEL_PATH.name}")
    print(f"Demo data saved: {DATA_PATH.name}")
    print(f"Test accuracy on simulated data: {accuracy:.2%}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))


if __name__ == "__main__":
    main()
