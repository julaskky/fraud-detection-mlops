
from sklearn.ensemble import RandomForestClassifier

from src.models.common import (
    load_prepared_data,
    save_validation_result,
)


def train():
    data = load_prepared_data()

    model = RandomForestClassifier(
        n_estimators=400,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    print("Training Random Forest...")

    model.fit(
        data["X_train"],
        data["y_train"],
    )

    save_validation_result(
        "random_forest",
        model,
        data["X_validation"],
        data["y_validation"],
    )


if __name__ == "__main__":
    train()
