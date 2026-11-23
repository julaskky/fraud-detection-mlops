
from sklearn.linear_model import LogisticRegression

from src.models.common import (
    load_prepared_data,
    save_validation_result,
)


def train():
    data = load_prepared_data()

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        solver="liblinear",
        random_state=42,
    )

    print("Training Logistic Regression...")

    model.fit(
        data["X_train"],
        data["y_train"],
    )

    save_validation_result(
        "logistic_regression",
        model,
        data["X_validation"],
        data["y_validation"],
    )


if __name__ == "__main__":
    train()
