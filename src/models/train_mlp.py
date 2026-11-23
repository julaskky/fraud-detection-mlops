
from src.models.common import (
    load_prepared_data,
    save_validation_result,
)
from src.models.mlp import TorchMLPClassifier


def train():
    data = load_prepared_data()

    model = TorchMLPClassifier(
        random_state=42,
        epochs=60,
        batch_size=512,
        learning_rate=0.001,
        patience=10,
    )

    print("Training MLP...")

    model.fit(
        data["X_train"],
        data["y_train"],
        data["X_validation"],
        data["y_validation"],
    )

    save_validation_result(
        "mlp",
        model,
        data["X_validation"],
        data["y_validation"],
    )


if __name__ == "__main__":
    train()
