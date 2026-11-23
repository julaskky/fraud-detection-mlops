
import numpy as np
import torch

from pytorch_tabnet.tab_model import TabNetClassifier

from src.models.common import (
    load_prepared_data,
    save_validation_result,
    to_dense_float32,
)


def train():
    data = load_prepared_data()

    X_train = to_dense_float32(
        data["X_train"]
    )

    X_validation = to_dense_float32(
        data["X_validation"]
    )

    y_train = np.asarray(
        data["y_train"],
        dtype=np.int64,
    )

    y_validation = np.asarray(
        data["y_validation"],
        dtype=np.int64,
    )

    torch.manual_seed(42)
    np.random.seed(42)

    model = TabNetClassifier(
        n_d=16,
        n_a=16,
        n_steps=4,
        gamma=1.5,
        lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params={
            "lr": 0.02,
        },
        mask_type="entmax",
        seed=42,
        verbose=1,
        device_name="cpu",
    )

    print("Training TabNet...")
    print("Training records:", len(X_train))
    print("Input features:", X_train.shape[1])

    model.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[
            (
                X_validation,
                y_validation,
            )
        ],
        eval_name=["validation"],
        eval_metric=["auc"],
        max_epochs=100,
        patience=15,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
        weights=1,
    )

    save_validation_result(
        "tabnet",
        model,
        X_validation,
        y_validation,
    )


if __name__ == "__main__":
    train()
