
import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.common import to_dense_float32


class FraudMLP(nn.Module):
    """Neural network for binary fraud classification."""

    def __init__(self, input_features):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(64, 1),
        )

    def forward(self, inputs):
        return self.network(inputs).squeeze(1)


class TorchMLPClassifier:
    """Trainable PyTorch classifier with a scikit-learn-style API."""

    def __init__(
        self,
        random_state=42,
        epochs=60,
        batch_size=512,
        learning_rate=0.001,
        patience=10,
    ):
        self.random_state = random_state
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience

        self.device = torch.device("cpu")
        self.model = None
        self.input_features = None
        self.best_epoch = 0

    def fit(
        self,
        X_train,
        y_train,
        X_validation,
        y_validation,
    ):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_train = to_dense_float32(X_train)
        X_validation = to_dense_float32(X_validation)

        y_train = np.asarray(y_train, dtype=np.float32)
        y_validation = np.asarray(
            y_validation,
            dtype=np.float32,
        )

        self.input_features = X_train.shape[1]

        self.model = FraudMLP(
            self.input_features
        ).to(self.device)

        positive_count = np.sum(y_train == 1)
        negative_count = np.sum(y_train == 0)

        positive_weight = (
            negative_count / max(positive_count, 1)
        )

        pos_weight = torch.tensor(
            [positive_weight],
            dtype=torch.float32,
        )

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,
        )

        train_dataset = TensorDataset(
            torch.tensor(X_train),
            torch.tensor(y_train),
        )

        generator = torch.Generator()
        generator.manual_seed(self.random_state)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )

        X_val_tensor = torch.tensor(X_validation)
        y_val_tensor = torch.tensor(y_validation)

        best_loss = float("inf")
        best_state = None
        epochs_without_improvement = 0

        print("Training PyTorch MLP...")
        print("Device:", self.device)
        print("Input features:", self.input_features)

        for epoch in range(self.epochs):
            self.model.train()

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()

                logits = self.model(batch_X)

                loss = criterion(
                    logits,
                    batch_y,
                )

                loss.backward()
                optimizer.step()

            self.model.eval()

            with torch.no_grad():
                validation_logits = self.model(
                    X_val_tensor
                )

                validation_loss = criterion(
                    validation_logits,
                    y_val_tensor,
                ).item()

            print(
                f"Epoch {epoch + 1}/{self.epochs} "
                f"- Validation loss: {validation_loss:.5f}"
            )

            if validation_loss < best_loss:
                best_loss = validation_loss
                best_state = copy.deepcopy(
                    self.model.state_dict()
                )
                self.best_epoch = epoch + 1
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.patience:
                print("Early stopping activated.")
                break

        if best_state is None:
            raise RuntimeError(
                "MLP training did not produce a valid model."
            )

        self.model.load_state_dict(best_state)
        self.model.eval()

        print("Best epoch:", self.best_epoch)

        return self

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError(
                "Train the MLP before making predictions."
            )

        X = to_dense_float32(X)

        self.model.eval()

        with torch.no_grad():
            logits = self.model(
                torch.tensor(X)
            )

            probabilities = torch.sigmoid(
                logits
            ).numpy()

        return np.column_stack([
            1 - probabilities,
            probabilities,
        ])
