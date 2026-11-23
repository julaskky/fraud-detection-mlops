
from xgboost import XGBClassifier

from src.models.common import (
    load_prepared_data,
    save_validation_result,
)


def train():
    data = load_prepared_data()

    y_train = data["y_train"]

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    scale_pos_weight = (
        negative_count / positive_count
    )

    model = XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="aucpr",
        early_stopping_rounds=40,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    print("Training XGBoost...")
    print("Positive class weight:", scale_pos_weight)

    model.fit(
        data["X_train"],
        y_train,
        eval_set=[
            (
                data["X_validation"],
                data["y_validation"],
            )
        ],
        verbose=False,
    )

    save_validation_result(
        "xgboost",
        model,
        data["X_validation"],
        data["y_validation"],
    )


if __name__ == "__main__":
    train()
