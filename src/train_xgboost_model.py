import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import FEATURES, MODEL_SPLIT_DATE, PREDICTION_COLUMN, TARGET, XGBOOST_PREDICTIONS_PATH
from src.data.io import load_ml_dataset
from src.models.ml import (
    evaluate_predictions,
    feature_importance_frame,
    train_test_split_by_date,
    train_xgboost,
)


def main() -> None:
    df = load_ml_dataset()
    train_df, test_df = train_test_split_by_date(df, split_date=MODEL_SPLIT_DATE)

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    print("Training XGBoost model...")
    model = train_xgboost(X_train, y_train)

    predictions = model.predict(X_test)
    mse, rank_ic = evaluate_predictions(y_test, predictions)

    print()
    print("XGBoost Model Evaluation")
    print("Mean Squared Error:", mse)
    print("Rank IC:", rank_ic)

    importance_df = feature_importance_frame(model, FEATURES)

    print()
    print("Feature Importances:")
    print(importance_df)

    test_df = test_df.copy()
    test_df[PREDICTION_COLUMN] = predictions

    test_df.to_csv(
        XGBOOST_PREDICTIONS_PATH,
        index=False,
    )

    print()
    print(f"Predictions saved to {XGBOOST_PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
