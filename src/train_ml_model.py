import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (
    FEATURES,
    ML_BASELINE_REPORT_PATH,
    ML_PREDICTIONS_PATH,
    MODEL_SPLIT_DATE,
    PREDICTION_COLUMN,
    TARGET,
)
from src.data.io import load_ml_dataset
from src.models.ml import (
    evaluate_predictions,
    feature_importance_frame,
    evaluate_model_suite,
    train_baseline_models,
    train_random_forest,
    train_test_split_by_date,
)


def main() -> None:
    df = load_ml_dataset()
    train_df, test_df = train_test_split_by_date(df, split_date=MODEL_SPLIT_DATE)

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    print("Training model...")
    model = train_random_forest(X_train, y_train)
    baseline_models = train_baseline_models(X_train, y_train)

    predictions = model.predict(X_test)
    mse, rank_ic = evaluate_predictions(y_test, predictions)
    baseline_report = evaluate_model_suite(baseline_models, X_test, y_test)

    print()
    print("ML Model Evaluation")
    print()
    print("Mean Squared Error:", mse)
    print("Rank IC:", rank_ic)
    print()
    print("Baseline Comparison:")
    print()
    print(baseline_report.to_string(index=False))

    importance_df = feature_importance_frame(model, FEATURES)

    print()
    print("Feature Importances:")
    print()
    print(importance_df)

    test_df = test_df.copy()
    test_df[PREDICTION_COLUMN] = predictions

    test_df.to_csv(
        ML_PREDICTIONS_PATH,
        index=False,
    )
    baseline_report.to_csv(ML_BASELINE_REPORT_PATH, index=False)

    print()
    print(f"Predictions saved to {ML_PREDICTIONS_PATH}")
    print(f"Baseline comparison saved to {ML_BASELINE_REPORT_PATH}")


if __name__ == "__main__":
    main()
