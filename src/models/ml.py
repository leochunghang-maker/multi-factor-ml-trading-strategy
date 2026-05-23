import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import FEATURES, RANDOM_FOREST_PARAMS, TARGET, XGBOOST_PARAMS


def train_test_split_by_date(
    df: pd.DataFrame,
    split_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("date")
    return df[df["date"] < split_date], df[df["date"] >= split_date]


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    # The Random Forest learns nonlinear relationships between factors and
    # next-month returns. It is used for ranking stocks, not forecasting prices.
    model = RandomForestRegressor(**RANDOM_FOREST_PARAMS)
    model.fit(X_train, y_train)
    return model


def train_baseline_models(X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, object]:
    models = {
        "linear_regression": make_pipeline(StandardScaler(), LinearRegression()),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "lasso": make_pipeline(StandardScaler(), Lasso(alpha=0.0001, max_iter=10_000)),
        "random_forest": train_random_forest(X_train, y_train),
    }
    for name, model in models.items():
        if name != "random_forest":
            model.fit(X_train, y_train)
    return models


def evaluate_model_suite(
    models: dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        predictions = model.predict(X_test)
        mse, rank_ic = evaluate_predictions(y_test, predictions)
        rows.append({
            "model": name,
            "mean_squared_error": mse,
            "rank_ic": rank_ic,
            "prediction_std": pd.Series(predictions).std(),
        })
    return pd.DataFrame(rows).sort_values("rank_ic", ascending=False)


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series):
    from xgboost import XGBRegressor

    model = XGBRegressor(**XGBOOST_PARAMS)
    model.fit(X_train, y_train)
    return model


def evaluate_predictions(y_true: pd.Series, predictions) -> tuple[float, float]:
    mse = mean_squared_error(y_true, predictions)
    rank_ic, _ = spearmanr(y_true, predictions)
    return mse, rank_ic


def feature_importance_frame(model, features: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
