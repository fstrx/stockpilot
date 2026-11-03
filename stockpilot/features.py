"""Turn a daily sales table into model features.

Every feature for day t uses only sales up to day t-1, so the model forecasts
"tomorrow" with exactly the information an owner has the evening before.
"""
import pandas as pd

# The model predicts each day's sales as a multiple of the last 28 days' average.
# Working in ratios lets it follow a business that is growing or shrinking,
# which a model of raw sales levels cannot do.
FEATURES = ["store", "item", "dow", "month", "r_lag_1", "r_lag_7", "r_lag_14", "r_mean_7"]
TARGET = "y_ratio"
CATEGORICAL = ["store", "item", "dow", "month"]


def load_kaggle(path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values(["store", "item", "date"]).reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """df has columns date, store, item, sales (one row per series per day)."""
    df = df.sort_values(["store", "item", "date"]).copy()
    g = df.groupby(["store", "item"], sort=False)["sales"]
    df["lag_1"] = g.shift(1)                       # yesterday
    df["lag_7"] = g.shift(7)                       # same day last week (also the baseline)
    df["lag_14"] = g.shift(14)
    df["mean_7"] = g.transform(lambda s: s.shift(1).rolling(7).mean())
    df["mean_28"] = g.transform(lambda s: s.shift(1).rolling(28).mean())
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    scale = df["mean_28"].where(df["mean_28"] > 0, 1.0)
    for c in ["lag_1", "lag_7", "lag_14", "mean_7"]:
        df["r_" + c] = df[c] / scale
    df["y_ratio"] = df["sales"] / scale
    return df.dropna(subset=["lag_1", "lag_7", "lag_14", "mean_7", "mean_28"]).reset_index(drop=True)


def next_day_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Feature rows for the day after the last date in df (the real 'tomorrow')."""
    last = df["date"].max()
    tomorrow = last + pd.Timedelta(days=1)
    stub = df[["store", "item"]].drop_duplicates().assign(date=tomorrow, sales=float("nan"))
    full = add_features(pd.concat([df[["date", "store", "item", "sales"]], stub], ignore_index=True).assign(
        sales=lambda d: d["sales"]))
    return full[full["date"] == tomorrow].reset_index(drop=True)
