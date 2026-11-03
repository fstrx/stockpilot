"""Settings shared by every part of StockPilot."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"

KAGGLE_TRAIN = DATA / "train.csv"          # Kaggle "Store Item Demand Forecasting Challenge"
TRAIN_END = "2016-12-31"                   # train on 2013–2016
TEST_START, TEST_END = "2017-01-01", "2017-12-31"   # test on 2017, never seen in training

# Layer 1: one gradient-boosted model per quantile. Together they describe the
# whole range of likely demand, not just one number.
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# The Kaggle items are anonymous, so for the digital twin each item is given the
# economics of a typical school-canteen product. These are ASSUMPTIONS for the
# simulation, not measured values. Item n uses profile (n - 1) % 6.
#   cost / price / leftover value in rupees per unit, shelf life in days
ITEM_PROFILES = [
    {"name": "Milk packet",   "cost": 26, "price": 34, "leftover_value": 0,  "shelf_life": 1},
    {"name": "Bread loaf",    "cost": 35, "price": 45, "leftover_value": 10, "shelf_life": 2},
    {"name": "Samosa",        "cost": 8,  "price": 15, "leftover_value": 0,  "shelf_life": 1},
    {"name": "Veg sandwich",  "cost": 20, "price": 35, "leftover_value": 0,  "shelf_life": 1},
    {"name": "Paneer 200 g",  "cost": 70, "price": 95, "leftover_value": 0,  "shelf_life": 3},
    {"name": "Fruit cup",     "cost": 15, "price": 30, "leftover_value": 0,  "shelf_life": 1},
]

# Habit policy used for comparison: "order what sold yesterday plus a margin",
# which is how the owners we interviewed described their ordering.
HABIT_MARGIN = 0.15
HABIT_MARGIN_SENSITIVITY = [0.0, 0.15, 0.30]


# Optional "students first" setting: always cover demand on at least this share of days,
# even when the newsvendor rule says a lower level makes more profit.
MIN_SERVICE_OPTION = 0.70


def profile_for_item(item: int) -> dict:
    return ITEM_PROFILES[(int(item) - 1) % len(ITEM_PROFILES)]
