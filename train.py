"""Train StockPilot on the Kaggle data, test it on 2017 and run the digital twin.

    python train.py            # full run (a few minutes)

Writes everything the dashboard needs into artifacts/.
"""
import json
import time

import joblib
import numpy as np
import pandas as pd

from stockpilot import config as C
from stockpilot.features import add_features, load_kaggle, next_day_rows
from stockpilot.forecast import predict_quantiles, quantile_at, smape, train_quantile_models
from stockpilot.optimizer import critical_ratio, order_level
from stockpilot.twin import simulate, summarise


def main():
    t0 = time.time()
    C.ARTIFACTS.mkdir(exist_ok=True)
    print("Loading data and building features…")
    raw = load_kaggle(C.KAGGLE_TRAIN)
    feat = add_features(raw)
    train = feat[feat["date"] <= C.TRAIN_END]
    test = feat[(feat["date"] >= C.TEST_START) & (feat["date"] <= C.TEST_END)].sort_values(["store", "item", "date"]).reset_index(drop=True)
    print(f"  {len(train):,} training rows (to {C.TRAIN_END}), {len(test):,} test rows (2017)")

    print("Layer 1: training quantile models…")
    models = train_quantile_models(train)
    joblib.dump(models, C.ARTIFACTS / "models.joblib", compress=3)

    q = predict_quantiles(models, test)
    qs = sorted(models)
    median = q[:, qs.index(0.5)]
    test["forecast"] = median
    for i, lvl in enumerate(qs):
        test[f"q{int(lvl * 100)}"] = q[:, i]

    # ---- accuracy against the baseline -------------------------------------------------
    item_means = train.groupby("item")["sales"].mean().sort_values()
    slow, fast = item_means.index[:10], item_means.index[-10:]
    acc = {}
    for name, mask in [("all_items", slice(None)), ("slowest_10_items", test["item"].isin(slow)), ("fastest_10_items", test["item"].isin(fast))]:
        sub = test.loc[mask]
        acc[name] = {
            "baseline_same_day_last_week_smape": round(smape(sub["sales"], sub["lag_7"]), 2),
            "baseline_yesterday_smape": round(smape(sub["sales"], sub["lag_1"]), 2),
            "stockpilot_smape": round(smape(sub["sales"], sub["forecast"]), 2),
            "baseline_same_day_last_week_mae": round(float(np.abs(sub["sales"] - sub["lag_7"]).mean()), 2),
            "stockpilot_mae": round(float(np.abs(sub["sales"] - sub["forecast"]).mean()), 2),
        }
    coverage = {f"q{int(l * 100)}": round(float((test["sales"] <= q[:, i]).mean() * 100), 1) for i, l in enumerate(qs)}
    print("  accuracy:", json.dumps(acc["all_items"]))

    # ---- Layer 2 + digital twin ------------------------------------------------------------
    print("Layer 4: running the digital twin on 2017…")
    series = test[["store", "item"]].drop_duplicates().reset_index(drop=True)
    S, T = len(series), test["date"].nunique()
    shape = (S, T)
    demand = test["sales"].to_numpy().reshape(shape).astype(np.int64)
    prof = [C.profile_for_item(i) for i in series["item"]]
    cost = np.array([p["cost"] for p in prof], float)
    price = np.array([p["price"] for p in prof], float)
    leftover = np.array([p["leftover_value"] for p in prof], float)
    life = np.array([p["shelf_life"] for p in prof], int)
    cr = np.array([critical_ratio(p["cost"], p["price"], p["leftover_value"], p["shelf_life"]) for p in prof])
    cr_rows = np.repeat(cr, T)
    ai_target = quantile_at(q, cr_rows).reshape(shape)
    last_2016 = raw[raw["date"] == C.TRAIN_END].sort_values(["store", "item"])["sales"].to_numpy()

    ai_tot, ai_daily = simulate(demand, lambda t, prev: ai_target[:, t], cost, price, leftover, life, keep_daily=True)
    twin = {"stockpilot": summarise(ai_tot, T), "habit": {}}
    floor = np.array([order_level(c, C.MIN_SERVICE_OPTION) for c in cr])
    floor_target = quantile_at(q, np.repeat(floor, T)).reshape(shape)
    floor_tot, _ = simulate(demand, lambda t, prev: floor_target[:, t], cost, price, leftover, life)
    twin[f"stockpilot_min_service_{int(C.MIN_SERVICE_OPTION * 100)}pct"] = summarise(floor_tot, T)
    habit_daily = None
    for m in C.HABIT_MARGIN_SENSITIVITY:
        def habit(t, prev, m=m):
            base = last_2016 if prev is None else prev
            return (1 + m) * base
        tot, daily = simulate(demand, habit, cost, price, leftover, life, keep_daily=(m == C.HABIT_MARGIN))
        twin["habit"][f"margin_{int(m * 100)}pct"] = summarise(tot, T)
        if m == C.HABIT_MARGIN:
            habit_tot, habit_daily = tot, daily

    # per product type
    by_type = []
    for k, p in enumerate(C.ITEM_PROFILES):
        idx = np.array([C.profile_for_item(i)["name"] == p["name"] for i in series["item"]])
        sub = lambda tot: {kk: v[idx] for kk, v in tot.items()}
        a, h = summarise(sub(ai_tot), T), summarise(sub(habit_tot), T)
        by_type.append({"product": p["name"], "critical_ratio": round(critical_ratio(p["cost"], p["price"], p["leftover_value"], p["shelf_life"]), 2),
                        "shelf_life_days": p["shelf_life"],
                        "habit_wasted": h["units_wasted"], "stockpilot_wasted": a["units_wasted"],
                        "habit_stockout_day_pct": h["stockout_day_pct"], "stockpilot_stockout_day_pct": a["stockout_day_pct"],
                        "habit_profit": h["profit_rupees"], "stockpilot_profit": a["profit_rupees"]})
    twin["by_product"] = by_type
    print("  StockPilot:", twin["stockpilot"])
    print("  StockPilot, min service 70%:", twin[f"stockpilot_min_service_{int(C.MIN_SERVICE_OPTION * 100)}pct"])
    print("  Habit +15%:", twin["habit"]["margin_15pct"])

    # ---- daily records for the dashboard -------------------------------------------------
    daily = test[["date", "store", "item", "sales", "forecast", "q10", "q90"]].copy()
    daily["stockpilot_order"] = ai_daily["order"].ravel()
    daily["stockpilot_wasted"] = ai_daily["wasted"].ravel()
    daily["stockpilot_sold"] = ai_daily["sold"].ravel()
    daily["habit_order"] = habit_daily["order"].ravel()
    daily["habit_wasted"] = habit_daily["wasted"].ravel()
    daily["habit_sold"] = habit_daily["sold"].ravel()
    for c in daily.columns[4:]:
        daily[c] = daily[c].round(1)
    daily.to_csv(C.ARTIFACTS / "twin_daily_2017.csv.gz", index=False)

    # ---- the real "tomorrow": 1 January 2018 ------------------------------------------------
    nxt = next_day_rows(raw)
    qn = predict_quantiles(models, nxt)
    nxt_prof = [C.profile_for_item(i) for i in nxt["item"]]
    nxt_cr = np.array([critical_ratio(p["cost"], p["price"], p["leftover_value"], p["shelf_life"]) for p in nxt_prof])
    out = nxt[["date", "store", "item"]].copy()
    out["forecast"] = qn[:, qs.index(0.5)].round(1)
    out["likely_low"] = qn[:, 0].round(0)
    out["likely_high"] = qn[:, -1].round(0)
    out["product_profile"] = [p["name"] for p in nxt_prof]
    out["critical_ratio"] = nxt_cr.round(2)
    out["suggested_order"] = np.ceil(quantile_at(qn, nxt_cr)).astype(int)
    out.to_csv(C.ARTIFACTS / "tomorrow_2018-01-01.csv", index=False)

    results = {
        "data": {"series": int(S), "train_period": f"{train['date'].min().date()} to {C.TRAIN_END}",
                 "test_period": f"{C.TEST_START} to {C.TEST_END}", "train_rows": int(len(train)), "test_rows": int(len(test))},
        "accuracy": acc,
        "quantile_coverage_pct": coverage,
        "digital_twin_2017": twin,
        "assumptions": {"item_profiles": C.ITEM_PROFILES, "habit_policy": "order (1 + margin) x yesterday's sales, minus fresh stock on hand",
                        "delivery": "ordered the evening before, arrives next morning", "selling": "oldest stock first"},
        "runtime_seconds": round(time.time() - t0),
    }
    (C.ARTIFACTS / "results.json").write_text(json.dumps(results, indent=2))
    print(f"Done in {results['runtime_seconds']} s. Results in artifacts/results.json")


if __name__ == "__main__":
    main()
