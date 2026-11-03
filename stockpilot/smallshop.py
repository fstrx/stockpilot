"""Tomorrow's order for a small shop with only a few weeks of records (e.g. the school canteen).

A few weeks is too little to train the gradient-boosting model, so for each item
StockPilot uses a simple, explainable forecaster:

    forecast = recent level x day-of-week factor

and measures how far real days fell from that pattern to get a likely range.
The order itself comes from the same Layer 2 rules as the full model:
the newsvendor critical ratio for perishables, a reorder point for packaged items.

Log format (one row per item per open day):
    date, item, sold, leftover
`leftover` is what was left at closing: thrown away for 1-day items, still on
the shelf for longer-life and packaged items.
"""
import math
from datetime import date, timedelta
from statistics import NormalDist

import numpy as np
import pandas as pd

from .optimizer import choose_supplier, critical_ratio, order_level, reorder_point

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
SOLD_OUT_UPLIFT = 1.10      # a sold-out day hides extra demand; assume it was about 10% higher
SHRINK = 2.0                # pulls day-of-week factors towards 1 when there are few weeks of data
PACKAGED_COVER_DAYS = 5     # packaged items are topped up to cover a school week beyond the reorder point


def next_open_day(last: date, holidays=(), open_days=(0, 1, 2, 3, 4)) -> date:
    d = last + timedelta(days=1)
    while d.weekday() not in open_days or d in set(holidays):
        d += timedelta(days=1)
    return d


def _item_forecast(hist: pd.DataFrame, target_dow: int):
    hist = hist.sort_values("date")
    demand = hist["sold"].astype(float).where(hist["leftover"] > 0, hist["sold"] * SOLD_OUT_UPLIFT)
    dows = hist["date"].dt.dayofweek
    level_all = demand.mean()
    factors = {}
    for d in range(7):
        m = dows == d
        n = int(m.sum())
        raw = demand[m].mean() / level_all if n and level_all > 0 else 1.0
        factors[d] = (n * raw + SHRINK * 1.0) / (n + SHRINK)
    level = demand.tail(10).mean() / np.mean([factors[d] for d in dows.tail(10)])
    fitted = level_all / np.mean(list(factors[d] for d in dows)) * dows.map(factors)
    sd = max(float(np.std(demand - fitted, ddof=1)) if len(demand) > 2 else 0.0, 0.10 * level)
    return level * factors[target_dow], sd, factors[target_dow], level


def tomorrow_order(log: pd.DataFrame, items: pd.DataFrame, suppliers: pd.DataFrame, order_for: date, min_service=None):
    """Return a table with one row per item: suggested order, likely demand range, supplier, reason."""
    log = log.copy()
    log["date"] = pd.to_datetime(log["date"])
    z = NormalDist()
    dow = order_for.weekday()
    rows = []
    for _, it in items.iterrows():
        hist = log[log["item"] == it["item"]]
        if len(hist) < 3:
            rows.append({"Item": it["item"], "Order": "–", "Likely demand": "–", "Supplier": "–",
                         "Reason": "Needs at least 3 days of records"})
            continue
        fc, sd, factor, level = _item_forecast(hist, dow)
        lo, hi = max(fc + z.inv_cdf(0.1) * sd, 0), fc + z.inv_cdf(0.9) * sd
        last5 = hist.sort_values("date").tail(5)
        reasons = []
        if abs(factor - 1) >= 0.10:
            reasons.append(f"{DAYS[dow]}: demand usually {abs(factor - 1) * 100:.0f}% {'higher' if factor > 1 else 'lower'}")

        on_hand = float(hist.sort_values("date")["leftover"].iloc[-1])
        if it["type"] == "packaged":
            avg, sdd = level, sd
            rop, _ = reorder_point(avg, sdd, it["lead_days"])
            after_tomorrow = on_hand - fc
            if after_tomorrow <= rop:
                qty = math.ceil(rop + avg * PACKAGED_COVER_DAYS - after_tomorrow)
                reasons.append(f"Stock {on_hand:.0f} falls below the reorder point of {rop:.0f}")
            else:
                qty = 0
                reasons.append(f"Stock {on_hand:.0f} covers about {on_hand / max(avg, 1e-9):.0f} days; no order needed")
            max_lead = 3
        else:
            cr = critical_ratio(it["cost"], it["price"], it["leftover_value"], int(it["shelf_life"]))
            level_used = order_level(cr, min_service)
            target = fc + z.inv_cdf(min(max(level_used, 0.05), 0.95)) * sd
            usable = on_hand if it["shelf_life"] > 1 else 0.0
            qty = max(0, math.ceil(target - usable))
            sold_out = int((last5["leftover"] == 0).sum())
            left = int((last5["leftover"] > 0).sum()) if it["shelf_life"] == 1 else 0
            if sold_out >= 2:
                reasons.append(f"Sold out on {sold_out} of the last {len(last5)} days")
            if left >= 3:
                reasons.append(f"Left over on {left} of the last {len(last5)} days")
            if min_service and level_used > cr:
                pass                                   # the dashboard states the minimum-service setting once
            elif cr < 0.45:
                reasons.append("Ordered below typical demand: waste costs more than a missed sale")
            elif cr > 0.6:
                reasons.append("Ordered above typical demand: a missed sale costs more than a leftover")
            if usable > 0 and qty == 0:
                reasons = [f"{usable:g} {it['unit']} still fresh from today cover tomorrow"]
            elif usable > 0:
                reasons.append(f"{usable:g} {it['unit']} still fresh from today")
            max_lead = 1
        supplier, total = choose_supplier(it["item"], qty, suppliers, max_lead)
        rows.append({
            "Item": it["item"],
            "Order": f"{qty:g} {it['unit']}" if qty else "No order",
            "Likely demand": f"{lo:.0f}–{hi:.0f} {it['unit']}",
            "Supplier": supplier or ("–" if qty == 0 else "No supplier in time"),
            "Cost (₹)": round(total) if total else 0,
            "Reason": "; ".join(reasons[:2]) or "Steady demand",
            "_qty": qty, "_forecast": fc,
        })
    return pd.DataFrame(rows)
