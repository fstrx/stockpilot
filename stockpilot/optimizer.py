"""Layers 2 and 3: how much to order, and from whom."""
import math

import pandas as pd
from statistics import NormalDist


def critical_ratio(cost: float, price: float, leftover_value: float = 0.0, shelf_life: int = 1) -> float:
    """Newsvendor critical ratio Cu / (Cu + Co).

    Cu = profit lost by ordering one too few  = price - cost
    Co = money lost by ordering one too many  = cost - leftover value
    The best order covers demand on this share of days.

    An item that keeps for several days is usually sold the next day, so only
    about 1 in `shelf_life` extra units ends up thrown away. We divide Co by the
    shelf life as a simple approximation.
    """
    cu = price - cost
    co = max(cost - leftover_value, 0.0) / max(shelf_life, 1)
    if cu + co <= 0:
        return 0.5
    return cu / (cu + co)


def order_level(cr: float, min_service: float | None = None) -> float:
    """Share of days the order should cover: the critical ratio, raised to a minimum
    if the owner chooses one (e.g. a canteen that does not want students to go without)."""
    return max(cr, min_service) if min_service else cr


def reorder_point(avg_daily: float, sd_daily: float, lead_days: float, service_level: float = 0.95):
    """Shelf-stable items: reorder when stock falls to this level.

    Reorder point = (average daily demand x lead time) + z x sd x sqrt(lead time)
    """
    z = NormalDist().inv_cdf(service_level)
    safety = z * sd_daily * math.sqrt(lead_days)
    return avg_daily * lead_days + safety, safety


def choose_supplier(item: str, qty: float, suppliers: pd.DataFrame, max_lead_days: int):
    """Cheapest supplier for this item that delivers in time and accepts the quantity.

    suppliers columns: item, supplier, price_per_unit, delivery_cost, lead_days, min_qty
    Returns (supplier name, total cost) or (None, None).
    """
    opts = suppliers[(suppliers["item"] == item) & (suppliers["lead_days"] <= max_lead_days)].copy()
    if opts.empty or qty <= 0:
        return None, None
    opts["order_qty"] = opts["min_qty"].clip(lower=qty)
    opts["total"] = opts["order_qty"] * opts["price_per_unit"] + opts["delivery_cost"]
    best = opts.sort_values("total").iloc[0]
    return best["supplier"], float(best["total"])
