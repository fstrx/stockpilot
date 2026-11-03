"""Layer 4: the digital twin.

Replays real daily demand, one day at a time, for every store-item series at once,
under two ordering policies, and counts waste, stock-outs and profit.

Rules of the simulation:
* The order is placed the evening before and arrives in the morning.
* Stock is sold oldest first.
* A unit that reaches its shelf life at the end of a day is thrown away
  (and sold for its leftover value, if any).
* Demand the shop cannot meet is a lost sale.
"""
import numpy as np


def simulate(demand, target_fn, cost, price, leftover_value, shelf_life, keep_daily=False):
    """demand: int array (series, days). target_fn(t, prev_sold) -> stock level wanted for day t.

    cost, price, leftover_value, shelf_life: arrays of length `series`.
    """
    S, T = demand.shape
    L = int(shelf_life.max())
    keep_mask = np.arange(L)[None, :] + 1 < shelf_life[:, None]   # age after ageing must stay < life
    inv = np.zeros((S, L), dtype=np.int64)                          # inv[:, a] = units of age a
    prev_sold = None
    tot = {k: np.zeros(S) for k in ["ordered", "sold", "lost", "wasted", "stockout_days", "profit", "demand"]}
    daily = {k: np.zeros((S, T), dtype=np.int64) for k in ["order", "sold", "wasted"]} if keep_daily else None

    for t in range(T):
        d = demand[:, t]
        onhand = inv.sum(axis=1)
        target = np.ceil(target_fn(t, prev_sold)).astype(np.int64)
        order = np.maximum(target - onhand, 0)
        inv[:, 0] += order

        remaining = d.copy()
        for a in range(L - 1, -1, -1):          # sell oldest first
            take = np.minimum(inv[:, a], remaining)
            inv[:, a] -= take
            remaining -= take
        sold = d - remaining

        # end of day: stock that would pass its shelf life is thrown away,
        # everything else gets one day older
        wasted = (inv * ~keep_mask).sum(axis=1)
        inv = np.where(keep_mask, inv, 0)
        aged = np.zeros_like(inv)
        aged[:, 1:] = inv[:, :-1]
        inv = aged

        tot["ordered"] += order
        tot["sold"] += sold
        tot["lost"] += remaining
        tot["wasted"] += wasted
        tot["stockout_days"] += remaining > 0
        tot["demand"] += d
        tot["profit"] += price * sold - cost * order + leftover_value * wasted
        if keep_daily:
            daily["order"][:, t], daily["sold"][:, t], daily["wasted"][:, t] = order, sold, wasted
        prev_sold = sold
    return tot, daily


def summarise(tot, days):
    S = len(tot["sold"])
    return {
        "units_ordered": int(tot["ordered"].sum()),
        "units_sold": int(tot["sold"].sum()),
        "units_wasted": int(tot["wasted"].sum()),
        "units_lost": int(tot["lost"].sum()),
        "waste_pct_of_ordered": round(100 * tot["wasted"].sum() / max(tot["ordered"].sum(), 1), 2),
        "stockout_day_pct": round(100 * tot["stockout_days"].sum() / (S * days), 2),
        "fill_rate_pct": round(100 * tot["sold"].sum() / max(tot["demand"].sum(), 1), 2),
        "profit_rupees": int(round(tot["profit"].sum())),
    }
