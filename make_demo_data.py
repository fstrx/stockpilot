"""Create SYNTHETIC demo files for the "Your shop" tab.

These are made-up numbers so the dashboard works out of the box. Replace
data/canteen_log.csv with your canteen's real records (same columns).
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd

from stockpilot.config import DATA

rng = np.random.default_rng(7)
HOLIDAYS = {date(2026, 9, 4), date(2026, 9, 14)}       # Janmashtami, Ganesh Chaturthi
days, d = [], date(2026, 8, 31)
while d <= date(2026, 9, 25):
    if d.weekday() < 5 and d not in HOLIDAYS:
        days.append(d)
    d += timedelta(days=1)

items = pd.DataFrame([
    # item, unit, type, cost, price, leftover_value, shelf_life (days), lead_days
    ["Milk", "litres", "perishable", 54, 70, 0, 1, 1],
    ["Bread", "loaves", "perishable", 35, 45, 10, 2, 1],
    ["Samosa", "pieces", "perishable", 8, 15, 0, 1, 1],
    ["Veg sandwich", "pieces", "perishable", 20, 35, 0, 1, 1],
    ["Paneer", "200 g packs", "perishable", 70, 95, 0, 3, 1],
    ["Fruit cup", "cups", "perishable", 15, 30, 0, 1, 1],
    ["Packaged juice", "packs", "packaged", 24, 35, 24, 180, 3],
    ["Chips", "packets", "packaged", 7.5, 10, 7.5, 120, 3],
], columns=["item", "unit", "type", "cost", "price", "leftover_value", "shelf_life", "lead_days"])

# demand pattern (synthetic): base level and weekday factors Mon..Fri
pattern = {
    "Milk": (16, [1.15, 1.0, 1.0, 1.0, 0.85]), "Bread": (20, [1.1, 1.0, 1.0, 1.0, 0.9]),
    "Samosa": (130, [1.0, 1.05, 1.0, 1.05, 1.2]), "Veg sandwich": (32, [1.0, 1.0, 1.1, 1.0, 0.9]),
    "Paneer": (14, [1.0, 1.1, 1.0, 1.1, 0.8]), "Fruit cup": (24, [1.0, 1.0, 1.0, 0.95, 0.8]),
    "Packaged juice": (38, [0.9, 1.0, 1.0, 1.05, 1.2]), "Chips": (55, [0.9, 1.0, 1.0, 1.05, 1.25]),
}
habit_order = {"Milk": 18, "Bread": 22, "Samosa": 130, "Veg sandwich": 38, "Paneer": 15, "Fruit cup": 30}

rows = []
stock = {"Packaged juice": 160, "Chips": 240}
fresh = {"Bread": 0, "Paneer": 0}
for day in days:
    for it, (base, f) in pattern.items():
        dem = rng.poisson(base * f[day.weekday()])
        if it in stock:
            if day.weekday() in (0, 3):                  # restocked every Monday and Thursday
                stock[it] = max(stock[it], {"Packaged juice": 200, "Chips": 300}[it])
            sold = min(dem, stock[it]); stock[it] -= sold
            rows.append([day, it, sold, stock[it]])
        else:
            available = habit_order[it] + fresh.get(it, 0)
            sold = min(dem, available)
            left = available - sold
            if it in fresh:                              # bread and paneer keep: only yesterday's batch is still fresh
                fresh[it] = min(left, habit_order[it])
            rows.append([day, it, sold, left])

log = pd.DataFrame(rows, columns=["date", "item", "sold", "leftover"])
suppliers = pd.DataFrame([
    # item, supplier, price_per_unit, delivery_cost, lead_days, min_qty  (assumed prices)
    ["Milk", "Dairy booth", 54, 0, 1, 1], ["Milk", "Dairy distributor", 51, 60, 1, 20],
    ["Bread", "Local bakery", 35, 0, 1, 1], ["Bread", "Bread distributor", 32, 50, 1, 20],
    ["Samosa", "Canteen kitchen", 8, 0, 1, 0], ["Veg sandwich", "Canteen kitchen", 20, 0, 1, 0],
    ["Paneer", "Local dairy", 72, 0, 1, 1], ["Paneer", "Dairy distributor", 66, 60, 2, 10],
    ["Fruit cup", "Local fruit seller", 15, 0, 1, 1], ["Fruit cup", "Wholesale mandi", 11, 40, 1, 20],
    ["Packaged juice", "FMCG distributor", 24, 0, 3, 24], ["Packaged juice", "Cash-and-carry store", 23, 80, 1, 48],
    ["Chips", "FMCG distributor", 7.5, 0, 3, 50], ["Chips", "Cash-and-carry store", 7.2, 80, 1, 100],
], columns=["item", "supplier", "price_per_unit", "delivery_cost", "lead_days", "min_qty"])

DATA.mkdir(exist_ok=True)
log.to_csv(DATA / "canteen_log_demo_synthetic.csv", index=False)
items.to_csv(DATA / "canteen_items.csv", index=False)
suppliers.to_csv(DATA / "suppliers.csv", index=False)
print(f"{len(log)} log rows over {len(days)} school days; files written to {DATA}")
