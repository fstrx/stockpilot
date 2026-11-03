# StockPilot AI

Tells a small food business what to buy tomorrow, and how much, so less food is thrown
away and fewer customers leave empty-handed. Class XII AI capstone project,
Delhi Public School, Kolar Road, Bhopal.

## Run it

```bash
pip install -r requirements.txt
python train.py            # trains the model, tests it on 2017, runs the digital twin (about 3 minutes)
streamlit run app.py       # opens the dashboard in your browser
```

`artifacts/` already contains a trained run, so `streamlit run app.py` works straight away.
In Google Colab, open `StockPilot.ipynb` and run the cells from top to bottom.

To put the dashboard online: push this folder to a GitHub repository, then create an app on
[Streamlit Community Cloud](https://streamlit.io/cloud) pointing at `app.py`.

## How it works

| Layer | What it does | Code |
| --- | --- | --- |
| 1. Demand forecast | Gradient-boosted trees (scikit-learn), one model per quantile from 10% to 90%, predicting tomorrow’s sales as a multiple of the 28-day average | `stockpilot/forecast.py`, `features.py` |
| 2. Order optimiser | Newsvendor critical ratio for fresh items, reorder point with safety stock for packaged items, optional “students first” minimum service level | `stockpilot/optimizer.py` |
| 3. Sourcing | Cheapest supplier that delivers in time and accepts the quantity | `stockpilot/optimizer.py` |
| 4. Digital twin | Replays every day of 2017 for 500 store–item series under the habit policy and under StockPilot | `stockpilot/twin.py` |
| Small shops | Recent level × day-of-week factor for shops with only a few weeks of records, then the same Layer 2 rules | `stockpilot/smallshop.py` |

## Results (from `artifacts/results.json`)

Trained on 2013–2016, tested on 2017 (182,500 store–item–days the model never saw).

| Forecast error (SMAPE) | Baseline: same day last week | StockPilot |
| --- | --- | --- |
| All items | 17.3% | **12.4%** |
| Slowest 10 items | 23.6% | **16.9%** |
| Fastest 10 items | 13.2% | **9.4%** |

The 90% line of the forecast range was enough on 91.4% of 2017 days, so the range is well calibrated.

Digital twin, all of 2017, compared with the habit “order yesterday’s sales + 15%”:

| Policy | Waste | Days that ran out | Demand met | Profit |
| --- | --- | --- | --- | --- |
| Habit: yesterday + 15% | 1,097,060 units | 36% | 95.2% | baseline |
| StockPilot, most profitable | **72% less** | 52% | 93.5% | **+11.4%** |
| StockPilot, students first (70%) | **31% less** | **24%** | **97.7%** | **+8.4%** |

The most-profitable setting runs out more often because, for low-margin items such as milk, a
leftover costs more than a missed sale. The students-first setting beats the habit on every
measure, so it is the dashboard default.

## Data and assumptions

* `data/train.csv`: Kaggle *Store Item Demand Forecasting Challenge* (10 stores × 50 items, daily, 2013–2017).
* The Kaggle items are anonymous, so in the twin each is given the prices and shelf life of a typical
  canteen product (`stockpilot/config.py`). Results depend on these assumptions.
* The twin treats recorded sales as true demand.
* `data/canteen_log_demo_synthetic.csv`, `canteen_items.csv` and `suppliers.csv` are **synthetic demo
  data** created by `make_demo_data.py`. Replace the log with the canteen’s real records
  (columns: `date, item, sold, leftover`).
