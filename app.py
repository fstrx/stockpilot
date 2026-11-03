"""StockPilot AI dashboard.  Run with:  streamlit run app.py"""
import json
from datetime import date

import altair as alt
import pandas as pd
import streamlit as st

from stockpilot import config as C
from stockpilot.smallshop import next_open_day, tomorrow_order

st.set_page_config(page_title="StockPilot AI", page_icon="📦", layout="wide")

INK, AI_BLUE, HABIT_GREY, DEMAND = "#1f2a37", "#2563eb", "#9ca3af", "#111827"
SCHOOL_HOLIDAYS_2026 = [date(2026, 10, 2), *[date(2026, 10, d) for d in range(16, 22)], *[date(2026, 11, d) for d in range(7, 12)],
                        date(2026, 11, 24), date(2026, 12, 25)]


@st.cache_data
def load_results():
    path = C.ARTIFACTS / "results.json"
    return json.loads(path.read_text()) if path.exists() else None


@st.cache_data
def load_daily():
    return pd.read_csv(C.ARTIFACTS / "twin_daily_2017.csv.gz", parse_dates=["date"])


@st.cache_data
def load_tomorrow():
    return pd.read_csv(C.ARTIFACTS / "tomorrow_2018-01-01.csv")


st.title("StockPilot AI")
st.caption("Tells a small food business what to buy tomorrow, and how much, so less food is thrown away and fewer customers leave empty-handed.")

tab_shop, tab_twin, tab_how = st.tabs(["Tomorrow’s order", "Digital twin (Kaggle data)", "How it works"])

# ------------------------------------------------------------------------------------------
with tab_shop:
    with st.sidebar:
        st.header("Your shop’s data")
        up_log = st.file_uploader("Sales log CSV (date, item, sold, leftover)", type="csv")
        up_items = st.file_uploader("Items CSV (optional)", type="csv")
        up_sup = st.file_uploader("Suppliers CSV (optional)", type="csv")
        st.divider()
        students_first = st.toggle("Students first", value=True,
                                   help="Always cover demand on at least this share of days, even if a lower order would make more profit.")
        min_service = st.slider("Cover demand on at least", 0.50, 0.95, 0.70, 0.05, format="%.2f",
                                disabled=not students_first) if students_first else None

    using_demo = up_log is None
    log = pd.read_csv(up_log) if up_log else pd.read_csv(C.DATA / "canteen_log_demo_synthetic.csv")
    items = pd.read_csv(up_items) if up_items else pd.read_csv(C.DATA / "canteen_items.csv")
    suppliers = pd.read_csv(up_sup) if up_sup else pd.read_csv(C.DATA / "suppliers.csv")
    log["date"] = pd.to_datetime(log["date"])

    if using_demo:
        st.info("Showing **synthetic demo data** for a school canteen. Upload your own sales log in the sidebar to get your real order.")

    with st.expander(f"Records ({log['date'].nunique()} days, {log['date'].min():%d %b} to {log['date'].max():%d %b %Y}) — edit or add rows here"):
        log = st.data_editor(log, num_rows="dynamic", use_container_width=True, key="log_editor")
        log["date"] = pd.to_datetime(log["date"])

    default_day = next_open_day(log["date"].max().date(), holidays=SCHOOL_HOLIDAYS_2026)
    c1, c2 = st.columns([1, 3])
    order_for = c1.date_input("Order for", value=default_day)
    table = tomorrow_order(log, items, suppliers, order_for, min_service=min_service)

    to_order = table[table["_qty"] > 0]
    m1, m2, m3 = c2.columns(3)
    m1.metric("Items to order", f"{len(to_order)} of {len(table)}")
    m2.metric("Order cost", f"₹{int(table['Cost (₹)'].sum()):,}")
    m3.metric("Order level", f"≥{min_service:.0%} of days" if min_service else "Max profit")

    st.subheader(f"Order for {order_for:%A %d %B %Y}")
    if min_service:
        st.caption(f"Students first is on: each fresh item is ordered to cover demand on at least {min_service:.0%} of days.")
    st.dataframe(table.drop(columns=["_qty", "_forecast"]), hide_index=True, use_container_width=True)
    st.download_button("Download order as CSV", table.drop(columns=["_qty", "_forecast"]).to_csv(index=False),
                       file_name=f"stockpilot_order_{order_for}.csv", mime="text/csv")

    st.subheader("Item history")
    pick = st.selectbox("Item", items["item"].tolist())
    h = log[log["item"] == pick].melt(id_vars="date", value_vars=["sold", "leftover"], var_name="measure", value_name="units")
    chart = alt.Chart(h).mark_line(point=True).encode(
        x=alt.X("date:T", title=None), y=alt.Y("units:Q", title=items.set_index("item").loc[pick, "unit"]),
        color=alt.Color("measure:N", scale=alt.Scale(domain=["sold", "leftover"], range=[AI_BLUE, HABIT_GREY]), title=None),
        tooltip=["date:T", "measure:N", "units:Q"]).properties(height=260)
    st.altair_chart(chart, use_container_width=True)

# ------------------------------------------------------------------------------------------
with tab_twin:
    res = load_results()
    if res is None:
        st.warning("Run `python train.py` first to train the model and create the digital twin results.")
    else:
        acc, tw = res["accuracy"]["all_items"], res["digital_twin_2017"]
        st.markdown(f"Trained on **{res['data']['train_period']}**, tested on **{res['data']['test_period']}** "
                    f"({res['data']['series']} store–item series the model never saw in that year).")
        a1, a2, a3 = st.columns(3)
        a1.metric("Forecast error, StockPilot (SMAPE)", f"{acc['stockpilot_smape']:.1f}%",
                  f"{acc['stockpilot_smape'] - acc['baseline_same_day_last_week_smape']:+.1f} pts vs baseline", delta_color="inverse")
        a2.metric("Baseline: same day last week", f"{acc['baseline_same_day_last_week_smape']:.1f}%")
        a3.metric("90% range actually covered", f"{res['quantile_coverage_pct']['q90']:.1f}% of days")

        st.subheader("A year of ordering, replayed")
        h = tw["habit"]["margin_15pct"]
        ms_key = [k for k in tw if k.startswith("stockpilot_min_service")][0]
        rows = [("Habit: yesterday’s sales + 15%", h), ("StockPilot: most profitable", tw["stockpilot"]),
                (f"StockPilot: students first ({ms_key.split('_')[-1].replace('pct', '%')})", tw[ms_key])]
        comp = pd.DataFrame([{
            "Policy": n, "Units thrown away": f"{r['units_wasted']:,}", "Waste vs habit": f"{100 * (r['units_wasted'] / h['units_wasted'] - 1):+.0f}%",
            "Days that ran out": f"{r['stockout_day_pct']:.0f}%", "Demand met": f"{r['fill_rate_pct']:.1f}%",
            "Profit vs habit": f"{100 * (r['profit_rupees'] / h['profit_rupees'] - 1):+.1f}%"} for n, r in rows])
        st.dataframe(comp, hide_index=True, use_container_width=True)
        st.caption("The Kaggle items are anonymous, so each is given the prices and shelf life of a typical canteen product "
                   "(see How it works). Orders are placed the evening before; stock sells oldest first; expired stock is thrown away.")

        bp = pd.DataFrame(tw["by_product"])
        long = pd.concat([bp[["product", "habit_wasted"]].rename(columns={"habit_wasted": "units"}).assign(policy="Habit +15%"),
                          bp[["product", "stockpilot_wasted"]].rename(columns={"stockpilot_wasted": "units"}).assign(policy="StockPilot")])
        st.altair_chart(alt.Chart(long).mark_bar().encode(
            y=alt.Y("product:N", title=None, sort=list(bp["product"])), x=alt.X("units:Q", title="Units thrown away in 2017"),
            yOffset="policy:N", color=alt.Color("policy:N", scale=alt.Scale(domain=["Habit +15%", "StockPilot"], range=[HABIT_GREY, AI_BLUE]), title=None),
            tooltip=["product", "policy", alt.Tooltip("units:Q", format=",")]).properties(height=300, title="Waste by product"),
            use_container_width=True)

        st.subheader("Look at one item")
        daily = load_daily()
        s1, s2, s3 = st.columns(3)
        store = s1.selectbox("Store", sorted(daily["store"].unique()))
        item = s2.selectbox("Item", sorted(daily["item"].unique()), format_func=lambda i: f"{i} ({C.profile_for_item(i)['name']})")
        month = s3.selectbox("Month (2017)", range(1, 13), index=5, format_func=lambda m: date(2017, m, 1).strftime("%B"))
        one = daily[(daily["store"] == store) & (daily["item"] == item)]
        view = one[one["date"].dt.month == month]
        lines = view.melt(id_vars="date", value_vars=["sales", "stockpilot_order", "habit_order"], var_name="series", value_name="units")
        names = {"sales": "Actual demand", "stockpilot_order": "StockPilot order", "habit_order": "Habit order"}
        lines["series"] = lines["series"].map(names)
        band = alt.Chart(view).mark_area(opacity=0.15, color=AI_BLUE).encode(x="date:T", y="q10:Q", y2="q90:Q")
        ln = alt.Chart(lines).mark_line(point=True).encode(
            x=alt.X("date:T", title=None), y=alt.Y("units:Q", title="Units"),
            color=alt.Color("series:N", scale=alt.Scale(domain=list(names.values()), range=[DEMAND, AI_BLUE, HABIT_GREY]), title=None),
            tooltip=["date:T", "series:N", "units:Q"])
        st.altair_chart((band + ln).properties(height=300), use_container_width=True)
        st.caption("Shaded band: StockPilot’s likely range (10th to 90th percentile). StockPilot line uses the most-profitable setting. For items that keep more than a day, "
                   "the order line is what was bought; leftovers from the day before are sold first.")
        k1, k2, k3 = st.columns(3)
        k1.metric("Thrown away in 2017", f"{int(one['stockpilot_wasted'].sum()):,}", f"{int(one['stockpilot_wasted'].sum() - one['habit_wasted'].sum()):+,} vs habit", delta_color="inverse")
        k2.metric("Units sold", f"{int(one['stockpilot_sold'].sum()):,}", f"{int(one['stockpilot_sold'].sum() - one['habit_sold'].sum()):+,} vs habit")
        k3.metric("Forecast error (SMAPE)", f"{(200 * (one['forecast'] - one['sales']).abs() / (one['forecast'].abs() + one['sales'].abs())).mean():.1f}%")

        st.subheader("Tomorrow: 1 January 2018")
        tm = load_tomorrow()
        st.dataframe(tm[tm["store"] == store].drop(columns=["date", "store"]).rename(columns={"item": "Item", "forecast": "Forecast", "likely_low": "Likely low", "likely_high": "Likely high", "product_profile": "Treated as", "critical_ratio": "Critical ratio", "suggested_order": "Suggested order"}), hide_index=True, use_container_width=True)

# ------------------------------------------------------------------------------------------
with tab_how:
    st.markdown("""
**Layer 1 · Demand forecast (machine learning).** A gradient-boosted tree model (scikit-learn
`HistGradientBoostingRegressor`) predicts tomorrow’s sales for every item from the day of the week, the month,
yesterday’s sales, sales on the same day last week and two weeks ago, and the 7- and 28-day averages. It predicts
sales as a multiple of the 28-day average, so it keeps up with a shop that is growing. Nine models, one per
quantile (10% to 90%), give the whole likely range instead of a single number.

**Layer 2 · Order optimiser (operations research).** For fresh items, the *newsvendor model* balances the cost of
ordering one too few (lost profit = price − cost) against one too many (waste = cost − leftover value):
""")
    st.latex(r"\text{critical ratio} = \frac{C_u}{C_u + C_o}")
    st.markdown("""
StockPilot orders the quantity that covers demand on that share of days. Items that keep for several days usually
sell the next day, so their waste cost is divided by the shelf life. **Students first** raises the level to a
minimum share of days. Packaged items use a reorder point:
""")
    st.latex(r"\text{reorder point} = \bar{d} \times L + z \times \sigma_d \times \sqrt{L}")
    st.markdown("""
**Layer 3 · Sourcing.** For each item, the cheapest supplier that can deliver in time and accepts the quantity.

**Layer 4 · Digital twin.** Replays every day of 2017 for all 500 store–item series under the habit policy and
under StockPilot, counting waste, stock-outs and profit.

**Small shops.** A canteen has only a few weeks of records, too little to train Layer 1. For each item StockPilot
uses *recent level × day-of-week factor*, measures how far real days fell from that pattern to get the likely
range, and then applies the same Layer 2 rules.

**Data.** Kaggle *Store Item Demand Forecasting Challenge* (10 stores × 50 items, daily, 2013–2017). The canteen
data shown by default is synthetic; product prices and shelf lives in the twin are assumptions, listed in
`stockpilot/config.py`.
""")
