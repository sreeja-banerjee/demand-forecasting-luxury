#Demand forecasting and replenishment simulation for a fictional luxury brand

# ===========================================================================
#  SETUP
# ==============================================================================
#  load relevant libraries : numpy and pandas

from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
# creating output/ data folders to store our simulated tables 
DATA = BASE / "data"
OUT = BASE / "outputs"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

# ===========================================================================
# ASSUMPTIONS
# ==============================================================================
# set imaginary numbers (can be changed at any time, rerunning the script will then
# change the analysis)

# set as True to run both forecasts for comparison
use_ML = True
SEED = 67
# set the total number of weeks (e.g. for 2 years)
num_weeks = 104
# use last 26 weeks for the "test" data for ML
test_weeks = 26
startdate = "2024-01-01"
# set a window forthe moving-average forecast
MA_window = 8
# when the imaginary planner checks stock and places orders
review_time = 4
# service level z score for each "tier", using SN distribution
#   A = 98% -> z: 2.05 ; B = 95% -> z: 1.65 ; C = 90% -> z: 1.28   
z_score = {"A": 2.05, "B": 1.65, "C": 1.28}
# if weeks of cover < 2 there is a "Stockout risk"; woc > 12 means "Overstock"
stockout_time = 2
overstock_time = 12

#===========================================================================
# ASSUMPTIONS for store and products
# ==============================================================================
# describe brand: 10 boutiques, 4 categories, supplier lead times.

# for each boutique: (store id, city, region, tier)
store_list = [
    ("S01", "Milan", "EMEA", "A"),
    ("S02", "Paris", "EMEA", "A"),
    ("S03", "Vienna", "EMEA", "B"),
    ("S04", "Berlin", "EMEA", "C"),
    ("S05", "Tokyo", "APAC", "A"),
    ("S06", "Seoul", "APAC", "B"),
    ("S07", "Singapore", "APAC", "B"),
    ("S08", "Hong Kong", "APAC", "B"),
    ("S09", "Sydney", "APAC", "C"),
    ("S10", "Bangkok", "APAC", "C"),
]
# simulate store traffic: e.g. an A store sells 1.5x a B store
tier_traffic = {"A": 1.5, "B": 1.0, "C": 0.5}

# each category: (number of products, price range EUR, weekly units in a B tier store, cost / price: e.g. cost = 28% of price)

categories = {
    "Handbags": (5, (2000, 4500), 2.2, 0.28),
    "Handmade leather goods": (7, (400, 950), 3.5, 0.25),
    "Shoes": (4, (700, 1300), 1.8, 0.30),
    "Accessories": (3, (250, 600), 2.8, 0.22),
}

# each pdt's average lead time is drawn from range in weeks
supplier_lead_time = (6, 12)

# lead time standard deviation (weeks)
lead_time_SD = 1.5

# APAC boutiques sell 25% more handbags - for special feature
apac_handbag_uplift = 1.25

NB_DISPERSION = 3.0
randomnumbergenerator = np.random.default_rng(SEED)

# turn store list into a pandas table (for merging)
stores = pd.DataFrame(store_list, columns=["store_id", "city", "region", "tier"])


#===========================================================================
# PRODUCT TABLE
# ==============================================================================
# Create each product and draw its price, popularity and lead time at random.
# empty list to collect one dictionary per product

rows, i = [], 1
# loop over the 4 categories
for cat, (n, (pmin, pmax), base, cost_ratio) in categories.items():
    # make n products in this category
    for _ in range(n):
        # random price, rounded 
        price = round(randomnumbergenerator.uniform(pmin, pmax), -1)
        rows.append({
            "sku_id": f"P{i:02d}",
            "category": cat,
            "retail_price_eur": price,
            # cost = price x cost ratio
            "unit_cost_eur": round(price * cost_ratio, 2),
            # popularity: the category's base sales x a random multiplier
            # (lognormal = usually near 1
            "base_demand": base * randomnumbergenerator.lognormal(0, 0.35),
            "lead_time_wk": int(randomnumbergenerator.integers(*supplier_lead_time, endpoint=True)),
        })
        i += 1

products = pd.DataFrame(rows)

#===========================================================================
# SEASONALITY
# ==============================================================================
# Function to capture that says how busy each week is, by region.
# Output is a multiplier: 1.0 = normal week, 1.6 = 60% busier.

def seasonality(week_idx, region):
    # week of year, 0-51 (week 52 = week 0 of the next year)
    woy = week_idx % 52
    s = 1 + 0.15 * np.sin(2 * np.pi * (woy - 10) / 52)
    # Christmas, weeks 48-51: x1.6 everywhere
    s = s * np.where(woy >= 48, 1.6, 1.0)
    if region == "APAC":
        # CNY, weeks 4-6 (late Jan to mid Feb)
        s = s * np.where((woy >= 4) & (woy <= 6), 1.5, 1.0)
    else:
        # summer tourists, weeks 26-34 
        s = s * np.where((woy >= 26) & (woy <= 34), 1.25, 1.0)
    return s

#===========================================================================
# SIMULATE 2 YEARS OF SALES AND STOCK
# ==============================================================================
# Simulate 104 weeks of client demand
# Let a "naive planner" run  

weeks = np.arange(num_weeks)
# the matching Monday dates: 2024-01-01, 2024-01-08, ..., 2025-12-22
dates = pd.date_range(startdate, periods= num_weeks, freq="W-MON")
# empty list: one dictionary per store x product x week will go in here
out = []

# loop over stores
for _, st in stores.iterrows():
    # this store's 104 weekly multipliers (depends on its region)
    season = seasonality(weeks, st.region)
    # loop over the products
    for _, pr in products.iterrows():
        # average weekly demand = product popularity x store traffic x seasonality
        mean = pr.base_demand * tier_traffic[st.tier] * season
        # APAC handbag boost
        if st.region == "APAC" and pr.category == "Handbags":
            mean = mean * apac_handbag_uplift
        # negative binomial's "p" parameter
        p = NB_DISPERSION / (NB_DISPERSION + mean)
        # draw 104 whole numbers around the average 
        demand = randomnumbergenerator.negative_binomial(NB_DISPERSION, p)

        # Naive planner: this product's average lead time
        lt = pr.lead_time_wk
        # opening stock = about 8 weeks of normal sales
        on_hand = int(round(pr.base_demand * tier_traffic[st.tier] * 8))
        pipeline = {}

        # 104 weeks in order
        for w in weeks:
            # any delivery due this week arrives (0 if none)
            receipts = pipeline.pop(w, 0)
            on_hand += receipts
            # note stock at the start of the week
            start = on_hand
            # sell what clients demand, but never more than what is present 
            sales = min(demand[w], on_hand)
            on_hand -= sales

            # every 4th week is order day
            if w % review_time == 0:
                # look at the last 8 weeks of demand (in week 0, use the average instead)
                hist = demand[max(0, w - 8):w] if w > 0 else np.array([mean[0]])
                # average of those weeks (minimum 0.2)
                avg = max(hist.mean(), 0.2)
                # planner's target: enough for lead time + 4 weeks until next review + 2 spare weeks
                target = avg * (lt + review_time + 2)
                # what we already have: stock in store + already ordered
                position = on_hand + sum(pipeline.values())
                # order the gap 
                qty = int(max(0, round(target - position)))
                if qty > 0:
                    # real lead time: average +/- random delay (at least 1 week)
                    actual_lt = max(1, int(round(randomnumbergenerator.normal(lt, lead_time_SD))))
                    arrive = w + actual_lt
                    # add to the pipeline (adds to any order already due that week)
                    pipeline[arrive] = pipeline.get(arrive, 0) + qty

            # record this week as one row
            out.append({
                "week": int(w), "week_start": dates[w].date(),
                "store_id": st.store_id, "sku_id": pr.sku_id,
                # true demand: not visible irl
                "demand_true": int(demand[w]),
                "units_sold": int(sales),
                "on_hand_start": int(start), "receipts": int(receipts),
                "on_hand_end": int(on_hand),
                "on_order": int(sum(pipeline.values())),
            })

# turn the dictionaries into one table
weekly = pd.DataFrame(out)

#===========================================================================
# MONEY COLUMNS AND THE STOCKOUT FLAG
# ==============================================================================

# Bring in price and cost, then calculate sales, lost sales and stock value.
# add each product's price and cost to every row (matched on sku_id)
weekly = weekly.merge(products[["sku_id", "retail_price_eur", "unit_cost_eur"]], on="sku_id")
# revenue = units sold x price
weekly["net_revenue_eur"] = weekly.units_sold * weekly.retail_price_eur
# lost sales = clients we couldn't serve x price
weekly["lost_sales_eur"] = (weekly.demand_true - weekly.units_sold) * weekly.retail_price_eur
# stock value = stock left at end of week x cost (valued at cost)
weekly["stock_value_eur"] = weekly.on_hand_end * weekly.unit_cost_eur
# stockout week = shelf hit zero AND at least one client left empty-handed
weekly["stockout"] = (weekly.on_hand_end == 0) & (weekly.demand_true > weekly.units_sold)
# drop price and cost again 
weekly = weekly.drop(columns=["retail_price_eur", "unit_cost_eur"])

#===========================================================================
# TRAIN / TEST DATA SPLIT
# ==============================================================================
#     Decide which weeks train models

# let week 78 be first test week (104 - 26)
split_week = num_weeks - test_weeks 
print(f"Training data: {dates[0].date()} to {dates[split_week - 1].date()}")
print(f"Test data:     {dates[split_week].date()} to {dates[-1].date()}")


# Defining the forecast accuracy scoring functions

# WAPE = total size of the misses / total actual sales. 
def wape(actual, forecast):
    return np.abs(actual - forecast).sum() / actual.sum()

# Bias = (total forecast - total actual) / total actual
def bias(actual, forecast):
    return (forecast - actual).sum() / actual.sum()

#===========================================================================
# BASELINE FORECAST: 8-WEEK MOVING AVERAGE 
# ==============================================================================

# MA Forecast = average units sold in the 8 weeks before 

# sort 
w = weekly.sort_values(["store_id", "sku_id", "week"]).copy()
# within each store x product:
#   shift(1) rolling(8, min_periods=4) 
w["fc_ma"] = (w.groupby(["store_id", "sku_id"]).units_sold
                .transform(lambda s: s.shift(1).rolling(MA_window, min_periods=4).mean()))

# keep only the test weeks
test = w[w.week >= split_week]
results = [{"method": f"Moving average ({MA_window} wk)",
            "WAPE": wape(test.units_sold, test.fc_ma),
            "bias": bias(test.units_sold, test.fc_ma)}]

fc_col = "fc_ma"

#===========================================================================
# ML FORECAST: FEATURE BUILDING
# ==============================================================================

# the list of all clues for model 
features = ["lag_1", "lag_2", "lag_4", "lag_8", "lag_52", "roll_mean_4", "roll_mean_8",
            "roll_std_8", "week_of_year", "is_december", "is_cny", "tier", "region",
            "category", "retail_price_eur"]

if use_ML:
    # copy of `w` with store tier/region and product category/price added
    f = w.merge(stores[["store_id", "tier", "region"]], on="store_id")
    f = f.merge(products[["sku_id", "category", "retail_price_eur"]], on="sku_id")

    # group units sold by store x product
    g = f.groupby(["store_id", "sku_id"]).units_sold
    # lags: units sold 1, 2, 4, 8 and 52 weeks earlier (52 = same week last year)
    for lag in (1, 2, 4, 8, 52):
        f[f"lag_{lag}"] = g.shift(lag)
    f["roll_mean_4"] = g.transform(lambda s: s.shift(1).rolling(4).mean())
    f["roll_mean_8"] = g.transform(lambda s: s.shift(1).rolling(8).mean())
    
    # how jumpy sales were over the last 8 weeks
    f["roll_std_8"] = g.transform(lambda s: s.shift(1).rolling(8).std())

    # calendar clues: week of year (0-51), December flag, Chinese New Year flag (APAC only)
    f["week_of_year"] = f.week % 52
    f["is_december"] = (f.week_of_year >= 48).astype(int)
    f["is_cny"] = ((f.week_of_year >= 4) & (f.week_of_year <= 6) & (f.region == "APAC")).astype(int)
    # turn text labels into codes (alphabetical)
    # tier A=0, B=1, C=2 ; region APAC=0, EMEA=1 ; Accessories=0, Handbags=1, ...
    for col in ("tier", "region", "category"):
        f[col] = f[col].astype("category").cat.codes


#===========================================================================
# ML FORECAST, TRAIN THE MODEL AND COMPARE
# ==============================================================================

if use_ML:
    from sklearn.ensemble import HistGradientBoostingRegressor

    train = f[f.week < split_week].dropna(subset=["lag_8"])
    # poisson loss for small whole number counts, up to 300 trees w 15 nodes
    model = HistGradientBoostingRegressor(loss="poisson", max_iter=300, learning_rate=0.05,
                                          max_leaf_nodes=15, random_state=SEED)
    model.fit(train[features], train.units_sold)
    f["fc_ml"] = model.predict(f[features])

    # copy the ML forecast back into 'w'
    w = w.merge(f[["store_id", "sku_id", "week", "fc_ml"]], on=["store_id", "sku_id", "week"])
    test = w[w.week >= split_week]
    results.append({"method": "Gradient boosting (Poisson)",
                    "WAPE": wape(test.units_sold, test.fc_ml),
                    "bias": bias(test.units_sold, test.fc_ml)})
    # check if WAPE is lower than MA's
    if results[1]["WAPE"] < results[0]["WAPE"]:
        fc_col = "fc_ml"

accuracy = pd.DataFrame(results)

#===========================================================================
# TODAY'S STOCK CHECK
# ==============================================================================

# Check last week, Week 103 (post Christmas), measure sd

last = w.week.max()
recent = w[w.week > last - 26]
sd = recent.groupby(["store_id", "sku_id"]).units_sold.std().rename("sd_weekly")
# only today's rows, indexed by store x product
now = w[w.week == last].set_index(["store_id", "sku_id"])

# today's stock, stock on order and forecast
plan = now[["on_hand_end", "on_order", fc_col]].join(sd).reset_index()
plan = plan.merge(stores, on="store_id").merge(
    products[["sku_id", "category", "retail_price_eur", "unit_cost_eur", "lead_time_wk"]], on="sku_id")

#     safety stock  = z x sqrt( L x sd^2  +  d^2 x lead_time_sd^2 ) AKA buffer for risks
#     reorder point = d x L + safety stock AKA expected sales while waiting and buffer
#     (d = forecast weekly sales, L = lead time in weeks, z = service level number)

d = plan[fc_col].clip(lower=0.1)
L = plan.lead_time_wk
z = plan.tier.map(z_score)
plan["safety_stock"] = np.ceil(z * np.sqrt(L * plan.sd_weekly**2 + d**2 * lead_time_SD**2))
plan["reorder_point"] = np.ceil(d * L + plan.safety_stock)


# Work out how long current stock lasts, how much to order, and label each pair.

# weeks of cover = stock / weekly sales  (e.g. 6 bags, selling 3 a week -> 2.0 weeks)
plan["weeks_of_cover"] = (plan.on_hand_end / d).round(1)
# stock position = in store + already ordered
position = plan.on_hand_end + plan.on_order
# order up to: reorder point + 4 more weeks of sales (covers us until the next review)
order_up_to = plan.reorder_point + d * review_time
# only order if position has fallen to the reorder point or below; order the gap
plan["suggested_order_qty"] = np.where(position <= plan.reorder_point,
                                       np.ceil(order_up_to - position), 0).astype(int)

# Stockout risk (under 2 weeks) ; Overstock (over 12) ; else Healthy
plan["status"] = np.select(
    [plan.weeks_of_cover < stockout_time, plan.weeks_of_cover > overstock_time],
    ["Stockout risk", "Overstock"], "Healthy")

# tidy up: friendlier column names, rounding, column order
plan = plan.rename(columns={fc_col: "forecast_weekly", "on_hand_end": "on_hand"})
plan["forecast_weekly"] = plan.forecast_weekly.round(2)
plan["sd_weekly"] = plan.sd_weekly.round(2)
cols = ["store_id", "city", "region", "tier", "sku_id", "category", "on_hand", "on_order",
        "forecast_weekly", "sd_weekly", "lead_time_wk", "safety_stock", "reorder_point",
        "weeks_of_cover", "suggested_order_qty", "status"]
# sort: stockout risk first, then overstock, then healthy; lowest cover first within each
status_order = {"Stockout risk": 0, "Overstock": 1, "Healthy": 2}
plan = plan[cols].sort_values(
    ["status", "weeks_of_cover"],
    key=lambda s: s.map(status_order) if s.name == "status" else s)


# ===========================================================================
# SAVE THE FILES
# ==============================================================================
# >>> BEFORE: everything lives only in Python's memory.
#     We write it to disk for Power BI / Tableau and for planners.

stores.to_csv(DATA / "stores.csv", index=False)
products.drop(columns="base_demand").to_csv(DATA / "products.csv", index=False)
weekly.to_csv(DATA / "weekly.csv", index=False)
accuracy.round(3).to_csv(OUT / "forecast_accuracy.csv", index=False)

with pd.ExcelWriter(OUT / "reorder_plan.xlsx") as xw:
    plan.to_excel(xw, sheet_name="Reorder plan", index=False)
    accuracy.round(3).to_excel(xw, sheet_name="Forecast accuracy", index=False)

#===========================================================================
# HEADLINE NUMBERS FOR README
# ==============================================================================

print("\nForecast accuracy on the test data:")
print(accuracy.round(3).to_string(index=False))
print(f"\nPlan built with: {fc_col}")

print("\nStatus of store x product pairs today:")
print(plan.status.value_counts().to_string())

# year 2 rows only
yr2 = weekly[weekly.week >= 52]
lost = yr2.lost_sales_eur.sum()
# share of store x product x weeks that were stockouts
print(f"\nYear 2 stockout rate: {yr2.stockout.mean():.1%} of store-product-weeks")
# lost sales as a share of total demand (sold + lost)
print(f"Year 2 lost sales: EUR {lost:,.0f} ({lost / (yr2.net_revenue_eur.sum() + lost):.1%} of demand)")
print(f"Stock value today: EUR {weekly[weekly.week == last].stock_value_eur.sum():,.0f}")
print(f"Units to order now: {plan.suggested_order_qty.sum()} across "
      f"{(plan.suggested_order_qty > 0).sum()} store-product lines")
print(f"\nFiles saved to: {DATA} and {OUT}")

# Checks every number in the README's "Key findings" from the saved output files.
# Run from the Demand_Forecasting folder: python check_findings.py

from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
weekly = pd.read_csv(BASE / "data" / "weekly.csv")
plan = pd.read_excel(BASE / "outputs" / "reorder_plan.xlsx", sheet_name="Reorder plan")

# ---------------------------------------------------------------------------
# Finding 1: lost demand and stockout rate in year 2 (weeks 52-103)
# ---------------------------------------------------------------------------
yr2 = weekly[weekly.week >= 52]
lost = yr2.lost_sales_eur.sum()
revenue = yr2.net_revenue_eur.sum()
print("FINDING 1: lost demand, year 2")
print(f"  rows in year 2:            {len(yr2):,}  (expect 190 pairs x 52 weeks = 9,880)")
print(f"  stockout weeks:            {yr2.stockout.sum():,}")
print(f"  stockout rate:             {yr2.stockout.mean():.1%}")
print(f"  lost sales:                EUR {lost:,.0f}")
print(f"  revenue:                   EUR {revenue:,.0f}")
print(f"  lost / (revenue + lost):   {lost / (revenue + lost):.1%}")
# cross-check in units, independent of prices
units_lost = (yr2.demand_true - yr2.units_sold).sum()
print(f"  units lost / units demanded: {units_lost:,} / {yr2.demand_true.sum():,} "
      f"= {units_lost / yr2.demand_true.sum():.1%}")

# ---------------------------------------------------------------------------
# Finding 2: stockout risk by region
# ---------------------------------------------------------------------------
print("\nFINDING 2: stockout risk by region")
by_region = pd.crosstab(plan.region, plan.status, margins=True)
print(by_region)
print((plan.status == "Stockout risk").groupby(plan.region).mean().map("{:.0%}".format))

# ---------------------------------------------------------------------------
# Finding 3: stockout risk by category
# ---------------------------------------------------------------------------
print("\nFINDING 3: stockout risk by category")
print(pd.crosstab(plan.category, plan.status, margins=True))
print((plan.status == "Stockout risk").groupby(plan.category).mean().map("{:.0%}".format))

# ---------------------------------------------------------------------------
# Finding 4: size of the reorder, and why
# ---------------------------------------------------------------------------
print("\nFINDING 4: reorder size")
print(f"  units to order:            {plan.suggested_order_qty.sum():,}")
print(f"  lines with an order:       {(plan.suggested_order_qty > 0).sum()} of {len(plan)}")

# (a) is stock low because of Christmas? total units on hand, last 14 weeks
stock = weekly.groupby(["week", "week_start"]).on_hand_end.sum().reset_index()
print("\n  (a) total units on hand across all boutiques:")
print(stock[stock.week >= 90].to_string(index=False))
print(f"  year-2 weekly average: {stock[stock.week >= 52].on_hand_end.mean():,.0f}")

# (b) how big is the buffer the lead times require?
print("\n  (b) what the policy asks for vs what exists:")
print(f"  sum of reorder points:     {plan.reorder_point.sum():,.0f}")
print(f"  sum of safety stock:       {plan.safety_stock.sum():,.0f}")
print(f"  on hand + on order:        {(plan.on_hand + plan.on_order).sum():,}")
print(f"  median weeks of cover:     {plan.weeks_of_cover.median()}")

