# Luxury Boutique Demand Forecasting & Replenishment
<p align="center">
  <img src="assets/demand-terminal.svg" alt="Retro terminal summarising the project: ML forecast WAPE 0.679 vs 0.724 baseline, 45 of 190 store–product pairs at stockout risk, 6,041 units to reorder" width="466">
</p>
A simulated dataset of weekly stock for a fictional luxury brand with 10 boutiques and 19 products across various regions. With this independent project I tried to forecast weekly demand, train a ML model (with selected features) and compare it with a MA baseline, calculate safety stock and reorder points and draw attention to the store–product pairs that are overstocked or are at a risk of stockout. 

> **All data is simulated with the help of LLM.** No real company data is used. Every assumption in the script can be changed.

---

## Main Business Question❓

Luxury boutiques/stores carry small quantities of expensive goods to maintain exclusivity, and the ateliers may take ~ 6 to 12 weeks to deliver. Not enough goods could lead to loss of potential revenue and too much could also be a problem: discounting goods that are stuck could diminish perceived luxury value. 

**For each boutique and product: how much stock helps cover demand while waiting for next delivery, and which items need to be ordered now?**

---

## Results

### Forecast accuracy (Testing Data Period: 2025-06-30 to 2025-12-22, 26 weeks)

| Method | WAPE | Bias |
| --- | --- | --- |
| 8-week moving average (baseline) | 0.724 | −2.8% |
| Gradient boosting, Poisson loss | **0.679** | +1.8% |

ML gradient-boosting forecasting model reduced forecast error by 4.5 pp and was used to craft the reorder plan. Error for both methods remain high due to small noisy dataset simulated at random (but this is expected).

### "Today's" Stock health 

| Status | Store–product pairs | Share |
| --- | --- | --- |
| Healthy (2–12 weeks of cover) | 124 | 65% |
| Stockout risk (under 2 weeks) | 45 | 24% |
| Overstock (over 12 weeks) | 21 | 11% |

### Key findings

- **With the naive planner assumption, 9.3% of demand is lost.** In year 2, 7.6% of store–product weeks ran out of stock, ~ €3.73M in lost sales.
- **APAC under-stocked relative to EMEA.** 29% of APAC pairs (33 of 114) are at risk of stockout, against 16% in EMEA. APAC carries the extra handbag demand and the Chinese New Year peak (under assumptions made when simulating dataset).
- **Handbags and handmade leather goods are most exposed.** 28% of handbag pairs and 29% of leather-goods pairs are at stockout risk, against 13–18% for shoes and accessories.
- **Reorder plan suggests 6041 units which seems big.** But this is because "today" falls straight after the December Christmas peak, leaving lowered stock, and considering lead times of 6 to 12 weeks a large buffer is needed.

---

### Forecasting

- **Baseline:** average units sold over the past 8 weeks, per store and product.
- **ML model:** one `HistGradientBoostingRegressor` (scikit-learn, Poisson loss) trained across store–product pairs.
  - **Features:** sales 1, 2, 4, 8 and 52 weeks earlier; 4 and 8-week rolling mean; 8-week rolling standard deviation; week of year; December and CNY periods; store tier, region, category, price.
- **Scoring:**
  - WAPE = Σ|actual − forecast| ÷ Σ actual. It handles zero-sale weeks, unlike MAPE.
  - Bias = Σ(forecast − actual) ÷ Σ actual.

### Stock policy

```
Safety stock   = z × √( L × σ_d²  +  d² × σ_L² )
Reorder point  = d × L + safety stock
Order up to    = reorder point + d × 4   (covers the next 4-week review period)

```

where `d` = forecast weekly demand ; `L` = lead time (weeks) ; `σ_d` = standard deviation of weekly sales (last 26 weeks) ; `σ_L` = lead-time standard deviation (1.5 weeks) ;`z` = service level by tier (A 98% → 2.05, B 95% → 1.65, C 90% → 1.28)

Suggestion for order given if stock on hand plus stock on order is at or below the reorder point.

**Example: Milan, handbag** : (tier A, forecast 4.34 a week, σ_d 3.13, lead time 9 weeks)
- Safety stock: 24 units
- Reorder point: 64 units
- On hand 0 and on order 14 → 0 weeks of cover → **Stockout risk**, suggested order **68 units**

---

## Limitations

- **Simulated data:** We tried to make the patterns (seasonality, lumpy demand, lead times) realistic by design. But results are (obviously) not going to be realistic to what a real brand might have to do.
- **True demand not captured:** Both forecasts train from units sold, which might understate true demand (e.g. maybe a product is trending currently that was not in the past). There might be under-forecasting present.
- **Overinflated snapshot:** reorder plan is built using the week after Christmas, which inflates order quantities.

---

**Tools:** Python (pandas, NumPy, scikit-learn)
