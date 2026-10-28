# Integrated Retail Analytics for Store Optimization and Demand Forecasting

Weekly sales of a retail chain with 45 stores and 81 departments (Feb 2010 to Oct 2012), analysed to find unusual sales weeks, group the stores into segments, find departments that sell together, forecast weekly demand and turn the results into inventory and marketing strategies.

Everything is in one notebook that follows the provided ML submission template:
**[Retail_Analytics_Store_Optimization_Demand_Forecasting.ipynb](Retail_Analytics_Store_Optimization_Demand_Forecasting.ipynb)**

## Repository layout

```
retail-analytics-capstone/
├── Retail_Analytics_Store_Optimization_Demand_Forecasting.ipynb   the whole project, run top to bottom
├── data/
│   ├── sales_data.csv        weekly sales per store and department (421,570 rows)
│   ├── features_data.csv     weekly store features: temperature, fuel, markdowns, CPI, unemployment
│   └── stores_data.csv       type and size of the 45 stores
├── models/                   saved models (created by the notebook)
│   ├── lgbm_long_term_forecaster.joblib
│   ├── store_segmentation.joblib
│   └── isolation_forest_anomaly.joblib
├── outputs/
│   ├── figures/              every chart of the notebook as a png
│   ├── anomalies.csv                    store-dept weeks flagged by the Isolation Forest
│   ├── store_segments.csv               store features, segment and silhouette
│   ├── association_rules.csv            department association rules
│   ├── department_abc_xyz.csv           inventory class of each department
│   └── forecast_nov2012_jul2013.csv     39-week forecast per store and department
├── requirements.txt
└── README.md
```

The csv files in `data/` are the files provided for the project, renamed without spaces.

## Project components

| # | Component | Method | Where in the notebook |
|---|---|---|---|
| 1 | Anomaly detection | Isolation Forest vs z-score rule, scored on injected anomalies | Charts 5-6, ML Model 1 |
| 2 | Time-based anomalies and seasonality | STL decomposition, holiday lift, per-store STL residuals | Charts 1-4 and 6 |
| 3 | Preprocessing and feature engineering | merge, markdown imputation, lags, holiday features, store profile | Sections 3 and 6 |
| 4 | Store segmentation | KMeans (compared with Ward and Gaussian mixture) | ML Model 2 |
| 5 | Market basket analysis | Apriori on store-weeks of departments with a high sales share | Section 7.1 |
| 6 | Demand forecasting | seasonal naive, Holt-Winters, LightGBM long-term and short-term | ML Model 3 |
| 7 | External factors | ablation, permutation importance, partial dependence, regional growth | Section 7.2 |
| 8 | Personalization and inventory | markdown response per segment, department mix, ABC-XYZ | Section 7.3 |
| 9 | Segmentation quality | silhouette, Davies-Bouldin, bootstrap stability, homogeneity and separation | ML Model 2 part 5 |
| 10 | Strategy and real-world challenges | | Section 7.4 and Conclusion |

## Main results

- **Season:** sales are flat at about 46M $ a week. Thanksgiving week sells 43% more and the week before Christmas 70% more. That week is not flagged as a holiday in the data, while the flagged Christmas week is below normal. Easter moves between years, so it got its own flag, which also lowered the forecast error.
- **Time-based anomalies:** the biggest ones are Easter (it moves between years) and the shift of Christmas Eve between weeks, plus a few store-specific events.
- **Anomaly detection:** the tuned Isolation Forest reaches F1 0.552 on injected anomalies, against 0.480 for the best z-score rule. It flags 1.5% of store-department weeks. Seasonal anomalies and ones that come back every year (Easter, Valentine's Day) are kept as real demand, and only one-off anomalies are cleaned.
- **Segments:** 4 store segments with silhouette 0.415 and bootstrap stability 0.82, up from 0.198 and 0.52 with all 13 features:
  1. large stores in high-CPI regions
  2. large stores in low-CPI regions
  3. mid-size promotion-driven stores
  4. small steady stores
- **Market basket:** 17 rules forming 5 department groups (24/33/31, 5/55/72, 90/91/92, 18/1 and 16/56) with lift 2.5 to 4.2.
- **Forecasting** (test = last 39 weeks):

| Model | Test WMAE |
|---|---|
| Seasonal naive (same week last year) | 1,807 |
| Holt-Winters, per series (full-history series only) | 1,803 (seasonal naive on the same series: 1,946) |
| LightGBM long-term, tuned (39 weeks ahead) | 1,533 |
| LightGBM short-term (1 week ahead) | 1,300 |

  WMAE is the mean absolute error per store-department week in dollars, with holiday weeks counted 5 times.
- **External factors:** together they improve the long-term forecast by 5.4%. Temperature and unemployment matter most, and stores whose regional unemployment fell more also grew more (Spearman -0.33). The biggest markdowns did not bring extra growth, except in the promotion-driven segment.

## Running the notebook

On **Google Colab**: open the notebook and run all cells. If the `data/` folder is not next to the notebook, it reads the csv files from this GitHub repo. If `mlxtend` is not installed, the first code cell installs it.

Locally (Python 3.12):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook Retail_Analytics_Store_Optimization_Demand_Forecasting.ipynb
```

On macOS, LightGBM also needs OpenMP: `brew install libomp`.

A full run takes about 12 minutes on a 12-core laptop. Most of that time is the Holt-Winters models (2,660 series), the forecasting cross-validation and the ablation. The results are reproducible: every model uses a fixed random seed and LightGBM runs in deterministic mode.

## Data notes

- Dates in the csv files are day/month/year.
- Markdowns were only recorded from 11 Nov 2011. Before that, the empty values mean "not recorded", and after that they mean "no markdown". The notebook fills both with 0 and keeps a `MarkDown_Recorded` flag.
- CPI and unemployment are missing only for May to July 2013, which are future weeks.
- 1,285 store-department weeks have negative sales (returns). They are set to 0 for modelling.
- Department names are not given, so the departments are referred to by number.
