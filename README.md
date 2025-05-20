# Flight-Delay Prediction & Mitigation  
Predicting U.S. domestic flight arrival delays with an early-warning model and a root-cause OLS benchmark

## 1. Project Overview
Commercial aviation suffers from cascading delays that cost airlines **$30 billion+ annually**.  
This project builds a **two-tier predictive-analytics pipeline** that

* **Flags at-risk flights _before push-back_** (actionable for gate, crew and reroute decisions)  
* **Explains root causes** once real-time delay components are known  

The work forms part of my Drexel University master’s portfolio and demonstrates full-stack data-science skills: data wrangling, feature engineering, model selection, validation and business storytelling.

## 2. Dataset
| Source | Kaggle “Flight Delays and Cancellations” (2015 sample) |
| ------ | ----------------------------------------------------- |
| Rows   | **5,860** flights (after cleaning)                    |
| Columns| 31 raw + engineered features                          |
| Target | `ARRIVAL_DELAY` (minutes)                             |

### 2.1 Key Engineered Features
* **Scheduling:** hour-of-day, day-of-week, month, holiday flag  
* **Route:** carrier, origin, destination, distance bands  
* **Delay components:** `DEPARTURE_DELAY`, `AIR_SYSTEM_DELAY`, etc. (for root-cause model only)

## 3. Repository Structure
```
.
├── notebooks/
│   └── 01_flight_delay_pipeline.ipynb   # full EDA + modelling
├── data/
│   ├── flights.csv                      # cleaned sample (5,860 rows)
│   └── raw/                             # original Kaggle extract
├── models/
│   └── trained_model.joblib             # saved OLS diagnostic model
├── reports/
│   └── Predictive_modeling_flight_delays.pdf # 10-slide summary deck
└── README.md
```

## 4. Modelling Approach
| Tier | Features available | Algorithms compared | Best metric |
|------|--------------------|---------------------|-------------|
| **Early-warning (pre-departure)** | Carrier, route, sched-hour, day-of-week, distance | Linear Regression, Random Forest, Gradient Boosting | **MAE ≈ 22 min, ROC-AUC 0.57 (10-fold CV)** |
| **Root-cause OLS** | Early features **+** real-time delay components | statsmodels OLS | **R² = 0.99997, MSE = 0.10** |

*10-fold cross-validation is **blocked by flight date** to prevent temporal leakage.*

## 5. Results & Insights
* **Early-warning model** detects ~60% of >15 min delays with a 40% false-positive rate—enough lead time (~2 h) for crew/gate re-sequencing.  
* **Root-cause OLS** confirms that **departure delay explains 74%** of arrival delay variance, with weather and air-traffic components adding most of the remainder.  
* A hypothetical **5 min average reduction** on flagged flights saves ~\$5 M per 10 K flights (FAA delay-cost guideline \$75/min).

## 6. Quick Start
```bash
# 1. Clone repo & create env
git clone https://github.com/<your-handle>/flight-delay-prediction.git
cd flight-delay-prediction
conda env create -f environment.yml   # or pip install -r requirements.txt
conda activate flight-delays

# 2. Launch notebook
jupyter lab notebooks/01_flight_delay_pipeline.ipynb
```

## 7. Reproduce Results
1. Download the full 2015 BTS On-Time dataset (optional, ~600 MB)  
2. Run `python src/prepare_data.py --year 2015` to regenerate `flights.csv`  
3. Execute the notebook or `python src/train.py --tier early`  

## 8. Next Steps
* Integrate **NOAA weather forecasts** to improve early-warning ROC-AUC > 0.70  
* Deploy as a **REST API** with FastAPI + Docker for real-time scoring  
* Add **XGBoost & CatBoost** baselines; compare SHAP feature importance  
* Scale to multi-year BTS data (~7 M rows) for production realism  

## 9. Tech Stack
Python • pandas • scikit-learn • statsmodels • joblib • Matplotlib/Seaborn • Jupyter
