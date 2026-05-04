# Demand Forecasting App

A machine learning web application built with **Streamlit** and **XGBoost** that predicts product demand based on pricing, inventory, promotions, and category data.

---

## Features

- Predict product demand using a trained XGBoost regression model
- Interactive input controls for all model features
- Prediction validation — compare your prediction against actual demand with error metrics
- Clean, minimal UI powered by Streamlit

---

## Project Structure

```
demand-forecasting/
│
├── app.py                        # Streamlit web application
├── demand_forecasting.csv        # Raw dataset
├── xgboost_demand_model.pkl      # Trained XGBoost model
├── label_encoder.pkl             # Fitted LabelEncoder for categorical features
│
├── Analysis.ipynb                # EDA and data visualization notebook
├── machinelearning.ipynb         # Model training and evaluation notebook
│
└── README.md
```

---

## Dataset

The dataset contains historical retail demand data with the following columns:

| Column | Description |
|---|---|
| `Date` | Date of record |
| `Store ID` | Unique store identifier |
| `Product ID` | Unique product identifier |
| `Category` | Product category (Clothing, Electronics, Furniture, Groceries, Toys) |
| `Region` | Geographic region |
| `Inventory Level` | Stock available at the time |
| `Units Sold` | Actual units sold |
| `Price` | Listed price |
| `Discount` | Discount percentage applied |
| `Promotion` | Whether a promotion was active (0/1) |
| `Competitor Pricing` | Competitor's price for the same product |
| `Seasonality` | Season indicator |
| `Epidemic` | Epidemic impact flag |
| `Demand` | **Target variable** — demand units |

 Dataset source: [GitHub — DemandForecastingDataset](https://github.com/Onurbltc/DemandForecastingDataset/blob/main/demand_forecasting.csv)

---

## Model

- **Algorithm**: XGBoost Regressor (`reg:squarederror`)
- **Tuning**: RandomizedSearchCV (25 iterations, 3-fold CV)
- **Features used**:
  - `Price`
  - `Discount`
  - `Inventory Level`
  - `Promotion`
  - `Competitor Pricing`
  - `Category` *(label encoded)*

---

## Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/demand-forecasting.git
cd demand-forecasting
```

### 2. Install dependencies

```bash
pip install streamlit pandas numpy xgboost scikit-learn
```

### 3. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## Usage

1. Enter the product details in the **Input Features** section
2. Click **Predict Demand** to get the model's forecast
3. Optionally, enter the **Actual Demand** in the validation section to see how accurate the prediction was

---

## Notebooks

| Notebook | Description |
|---|---|
| `Analysis.ipynb` | Exploratory data analysis, visualizations, feature engineering |
| `machinelearning.ipynb` | Model training, hyperparameter tuning, evaluation, saving artifacts |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas / NumPy | Data manipulation |
| Scikit-learn | Preprocessing & model evaluation |
| XGBoost | Gradient boosting regression model |
| Streamlit | Web app framework |
| Pickle | Model serialization |

---

## Model Performance

Evaluated on a held-out test set of **15,200 samples** (20% of the full dataset), using a fixed random seed for reproducibility.

| Metric | Value | What it means |
|---|---|---|
| **MAE** | 24.79 units | On average, predictions are off by ~25 units |
| **RMSE** | 32.52 units | Penalises larger errors more heavily than MAE |
| **R² Score** | 0.52 | The model explains 52% of variance in demand |
| **MAPE** | 35.61% | Average percentage error per prediction |

**Context:** Demand ranges from 4 to 357 units with a mean of ~104 units, so an MAE of ~25 units represents roughly a 24% average deviation from the mean — reasonable given that only 6 of the 13 available features were used for training. Features with strong predictive potential — `Seasonality`, `Region`, `Weather Condition`, and `Epidemic` — were intentionally excluded from the current version and represent a clear path to improving these metrics.

---

## Financial Services Applications

The demand forecasting methodology demonstrated in this project transfers directly to a wide range of predictive problems in financial services. In retail banking, the same gradient boosting pipeline can be applied to **loan demand forecasting** — where a bank predicts how many personal, home, or business loan applications it will receive in a given period based on features like interest rates, promotional campaigns, competitor rates, and macroeconomic indicators. Similarly, **credit card spend prediction** uses an identical regression framework, where historical transaction patterns, customer segments, promotional offers, and seasonal effects replace product-level retail features. Insurance companies apply the same approach to **claims volume forecasting**, predicting how many claims will be filed in a period using weather data, policy counts, and regional risk factors — enabling better reserve allocation and staffing. In each case, the core problem is identical to this project: a set of observable input features drives a continuous target variable, and XGBoost's ability to handle mixed feature types, non-linear relationships, and interactions makes it well-suited for all of them.

Perhaps the most high-stakes application in banking is **Non-Performing Asset (NPA) prediction** in a lending portfolio. Here, the target variable shifts from a demand count to a probability or volume of loans likely to default within a given window. Features mirror those in this project — loan amount (analogous to price), borrower credit score (analogous to inventory level), interest rate spread over competitors (analogous to competitor pricing), and whether a restructuring scheme is active (analogous to promotion). The model training, hyperparameter tuning via RandomizedSearchCV, and evaluation pipeline built here are directly portable to that problem. The key difference is business impact: in retail, a poor demand forecast leads to stockouts or overstock; in lending, a poor NPA forecast leads to under-provisioning, regulatory risk, and balance sheet stress. This project demonstrates the foundational ML engineering skills — feature engineering, gradient boosting, serialisation, and deployment — that underpin these higher-stakes financial forecasting systems.

---

## License

This project is for educational purposes. Dataset credit: [Onurbltc](https://github.com/Onurbltc/DemandForecastingDataset).
