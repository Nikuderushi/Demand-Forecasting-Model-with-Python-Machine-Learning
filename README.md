# 📦 Demand Forecasting App

A machine learning web application built with **Streamlit** and **XGBoost** that predicts product demand based on pricing, inventory, promotions, and category data.

---

## 🚀 Features

- Predict product demand using a trained XGBoost regression model
- Interactive input controls for all model features
- Prediction validation — compare your prediction against actual demand with error metrics
- Clean, minimal UI powered by Streamlit

---

## 🗂️ Project Structure

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

## 📊 Dataset

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

📂 Dataset source: [GitHub — DemandForecastingDataset](https://github.com/Onurbltc/DemandForecastingDataset/blob/main/demand_forecasting.csv)

---

## 🧠 Model

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

## ⚙️ Setup & Installation

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

## 🖥️ Usage

1. Enter the product details in the **Input Features** section
2. Click **Predict Demand** to get the model's forecast
3. Optionally, enter the **Actual Demand** in the validation section to see how accurate the prediction was

---

## 📈 Notebooks

| Notebook | Description |
|---|---|
| `Analysis.ipynb` | Exploratory data analysis, visualizations, feature engineering |
| `machinelearning.ipynb` | Model training, hyperparameter tuning, evaluation, saving artifacts |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas / NumPy | Data manipulation |
| Scikit-learn | Preprocessing & model evaluation |
| XGBoost | Gradient boosting regression model |
| Streamlit | Web app framework |
| Pickle | Model serialization |

---

## 📄 License

This project is for educational purposes. Dataset credit: [Onurbltc](https://github.com/Onurbltc/DemandForecastingDataset).
