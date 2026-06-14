# Delivery-Time-Prediction
# 🛵 Food Delivery Time Prediction

A machine learning project that predicts food delivery times using a tuned **Random Forest Regressor**. The model takes real-world factors like distance, weather, traffic, and courier experience to estimate how long a delivery will take. Includes a fully interactive **Streamlit web app** for live predictions.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Model Performance](#model-performance)
- [Feature Importance](#feature-importance)
- [Streamlit App](#streamlit-app)
- [Technologies Used](#technologies-used)

---

## Overview

Accurate delivery time prediction is critical for food delivery platforms — it sets customer expectations, improves satisfaction, and helps operations teams allocate couriers efficiently. This project builds an end-to-end ML pipeline:

1. Exploratory Data Analysis (EDA) and statistical feature selection
2. Data preprocessing with label encoding and imputation
3. Baseline Decision Tree model followed by a tuned Random Forest
4. Hyperparameter optimization using `RandomizedSearchCV`
5. Interactive Streamlit UI for real-time inference

---

## Dataset

**File:** `Food_Delivery_Times.csv`

| Property | Value |
|---|---|
| Total records | 1,000 orders |
| Features | 8 (including Order ID) |
| Target variable | `Delivery_Time_min` |
| Target range | 8 – 153 minutes |
| Average delivery time | ~57 minutes |
| Missing values | Weather, Traffic Level, Time of Day, Courier Experience |

### Feature Descriptions

| Feature | Type | Description |
|---|---|---|
| `Order_ID` | Integer | Unique order identifier (dropped before training) |
| `Distance_km` | Float | Distance from restaurant to delivery location (km) |
| `Weather` | Categorical | Weather condition — Clear, Rainy, Foggy, Windy, Snowy |
| `Traffic_Level` | Categorical | Traffic intensity — Low, Medium, High |
| `Time_of_Day` | Categorical | Order time slot — Morning, Afternoon, Evening, Night |
| `Vehicle_Type` | Categorical | Delivery vehicle — Bike, Scooter, Car |
| `Preparation_Time_min` | Integer | Time for the restaurant to prepare the order (min) |
| `Courier_Experience_yrs` | Float | Years of experience of the assigned courier |
| `Delivery_Time_min` | Integer | **Target** — total delivery time in minutes |

---

## Project Structure

```
food-delivery-prediction/
│
├── Food_Delivery_Times.csv           # Raw dataset
├── time_prediction.ipynb             # Main analysis and training notebook
├── best_random_forest_model.pkl      # Saved tuned Random Forest model
├── label_encoders.pkl                # Saved LabelEncoders for categorical features
├── app.py                            # Streamlit web application
└── README.md                         # Project documentation
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Steps

```bash
# 1. Clone or download this repository
git clone https://github.com/your-username/food-delivery-prediction.git
cd food-delivery-prediction

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### requirements.txt

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
scipy>=1.10.0
statsmodels>=0.14.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## Usage

### Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### Run the Notebook

Open `time_prediction.ipynb` in Jupyter or VS Code to walk through the full analysis, from EDA to model training and evaluation.

```bash
jupyter notebook time_prediction.ipynb
```

---

## Methodology

### 1. Exploratory Data Analysis

- Distribution analysis of `Delivery_Time_min` (right-skewed with outliers)
- Scatter plots of numerical features vs. delivery time
- Box plots of categorical features vs. delivery time
- Pearson correlation heatmap for numerical features

### 2. Statistical Feature Selection

**Pearson Correlation (numerical features):**

| Feature | Correlation | p-value | Significant? |
|---|---|---|---|
| `Distance_km` | 0.781 | < 0.05 | ✅ Yes |
| `Preparation_Time_min` | 0.307 | < 0.05 | ✅ Yes |
| `Courier_Experience_yrs` | -0.089 | < 0.05 | ✅ Yes |

**ANOVA Test (categorical features):**

| Feature | p-value | Significant? |
|---|---|---|
| `Weather` | < 0.05 | ✅ Yes |
| `Traffic_Level` | < 0.05 | ✅ Yes |
| `Time_of_Day` | 0.792 | ❌ No |
| `Vehicle_Type` | 0.555 | ❌ No |

> Note: All features were retained for training as they were part of the original feature set, but the ANOVA results inform us that `Time_of_Day` and `Vehicle_Type` have limited predictive value.

### 3. Preprocessing

- **Missing value imputation:** Categorical columns filled with mode; `Courier_Experience_yrs` filled with median (applied after train-test split to prevent data leakage)
- **Label Encoding:** Applied to `Weather`, `Traffic_Level`, `Time_of_Day`, and `Vehicle_Type`
- **Train-Test Split:** 80% training / 20% testing (`random_state=42`)

### 4. Model Building

Two models were trained and compared:

| Model | MSE | R² |
|---|---|---|
| Decision Tree Regressor (default) | Higher | Lower |
| Random Forest Regressor (default) | Lower | Higher |

### 5. Hyperparameter Tuning

`RandomizedSearchCV` was used to tune the Random Forest with the following search space:

```python
param_grid = {
    'n_estimators': randint(100, 501),
    'max_features': ['sqrt', 'log2', 1.0],
    'max_depth': list(np.arange(10, 51)) + [None],
    ...
}
```

**Best model:** `n_estimators = 311`

---

## Model Performance

Evaluated on the held-out test set (200 samples):

| Metric | Value |
|---|---|
| **R² Score** | **0.7854** |
| **MAE** | **7.14 minutes** |
| **RMSE** | **9.81 minutes** |
| Train size | 800 samples |
| Test size | 200 samples |

The model explains ~78.5% of the variance in delivery times, with a mean absolute error of about 7 minutes.

---

## Feature Importance

The trained Random Forest ranked features by importance as follows:

| Rank | Feature | Importance |
|---|---|---|
| 1 | `Distance_km` | 60.41% |
| 2 | `Preparation_Time_min` | 16.53% |
| 3 | `Courier_Experience_yrs` | 7.22% |
| 4 | `Weather` | 4.99% |
| 5 | `Traffic_Level` | 4.42% |
| 6 | `Time_of_Day` | 3.65% |
| 7 | `Vehicle_Type` | 2.79% |

`Distance_km` is the dominant predictor, accounting for over 60% of the model's decision-making — consistent with both the Pearson correlation (0.781) and domain intuition.

---

## Streamlit App

The `app.py` file provides an interactive web interface with:

- **Live input form** — sliders for numerical features, dropdowns for categorical features
- **Real-time order summary** — updates as inputs change
- **Prediction output** — estimated delivery time with a 10th–90th percentile range derived from individual tree predictions
- **Contextual insight** — compares the prediction to the dataset average
- **Feature importance sidebar** — visual bar chart of all 7 features
- **Dataset statistics** — total orders, average time, max distance at a glance

---

## Technologies Used

| Tool / Library | Purpose |
|---|---|
| Python 3.x | Core programming language |
| pandas | Data loading, manipulation, and preprocessing |
| numpy | Numerical operations |
| scikit-learn | ML models, preprocessing, evaluation, and tuning |
| scipy / statsmodels | Statistical testing (Pearson, ANOVA) |
| matplotlib / seaborn | EDA visualizations |
| Streamlit | Interactive web application |
| pickle | Model and encoder serialization |

---

## Author

**Dushyant**

---

*Built as part of a data science portfolio project demonstrating end-to-end ML workflow — from raw data to a deployed prediction interface.*
