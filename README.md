# Mama-T-Restaurant-model
## BY Japhet Ujile
Building a regression machine learning model to predict the amount of tip (in Nigerian Naira)

Here is the content for your GitHub README file, formatted in Markdown:

````markdown
# 🍜 Mama Tee ML
**By Japhet Ujile**

This repository contains an exploratory data analysis and regression modeling project aimed at predicting restaurant tip amounts using various machine learning models.

---

## 💻 Project Setup and Data Loading

The project uses the `tips.csv` dataset, which contains information about dining bills, tips, and customer demographics.

```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load the dataset
df = pd.read_csv('data/tips.csv')
````

-----

## 📊 Exploratory Data Analysis (EDA)

### Data Snapshot

Viewing the first few rows of the dataset:

```python
df.head()
```

| | total\_bill | tip | gender | smoker | day | time | size |
|---:|:---|:---|:---|:---|:---|:---|:---|
| 0 | 2125.50 | 360.79 | Male | No | Thur | Lunch | 1 |
| 1 | 2727.18 | 259.42 | Female | No | Sun | Dinner | 5 |
| 2 | 1066.02 | 274.68 | Female | Yes | Thur | Dinner | 4 |
| 3 | 3493.45 | 337.90 | Female | No | Sun | Dinner | 1 |
| 4 | 3470.56 | 567.89 | Male | Yes | Sun | Lunch | 6 |

### Dataset Shape and Structure

```python
df.shape
```

```
(744, 7)
```

```python
df.columns
```

```
Index(['total_bill', 'tip', 'gender', 'smoker', 'day', 'time', 'size'], dtype='object')
```

### Missing Values Check

```python
df.isnull().sum()
```

```
total_bill    0
tip           0
gender        0
smoker        0
day           0
time          0
size          0
dtype: int64
```

**Conclusion:** No missing values are present in the dataset.

### Data Types and Information

```python
df.info()
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 744 entries, 0 to 743
Data columns (total 7 columns):
 #   Column      Non-Null Count  Dtype  
---  ------      --------------  -----  
 0   total_bill  744 non-null    float64
 1   tip         744 non-null    float64
 2   gender      744 non-null    object 
 3   smoker      744 non-null    object 
 4   day         744 non-null    object 
 5   time        744 non-null    object 
 6   size        744 non-null    int64  
dtypes: float64(2), int64(1), object(4)
memory usage: 40.8+ KB
```

### Descriptive Statistics

```python
df.describe()
```

| | total\_bill | tip | size |
|:---|:---|:---|:---|
| count | 744.000000 | 744.000000 | 744.000000 |
| mean | 2165.006640 | 325.948091 | 3.180108 |
| std | 954.248806 | 148.778225 | 1.532890 |
| min | 44.690000 | 0.000000 | 1.000000 |
| 25% | 1499.022500 | 218.000000 | 2.000000 |
| 50% | 2102.610000 | 320.460000 | 3.000000 |
| 75% | 2743.802500 | 415.562500 | 4.000000 |
| max | 5538.290000 | 1090.000000 | 6.000000 |

-----

## 🛠️ Data Preprocessing

Separating features (`x`) and target variable (`y`):

```python
y = df['tip']
x = df.drop('tip', axis=1)
```

### Feature Engineering (One-Hot Encoding)

Converting categorical features into a format suitable for machine learning models:

```python
# Convert categorical features using OneHotEncoding
x = pd.get_dummies(x, drop_first=True)
x
```

| | total\_bill | size | gender\_Male | smoker\_Yes | day\_Mon | day\_Sat | day\_Sun | day\_Thur | day\_Tues | day\_Wed | time\_Lunch |
|---:|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| 0 | 2125.50 | 1 | True | False | False | False | True | False | False | True |
| 1 | 2727.18 | 5 | False | False | False | True | False | False | False | False |
| 2 | 1066.02 | 4 | False | True | False | False | True | False | False | False |
| 3 | 3493.45 | 1 | False | False | False | True | False | False | False | False |
| 4 | 3470.56 | 6 | True | True | False | False | True | False | False | True |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 739 | 3164.27 | 3 | True | False | True | False | False | False | False | False |
| 740 | 2962.62 | 2 | False | True | True | False | False | False | False | False |
| 741 | 2471.03 | 2 | True | True | True | False | False | False | False | False |
| 742 | 1942.38 | 2 | True | False | True | False | False | False | False | False |
| 743 | 2047.02 | 2 | False | False | False | False | True | False | False | False |

### Train-Test Split

Splitting the data into training (70%) and testing (30%) sets:

```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
```

-----

## 🧠 Model Training and Evaluation

Three regression models are trained and evaluated: **Linear Regression**, **Decision Tree**, and **Random Forest**.

### 1\. Linear Regression

```python
reg = LinearRegression()
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)

# Evaluation
print("MAE", mean_absolute_error(y_test, y_pred))
```

```
MAE 118.89702992837499
```

### 2\. Decision Tree Regressor

```python
# Decision Tree
dt = DecisionTreeRegressor(random_state=42)
dt.fit(x_train, y_train)
y_pred_dt = dt.predict(x_test)
```

### 3\. Random Forest Regressor

```python
# Random Forest
rf = RandomForestRegressor(random_state=42, n_estimators=100)
rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)

# Evaluation
print("MAE", mean_absolute_error(y_test, y_pred_rf))
```

```
MAE 120.50495000000001
```

-----

## 🏆 Model Comparison

The performance of the three models is compared using Mean Absolute Error (MAE), Root Mean Square Error (RMSE), and the R-squared ($R^2$) score.

```python
results = pd.DataFrame({
    "Model": ["Linear Regression", "Decision Tree", "Random Forest"],
    "MAE": [
        mean_absolute_error(y_test, y_pred),
        mean_absolute_error(y_test, y_pred_dt),
        mean_absolute_error(y_test, y_pred_rf)
    ],
    "RMSE": [
        np.sqrt(mean_squared_error(y_test, y_pred)),
        np.sqrt(mean_squared_error(y_test, y_pred_dt)),
        np.sqrt(mean_squared_error(y_test, y_pred_rf))
    ],
    "R2 Score": [
        r2_score(y_test, y_pred),
        r2_score(y_test, y_pred_dt),
        r2_score(y_test, y_pred_rf)
    ]})

print("\n===== Model Comparison Table =====")
print(results)
```

```
===== Model Comparison Table =====
               Model         MAE        RMSE  R2 Score
0  Linear Regression  118.897030  151.220741  0.015844
1      Decision Tree  149.524643  199.897517 -0.719713
2      Random Forest  120.504950  154.852847 -0.03199
```

### Analysis of Results

  * **Linear Regression** performed the best among the three models with the lowest **MAE** (118.90) and the highest **$R^2$ Score** (0.016).
  * All models show relatively low **$R^2$ scores**, suggesting that the model is **not a strong predictor** of the `tip` variable based on the current features and data. The relationship between the input features and the target variable is likely weak or highly non-linear.
  * The negative **$R^2$ scores** for the Decision Tree and Random Forest indicate that these models performed worse than simply predicting the mean of the tips for all cases.

-----

## ⏭️ Next Steps

Further improvements could involve:

1.  Feature engineering (e.g., creating a tip percentage feature).
2.  Exploring other models (e.g., Ridge, Lasso, XGBoost).
3.  Hyperparameter tuning to optimize the Decision Tree and Random Forest models.


## 👨🏽‍💻 Author

*Japhet Ujile*
📧 [assistant.rawlings@gmail.com](mailto:assistant.rawlings@gmail.com)
🌐 [GitHub](https://github.com/assistantrawlings-lgtm) | [LinkedIn](https://www.linkedin.com/in/japhet-ujile-838442148?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app])

```
