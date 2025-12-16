import pandas as pd
import MultiColumnLabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBRegressor
import pickle

df = pd.read_csv("/content/data.zip")

df.head()

df.tail()

df.info()

df.describe()

df.duplicated().sum()
df = df.drop_duplicates()

df.isnull().sum()

df.drop(['wip'],axis="columns",inplace=True)
df.head()

df["date"] = pd.to_datetime(df["date"])
df.date

df['department'].value_counts()

df['year']  = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_num'] = df['date'].dt.day

df.drop('date', axis=1, inplace=True)
df.head()

me=MultiColumnLabelEncoder.MultiColumnLabelEncoder()
df = me.fit_transform(df)

df.head()

selected_features = [
    "quarter",
    "department",
    "day",
    "team",
    "targeted_productivity",
    "smv",
    "over_time",
    "incentive",
    "idle_time",
    "idle_men",
    "no_of_style_change",
    "no_of_workers"
]

x = df[selected_features]
y = df["actual_productivity"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# LinearRegression

lr = LinearRegression()
model_lr = lr.fit(x_train, y_train)
print("Intercept: ", model_lr.intercept_)
print("Coef: ", model_lr.coef_)
Score = model_lr.score(x_test, y_test)
print("Score: ", Score)
# Score:  0.19272470603588843

lr_train_pred = model_lr.predict(x_train)
lr_test_pred = model_lr.predict(x_test)

lr_train_pred = lr_train_pred.reshape(-1, 1)
lr_test_pred = lr_test_pred.reshape(-1, 1)

# Random Forest Regressor
rfg = RandomForestRegressor(n_estimators = 1000, random_state = 42)
model_rfg = rfg.fit(x_train, y_train)
Score = model_rfg.score(x_train, y_train)
print("Score: ", Score)
# Score:  0.9302832770347732

# Xg Boost


xgb_model = XGBRegressor(n_estimators = 1000, learning_rate = 0.05, max_depth = 5,
                   subsample = 0.8, colsample_bytree = 0.8, random_state = 42)
model_xgb = xgb_model.fit(x_train, y_train)
Score = model_xgb.score(x_train, y_train)
print("Score: ", Score)
# Score:  0.9922283192479376


# Model for Linear Regression
# FIle Name = Employee_Performance_lr

with open ("Employee_Performance_lr", "wb") as f:
  pickle.dump(model_lr, f)

# pickle

# Model For Random_Forest
# File Name = Employee_Performance_rfg
# wb = Write Binary
with open("Employee_Performance_rfg", "wb") as f1:
  pickle.dump(model_rfg, f1)


# Model for xgboost
# FIle Name = Employee_Performance_xgb

with open("Employee_Performance_xgb", "wb") as f2:
  pickle.dump(model_xgb, f2)

# Loading the Model

with open("Employee_Performance_rfg", "rb") as f3:
  data_rfg = pickle.load(f3)

import pandas as pd

input_data = {
    'team': [8],
    'targeted_productivity': [0.75],
    'smv': [26.16],
    'over_time': [7080],
    'incentive': [98],
    'idle_time': [0],
    'idle_men': [0],
    'no_of_style_change': [0],
    'no_of_workers': [59],
    'year': [2015],
    'month': [1],
    'day_num': [1],

    # One-hot encoded categorical values
    'quarter_2': [0],
    'quarter_3': [0],
    'quarter_4': [0],

    'department_sweing': [1],

    'day_Monday': [0],
    'day_Tuesday': [0],
    'day_Wednesday': [0],
    'day_Thursday': [1],
    'day_Friday': [0],
    'day_Saturday': [0],
    'day_Sunday': [0]
}

input_df = pd.DataFrame(input_data)

input_df = input_df.reindex(columns=x_train.columns, fill_value=0)

Predicted_data = data_rfg.predict(input_df)
print("Predicted Data: ", Predicted_data)

# Predicted Data:  [0.92694967]