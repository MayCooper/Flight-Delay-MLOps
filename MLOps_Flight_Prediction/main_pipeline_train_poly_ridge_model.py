# ------------------------------------------------------------------------------
# Author: May Cooper
# Script: train_poly_ridge_model.py
#
# Description:
# This script builds and evaluates a polynomial Ridge regression model to
# predict flight departure delays. It includes data preprocessing, feature
# engineering, hyperparameter tuning, and evaluation using real-world
# airline schedule data.
#
# Key Features:
# - Parses command-line arguments to control the number of alpha values tested
# - Splits training and test data by date (first 3 weeks vs. last week)
# - Applies one-hot encoding to categorical features (destination airports)
# - Generates polynomial features and performs Ridge regression with MSE tracking
# - Uses MLflow to log parameters, metrics, models, and visual artifacts
# - Exports final trained model and supporting files for deployment
#
# The script is designed to support reproducible, trackable machine learning
# experiments and produce artifacts that can be integrated into a live API.
# ------------------------------------------------------------------------------

import datetime
import pandas as pd
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder, OneHotEncoder
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import mlflow
import mlflow.sklearn
import logging
import os
import pickle
import json

# set up the argument parser
parser = argparse.ArgumentParser(description='Parse parameters for the polynomial regression')
parser.add_argument('--num_alphas', type=int, default=20, help='Number of Ridge penalty increments')
parser.add_argument('--cleaned_data_file', type=str, required=True, help='Path to the cleaned data file')
args = parser.parse_args()

num_alphas = args.num_alphas
cleaned_data_file = args.cleaned_data_file
order = 1

# configure logger
logname = "polynomial_regression.txt"
logging.basicConfig(filename=logname,
                    filemode='w',
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
logging.getLogger('matplotlib.font_manager').disabled = True
logging.info("Flight Departure Delays Polynomial Regression Model Log")

# read the data file
df = pd.read_csv(cleaned_data_file)
tab_info = pd.DataFrame(df.dtypes).T.rename(index={0:'column type'})

def grab_month_year(df: pd.DataFrame) -> tuple:
    months = pd.unique(df['MONTH'])
    years = pd.unique(df['YEAR'])
    if len(months) > 1:
        raise Exception("Multiple months found in data set, only one acceptable")
    else:
        month = int(months[0])
    if len(years) > 1:
        raise Exception("Multiple years found in data set, only one acceptable")
    else:
        year = int(years[0])
    return (month, year)

def format_hour(string: str) -> datetime.time:
    if pd.isnull(string):
        return np.nan
    else:
        if string == 2400:
            string = 0
        string = "{0:04d}".format(int(string))
        hour = datetime.time(int(string[0:2]), int(string[2:4]))
        return hour

def combine_date_hour(x: list) -> datetime.datetime:
    if pd.isnull(x.iloc[0]) or pd.isnull(x.iloc[1]):
        return np.nan
    else:
        return datetime.datetime.combine(x.iloc[0], x.iloc[1])

def create_flight_time(df: pd.DataFrame, col: str) -> pd.Series:
    lst = []
    for index, cols in df[['DATE', col]].iterrows():
        if pd.isnull(cols.iloc[1]):
            lst.append(np.nan)
        elif float(cols.iloc[1]) == 2400:
            cols.iloc[0] += datetime.timedelta(days=1)
            cols.iloc[1] = datetime.time(0,0)
            lst.append(combine_date_hour(cols))
        else:
            cols.iloc[1] = format_hour(cols.iloc[1])
            lst.append(combine_date_hour(cols))
    return pd.Series(lst)

def create_df(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df[['SCHEDULED_DEPARTURE','SCHEDULED_ARRIVAL','DEST_AIRPORT','DEPARTURE_DELAY']]
    df2 = df2.dropna(how='any')
    df2.loc[:, 'weekday'] = df2['SCHEDULED_DEPARTURE'].apply(lambda x: x.weekday())
    # delete delays > 1h
    df2.loc[:, 'DEPARTURE_DELAY'] = df2['DEPARTURE_DELAY'].apply(lambda x: x if x < 60 else np.nan)
    df2 = df2.dropna(how='any')
    # formatting times
    fct = lambda x: x.hour*3600 + x.minute*60 + x.second
    df2.loc[:, 'hour_depart'] = df2['SCHEDULED_DEPARTURE'].apply(lambda x: x.time())
    df2.loc[:, 'hour_depart'] = df2['hour_depart'].apply(fct)
    df2.loc[:, 'hour_arrive'] = df2['SCHEDULED_ARRIVAL'].apply(fct)
    df2 = df2[['hour_depart', 'hour_arrive', 'DEST_AIRPORT', 'DEPARTURE_DELAY', 'weekday']]
    df3 = df2.groupby(['hour_depart', 'hour_arrive', 'DEST_AIRPORT'], as_index=False).mean()
    return df3

df['DATE'] = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])
(month, year) = grab_month_year(df)
logging.info("Month and year of data: %s %s", month, year)
df['SCHEDULED_DEPARTURE'] = create_flight_time(df, 'SCHEDULED_DEPARTURE')
df['DEPARTURE_TIME'] = df['DEPARTURE_TIME'].apply(format_hour)
df['SCHEDULED_ARRIVAL'] = df['SCHEDULED_ARRIVAL'].apply(format_hour)
df['ARRIVAL_TIME'] = df['ARRIVAL_TIME'].apply(format_hour)

# define training data as first 3 weeks of the month and test data as that from the fourth week
df_train = df[df['SCHEDULED_DEPARTURE'].apply(lambda x: x.date()) < datetime.date(year, month, 23)]
df_test  = df[df['SCHEDULED_DEPARTURE'].apply(lambda x: x.date()) > datetime.date(year, month, 23)]

df3 = create_df(df_train)

# perform one-hot encoding of all destination airports in training data
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(df3['DEST_AIRPORT'])
zipped = zip(integer_encoded, df3['DEST_AIRPORT'])
label_airports = list(set(list(zipped)))
label_airports.sort(key=lambda x: x[0])

onehot_encoder = OneHotEncoder(sparse_output=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
b = np.array(df3[['hour_depart', 'hour_arrive']])
X = np.hstack((onehot_encoded, b))
Y = np.array(df3['DEPARTURE_DELAY']).reshape(-1, 1)
logging.info("Airport one-hot encoding successful")

# train/validation split at 30%
X_train, X_validate, Y_train, Y_validate = train_test_split(X, Y, test_size=0.3)

# Since 'mlflow run' already started a run, we make this a nested run:
with mlflow.start_run(run_name="Alpha Tuning", nested=True):
    score_min = 10000
    alpha_max = num_alphas * 2
    count = 1
    poly = PolynomialFeatures(degree=order)
    for alpha in range(0, alpha_max, 2):
        run_num = "Training Run Number " + str(count)
        ridgereg = Ridge(alpha=alpha/10)
        X_ = poly.fit_transform(X_train)
        ridgereg.fit(X_, Y_train)
        X_val_ = poly.fit_transform(X_validate)
        result = ridgereg.predict(X_val_)
        score = metrics.mean_squared_error(result, Y_validate)

        # This run is also nested:
        with mlflow.start_run(run_name=run_num, nested=True):
            mlflow.log_param("alpha", alpha/10)
            mlflow.log_metric("Training Data Mean Squared Error", score)
            mlflow.log_metric("Training Data Average Delay", np.sqrt(score))

        if score < score_min:
            score_min = score
            parameters = [alpha, order]
        logging.info("n={} alpha={} , MSE = {:<0.5}".format(order, alpha/10, score))
        count += 1

    X_val_ = poly.fit_transform(X_validate)
    tresult = ridgereg.predict(X_val_)
    tscore = metrics.mean_squared_error(tresult, Y_validate)
    logging.info('Training Data Final MSE = {}'.format(round(tscore, 2)))
    mlflow.log_metric("Training Data Mean Squared Error", tscore)
    mlflow.log_metric("Training Data Average Delay", np.sqrt(tscore))

logging.info("Model training loop completed with %s iterations", count-1)

# create a data frame of the test data
df3 = create_df(df_test)

label_conversion = {s[1]: int(s[0]) for s in label_airports}

# export airport label conversion for test data to json file for later use
jsonout = json.dumps(label_conversion)
with open("airport_encodings.json", "w") as f:
    f.write(jsonout)

logging.info("Export of airport one-hot encoding successful")
df3['DEST_AIRPORT'] = df3['DEST_AIRPORT'].map(pd.Series(label_conversion))

# manually one-hot encode destination airports for test data
for index, label in label_airports:
    temp = df3['DEST_AIRPORT'] == index
    temp = temp.apply(lambda x:1.0 if x else 0.0)
    if index == 0:
        matrix = np.array(temp)
    else:
        matrix = np.vstack((matrix, temp))
matrix = matrix.T

b = np.array(df3[['hour_depart', 'hour_arrive']])
X_test = np.hstack((matrix, b))
Y_test = np.array(df3['DEPARTURE_DELAY']).reshape(-1, 1)
logging.info("Wrangling of test data successful")

X_test_ = poly.fit_transform(X_test)
result = ridgereg.predict(X_test_)
score = metrics.mean_squared_error(result, Y_test)
logging.info('Test Data MSE = {}'.format(round(score, 2)))
logging.info("Predictions using test data successful")
logging.info('Test Data average delay = {:.2f} min'.format(np.sqrt(score)))

filename = 'finalized_model.pkl'
pickle.dump(ridgereg, open(filename, 'wb'))
logging.info("Final model export successful")

tips = pd.DataFrame()
if result.ndim == 1:
    tips["prediction"] = pd.Series(result)
else:
    tips["prediction"] = pd.Series(result[:, 0])
tips["original_data"] = pd.Series(Y[:,0].astype(float))

sns.jointplot(x="original_data", y="prediction", data=tips, height=6, ratio=7,
              joint_kws={'line_kws':{'color':'limegreen'}}, kind='reg')
plt.xlabel('Mean delays (min)', fontsize=15)
plt.ylabel('Predictions (min)', fontsize=15)
plt.plot(list(range(-10,25)), list(range(-10,25)), linestyle=':', color='r')
plt.savefig("model_performance_test.jpg", dpi=300)
logging.info("Model performance plot export successful")

# Final model run (also nested)
with mlflow.start_run(run_name="Final Model - Test Data", nested=True):
    # Log model params
    mlflow.log_param("Ridge Regression Alpha", parameters[0])
    mlflow.log_param("Polynomial Order", parameters[1])

    # Log performance metrics on test dataset
    mlflow.log_metric("MSE_Test_Data", score)
    mlflow.log_metric("Avg_Delay_Minutes", np.sqrt(score))

    # Log artifacts
    mlflow.log_artifact(logname)
    mlflow.log_artifact("model_performance_test.jpg")
    mlflow.log_artifact("finalized_model.pkl")
    mlflow.log_artifact("airport_encodings.json")

    print("Complete MLFlow logging: parameters, metrics, and artifacts recorded.")

logging.shutdown()
