import os
from math import sqrt

from numpy.lib.arraysetops import _unpack_tuple
from TaxiFareModel.data import BUCKET_NAME, get_data

import joblib
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error

from google.cloud import storage
from TaxiFareModel import params

PATH_TO_LOCAL_MODEL = 'model.joblib'

AWS_BUCKET_TEST_PATH = "s3://wagon-public-datasets/taxi-fare-test.csv"

GCP_BUCKET_TEST_PATH = "data/test.csv"

STORAGE_LOCATION = 'models/TaxiFarePipeline/model.joblib'

BUCKET_NAME = "wagon-data-695-gulay"

BUCKET_MODEL_PATH = f"gs://{BUCKET_NAME}/{STORAGE_LOCATION}"

BUCKET_DATA_PATH = f"gs://{BUCKET_NAME}/{GCP_BUCKET_TEST_PATH}"


def get_test_data(nrows, data="GCP"):
    """method to get the test data (or a portion of it) from google cloud bucket
    To predict we can either obtain predictions from train data or from test data"""
    # Add Client() here
    path = "data/test.csv"  # ⚠️ to test from actual KAGGLE test set for submission

    if data == "local":
        df = pd.read_csv(path)
    elif data == "full":
        df = pd.read_csv(AWS_BUCKET_TEST_PATH)
    elif data == "GCP":
        df = pd.read_csv(BUCKET_DATA_PATH)
    else:
        df = pd.read_csv(AWS_BUCKET_TEST_PATH, nrows=nrows)
    return df


def get_model(path_to_joblib):
    pipeline = joblib.load(path_to_joblib)
    return pipeline

def get_model_from_gcp(bucket_name, storage_location):
    client = storage.Client()

    bucket = client.bucket(bucket_name)

    blob = bucket.blob(storage_location)

    blob.download_to_filename("gcmodel.joblib")

    pipeline = joblib.load("gcmodel.joblib")

    return pipeline

def evaluate_model(y, y_pred):
    MAE = round(mean_absolute_error(y, y_pred), 2)
    RMSE = round(sqrt(mean_squared_error(y, y_pred)), 2)
    res = {'MAE': MAE, 'RMSE': RMSE}
    return res


def generate_submission_csv(nrows, kaggle_upload=False):
    #Get Test Data
    df_test = get_test_data(nrows)
    # pipeline = joblib.load(PATH_TO_LOCAL_MODEL)

    #Get Pipeline
    pipeline = get_model_from_gcp(bucket_name= BUCKET_NAME,
                                  storage_location = STORAGE_LOCATION)

    if "best_estimator_" in dir(pipeline):
        y_pred = pipeline.best_estimator_.predict(df_test)
    else:
        y_pred = pipeline.predict(df_test)
    df_test["fare_amount"] = y_pred
    df_sample = df_test[["key", "fare_amount"]]
    name = f"predictions_test_ex.csv"
    df_sample.to_csv(name, index=False)
    print("prediction saved under kaggle format")
    # Set kaggle_upload to False unless you install kaggle cli
    if kaggle_upload:
        kaggle_message_submission = name[:-4]
        command = f'kaggle competitions submit -c new-york-city-taxi-fare-prediction -f {name} -m "{kaggle_message_submission}"'
        os.system(command)


if __name__ == '__main__':

    # ⚠️ in order to push a submission to kaggle you need to use the WHOLE dataset
    nrows = 100

    generate_submission_csv(nrows, kaggle_upload=False)
