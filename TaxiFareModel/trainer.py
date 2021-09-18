import joblib
from termcolor import colored
import mlflow
from TaxiFareModel.data import get_data, clean_data, df_optimized
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.utils import compute_rmse
from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
from google.cloud import storage
from TaxiFareModel import params
import pandas as pd

#Estimators
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor

#Grid Search
from sklearn.model_selection import GridSearchCV


MLFLOW_URI = "https://mlflow.lewagon.co/"
EXPERIMENT_NAME = "[AUS] [MEL] [roadbusta] TaxiFarePipeline v1.2"
STORAGE_LOCATION = params.STORAGE_LOCATION


class Trainer(object):
    def __init__(self, X, y, estimator, experiment_name, model_name):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y
        # for MLFlow
        self.experiment_name = experiment_name
        self.mlflow_uri = MLFLOW_URI
        self.estimator = estimator
        self.model_name = model_name

    def set_experiment_name(self, experiment_name):
        '''defines the experiment name for MLFlow'''
        self.experiment_name = experiment_name

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, [
                "pickup_latitude",
                "pickup_longitude",
                'dropoff_latitude',
                'dropoff_longitude'
            ]),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")

        self.pipeline = Pipeline([
            ('preproc', preproc_pipe),
            ('estimator', self.estimator)
        ])

    def run(self, model_name):
        self.set_pipeline()
        self.mlflow_log_param("model", model_name)
        self.pipeline.fit(self.X, self.y)

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        self.mlflow_log_metric("rmse", rmse)
        return round(rmse, 2)

    def save_model(self):
        """Save the model into a .joblib format"""
        joblib.dump(self.pipeline, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))

    def upload_model_to_gcp(self,model_name):

        client = storage.Client()

        bucket = client.bucket(params.BUCKET_NAME)

        blob = bucket.blob(f"{STORAGE_LOCATION}{model_name}.joblib")

        blob.upload_from_filename('model.joblib')



    def save_model_to_gcp(self, model_name):
        """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
        HINTS : use joblib library and google-cloud-storage"""

        # saving the trained model to disk is mandatory to then beeing able to upload it to storage
        # Implement here
        joblib.dump(self.pipeline, 'model.joblib')
        print("saved model.joblib locally")

        # Implement here
        self.upload_model_to_gcp(model_name)
        print(
            f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}{model_name}.joblib"
        )




    # MLFlow methods
    @memoized_property
    def mlflow_client(self):
        # mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_tracking_uri(self.mlflow_uri)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(
                self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":
    #Experiment name
    experiment_name = "[AUS] [MEL] [roadbusta] TaxiFare v1.6"

    # Get and clean data
    N = params.SAMPLES
    df = get_data(nrows=N)
    df = clean_data(df)
    df = df_optimized(df)

    # Drop number of passengers
    df = df.drop(columns='passenger_count')

    # set X and y
    y = df["fare_amount"]

    #Drop the fare amount
    X = df.drop("fare_amount", axis=1)

    # hold out
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)

    #Create an estimator dictionary
    n_jobs = 1
    estimators = {
        "Linear Regression" : LinearRegression(n_jobs = n_jobs),
        "KNN" : KNeighborsRegressor(),
        "SVR" : SVR(),
        "Adaboost" : AdaBoostRegressor()
    }

    #Create a estimator_params dictionary
    #May need to reconsider naming the parameters
    hyperparameters = {

        "KNN" : {"n_neighbors" : [2, 5, 10],
                     "weights" : ["uniform", "distance"],
                   "leaf_size" : [15, 30, 45]
                   },

        "SVR" : {"kernel" : ["linear", "poly", "rbf"],
                      "C" : [0.01, 0.1, 0.5, 1]
                 },
        "Adaboost" : {"learning_rate" : [1, 5, 10],
                      "loss" : ["linear", "square", "exponential"]}
    }

    for model_name, estimator in estimators.items():
        #instanciate  pipeline class
        trainer_inst = Trainer(X_train, y_train,
                               estimator = estimator,
                               experiment_name = experiment_name,
                               model_name = model_name)

        #build a pipeline
        pipe = trainer_inst.set_pipeline()

        # train
        pipeline = trainer_inst.run(model_name)

        # evaluate
        rmse = trainer_inst.evaluate(X_val, y_val)
        print(f'rmse for {model_name}: ', rmse)

        trainer_inst.save_model_to_gcp(model_name)
    # print trainer model
    experiment_id = trainer_inst.mlflow_experiment_id #Try to figure out why this isn't working
    print(
        f"experiment URL: https://mlflow.lewagon.co/#/experiments/{experiment_id}"
    )

    #Original Model
    # N = params.SAMPLES
    # df = get_data(nrows=N)
    # df = clean_data(df)
    # df = df_optimized(df)  #Optimize the data data after loading the data

    # y = df["fare_amount"]
    # X = df.drop("fare_amount", axis=1)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # # Train and save model, locally and
    # trainer = Trainer(X=X_train, y=y_train)
    # # trainer.set_experiment_name(EXPERIMENT_NAME) #Need to ensure that this is a new name
    # trainer.run()
    # rmse = trainer.evaluate(X_test, y_test)
    # print(f"rmse: {rmse}")
    # experiment_id = trainer.mlflow_experiment_id
    # print(
    #     f"experiment URL: https://mlflow.lewagon.co/#/experiments/{experiment_id}"
    # )
