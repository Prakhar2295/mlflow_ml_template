import os
import mlflow
import mlflow.sklearn
import argparse
import time
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import numpy as np
import pandas as pd




def evaluate(actual,pred):
    rmse = np.sqrt(mean_squared_error(actual,pred))
    mae = mean_absolute_error(actual,pred)
    r2 = r2_score(actual,pred)
    return rmse,mae,r2


def get_data():
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    try:
        df = pd.read_csv( url,sep = ";")
        return df
    except Exception as e:
        raise e
        

def main(alpha,l1_ratio):
    df = get_data()
    train,test = train_test_split(df)
    
    TARGET = "quality"
    
    train_x = train.drop([TARGET],axis = 1)
    test_x = test.drop([TARGET],axis = 1)
    
    train_y = train[[TARGET]]
    test_y = test[[TARGET]]
    ## mlflow implemenation
    #mlflow.set_tracking_uri(uri='file:///D:/FSDS/MAchine_Learning/MLFLOW_ML_EXAMPLE/mlruns_1')
    with mlflow.start_run():
        
        mlflow.log_param("alpha",alpha)
        mlflow.log_param("l1_ratio",l1_ratio)
        
        model = ElasticNet(alpha = alpha,l1_ratio= l1_ratio,random_state = 100)
        model.fit(train_x, train_y)
        
        pred = model.predict(test_x)
        
        rmse,mae,r2 = evaluate(test_y,pred)
        
        mlflow.log_metric("rmse",rmse)
        mlflow.log_metric("mae",mae)
        mlflow.log_metric("r2",r2)
        
        
        mlflow.sklearn.log_model(model,"model")
           

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--alpha","-a",type= float,default = 0.6)
    args.add_argument("--l1_ratio","-l1",type= float,default =0.2)
    parsed_args = args.parse_args()
    
    main(parsed_args.alpha,parsed_args.l1_ratio)
    
    
    
    
        

        
        
    

