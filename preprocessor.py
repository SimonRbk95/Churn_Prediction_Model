import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

 p_cols=["DistanceFromHome", 
    "EnvironmentSatisfaction", 
    "JobInvolvement", 
    "JobSatisfaction", 
    "NumCompaniesWorked", 
    "OverTime", 
    "TotalWorkingYears", 
    "YearsInCurrentRole", 
    "YearsSinceLastPromotion", 
    "BusinessTravel_Travel_Frequently", 
    "BusinessTravel_Travel_Rarely", 
    "JobRole_Laboratory_Technician", 
    "MaritalStatus_Single"]

def prepocess(df, string):

    if string == "Batch":
        # deleting clearly unuseful variables (employee count, employee number, over 18, Standard Hours)
        # delete Attrition role as test csv will include attrition
        df = df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours', 'Attrition'], axis=1)

        # drop rows that contain at least one empty cell
        df.dropna(axis='rows', thresh=1)

        # encoding Gender categorical data
        df['Gender'] = df['Gender'].map({'Male':1, 'Female':0})
        
    # one-hot encoding of features with more than two categories
    df = pd.get_dummies(df, drop_first=True)

    # change variable name seperators
    df.columns = (column.replace(" ", "_").replace("(", 
                    "_").replace("&", "_").replace("-", "_")
                    for column in df.columns)

    # only consider statistically relevant columns
    df.drop([col for col in df.columns if col not in p_cols], axis=1, inplace=True)
    
    # feature scaling
    sc = MinMaxScaler()
    to_scale = [col for col in X_train.columns if X_train[col].nunique() > 2]
    for col in to_scale:
        X_train[col] = sc.fit_transform(X_train[[col]])
        X_test[col] = sc.fit_transform(X_test[[col]])  
    
    return df


