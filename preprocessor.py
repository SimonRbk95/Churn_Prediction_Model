import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


# create dataframe with project's data set
df = pd.read_csv('HR_testSet.csv')


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

p_cols_cat=["BusinessTravel_Travel_Frequently", 
    "BusinessTravel_Travel_Rarely", 
    "JobRole_Laboratory_Technician", 
    "MaritalStatus_Single",
    "OverTime"
]

def preprocess(df, string):

    to_predict=[]

    df['OverTime'] = df['OverTime'].map({'Yes':1, 'No':0})

    if string == "Batch":
        # deleting clearly unuseful variables (employee count, employee number, over 18, Standard Hours)
        # delete Attrition role as test csv will include attrition
        df = df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis=1)

        # drop rows that contain at least one empty cell
        df.dropna(axis='rows', thresh=1)

        # currently Gender column will only exist in batch test file 
        df['Gender'] = df['Gender'].map({'Male':1, 'Female':0})

        # one-hot encoding of features with more than two categories
        df = pd.get_dummies(df, drop_first=True)

        # only consider statistically relevant columns   
        df.drop([col for col in df.columns if col not in p_cols], axis=1, inplace=True)

        # one-hot encoding of features with more than two categories
        df = pd.get_dummies(df)

        # change variable name seperators
        df.columns = (column.replace(" ", "_").replace("(", 
                "_").replace("&", "_").replace("-", "_")
                for column in df.columns)

        sc = MinMaxScaler()
        to_scale = [col for col in df.columns if df[col].nunique() > 2]
        for col in to_scale:
            df[col] = sc.fit_transform(df[[col]]) 
    
    else:
        df.rename(columns={'BusinessTravel_0': f'BusinessTravel_{df["BusinessTravel"]}',
                            'JobRole_0': f'JobRole_{df["JobRole"]}',
                            'MartialStatus_0' : f'MartialStatus_{df["MartialStatus"]}'
                            }, inplace=True)
    
        # one-hot encoding of features with more than two categories
        df = pd.get_dummies(df)
        (print(df.columns))
        (print(df.values))
        # change variable name seperators
        df.columns = (column.replace(" ", "_").replace("(", 
                "_").replace("&", "_").replace("-", "_")
                for column in df.columns)

        sc = MinMaxScaler()
        to_scale = [col for col in df.columns if col not in p_cols_cat]
        for col in to_scale:
            df[col] = sc.fit_transform(df[[col]]) 
        print(df.values)


        
    # iterate through rows
    for i, row in df.iterrows():
        # list to store row values s
        row_list=[]
        # only consider columns that are in p_cols
        for col in p_cols:
            # if the col exists add its vale to list 
            if col in df.columns:
                row_list.append(row.loc[col])
            # if it does not exist add 0 as it must be a categorical data point that does not exist in row
            else:
                row_list.append(0)
        # add list of row to final list
        to_predict.append(row_list)
    print(to_predict)
    return to_predict
