'''----- modified code inspired by: https://neptune.ai/blog/how-to-implement-customer-churn-prediction ------'''

#Import libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

#load the model from disk

# load saved model

with open('model.pkl' , 'rb') as f:
    model = pickle.load(f)

#Import python scripts
from preprocessor import preprocess

def main():
    #Setting Application title
    st.title('Employee Attrition')

      #Setting Application description
    st.markdown("""
     :dart:  This Streamlit app is made to predict employee's attrition in an HR use case based on real company data.
    The application is functional for both online prediction and batch data prediction.
    """)
    st.markdown("<h3></h3>", unsafe_allow_html=True)

    #Setting Application sidebar default
    image = Image.open('App.jpg')
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?", ("Online", "Batch"))
    st.sidebar.info('This app is created to predict Employee Attrition')
    st.sidebar.image(image)

    if add_selectbox == "Online":
        st.info("Input data below")
        #Based on our optimal features selection
        #st.subheader("Personal Data")
        MartialStatus= st.selectbox('Martial Status:', ('Single', 'Married', 'Divorced'))
        Jobinvolvement = st.number_input('How involved are you in your current job?', min_value=1, max_value=4, value=1)
        OverTime = st.selectbox('Have you worked overtime?', ('Yes', 'No'))
        NumCompaniesWorked = st.number_input('How many companies have you worked for?', min_value=1, max_value=100, value=1)
        TotalWorkingYears = st.number_input('How many years have you worked XYZ Ltd.?', min_value=0, max_value=100, value=1)
        YearsInCurrentRole = st.number_input('How many years have you worked in your current role at XYZ Ltd.?', min_value=0, max_value=100, value=1)
        YearsSinceLastPromotion = st.number_input('How many years have passed since your last promotion', min_value=0, max_value=150, value=0)
        BusinessTravel = st.selectbox('How often do you travel in your current role?', ('None', 'Rarely', 'Frequently'))
        DistanceFromHome = st.number_input('How far you have to travel to get to work? (in km)', min_value=0, max_value=1000, value=0)
        JobRole = st.selectbox("What is your current job role?", ('Sales Executive', 'Research Scientist',
                                                                 'Laboratory Technician', 'Manufacturing Director', 
                                                                 'Healthcare Representative', 'Sales Representative', 
                                                                 'Manager', 'Sales Executive'))
        Environmentsatisfaction = st.number_input('How satisfied are you with the work environment?', min_value=1, max_value=4, value=1)
        JobSatisfaction = st.number_input('How satisfied are you with your job?', min_value=1, max_value=4, value=1)



        data={
                'EnvironmentSatisfaction' : Environmentsatisfaction,
                'JobInvolvement' : Jobinvolvement,
                'OverTime' : OverTime,
                'NumCompaniesWorked' : NumCompaniesWorked,
                'TotalWorkingYears' : TotalWorkingYears,
                'YearsInCurrentRole' : YearsInCurrentRole,
                'YearsSinceLastPromotion' : YearsSinceLastPromotion,
                'MartialStatus' : MartialStatus,
                'BusinessTravel' : BusinessTravel,
                'JobRole' : JobRole,
                'JobSatisfaction': JobSatisfaction,
                'DistanceFromHome' : DistanceFromHome,
        }


        features_df = pd.DataFrame.from_dict([data])
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.write('Overview of input is shown below')
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.dataframe(features_df)
        #Preprocess inputs
        

        if st.button('Predict'):
            to_predict = preprocess(features_df, 'Online')
            prediction = model.predict(to_predict)
            if prediction == 1:
                st.warning('Yes, the employee will leave the company.')
            else:
                st.success('No, the employee will not leave the company.')


    else:
        st.subheader("Dataset upload")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            #Get overview of data
            st.write(data.head())
            st.markdown("<h3></h3>", unsafe_allow_html=True)
            #Preprocess inputs
            topredict_list= preprocess(data, "Batch")
            if st.button('Predict'):
                #Get batch prediction
                prediction = model.predict(topredict_list)
                prediction_df = pd.DataFrame(prediction, columns=["Predictions"])
                prediction_df = prediction_df.replace({1:'Yes, the employee will leave the company.',
                                                    0:'No, the employee will not leave the company.'})

                st.markdown("<h3></h3>", unsafe_allow_html=True)
                st.subheader('Prediction')
                st.write(prediction_df)

if __name__ == '__main__':
        main()