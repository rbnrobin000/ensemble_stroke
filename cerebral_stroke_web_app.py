# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 09:49:29 2023

@author: rbnro
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

# loading the saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Predictive System', ['Cerebral Stroke'], icons=['person'], default_index=0)
    

# creating a function for prediction

def cspred(input_data):

    # Convert the dataframe row to a numpy array
    input_data_as_numpy_array = np.array(input_data.iloc[0])

    # Print the resulting numpy array
    #print(input_data_as_numpy_array)

    # change the input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the numpy array as we are predicting for one datapoint
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    c_s_prediction = loaded_model.predict(input_data_reshaped)
    print(c_s_prediction)

    if (c_s_prediction[0] == 0):
      return 'This person does not have any risk of cerebral stroke.'

    else:
      return 'This person has a risk of cerebral stroke.'
  
    

    

def main():
    
    # Cerebral Stroke Page
    if (selected == 'Cerebral Stroke'):
        
        # giving a title
        st.title('Cerebral Stroke Prediction')
        
        
        # getting the input data from the user
    
        gender = st.selectbox("Enter Gender (Male/Female/Other): ", ['Select One Option', 'Male', 'Female', 'Other'])
        age = st.text_input("Enter Age: ")
        hypertension = st.selectbox("Enter Hypertension (Yes/No): ", ['Select One Option', 'Yes', 'No'])
        heart_disease = st.selectbox("Enter Heart Disease (Yes/No): ", ['Select One Option', 'Yes', 'No'])
        ever_married = st.selectbox("Enter Ever Married (Yes/No): ", ['Select One Option', 'Yes', 'No'])
        work_type = st.selectbox("Enter Work Type (Govt_job/Private/Self-employed/Children/Never_worked): ", ['Select One Option', 'Govt_job', 'Private', 'Self-employed', 'Children', 'Never_worked'])
        Residence_type = st.selectbox("Enter Residence Type (Urban/Rural): ", ['Select One Option', 'Urban', 'Rural'])
        avg_glucose_level = st.text_input("Enter Average glucose Level: ")
        bmi = st.text_input("Enter BMI: ")
        smoking_status = st.selectbox("Enter Smoking Status (formerly smoked/never smoked/smokes/Unknown): ", ['Select One Option', 'formerly smoked', 'never smoked', 'smokes', 'Unknown'])
    
    # Create an empty pandas dataframe with the given columns
    input_data = pd.DataFrame(columns=['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'])
    
    # Add the user input to the pandas dataframe as a new row
    input_data.loc[0] = [gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type, avg_glucose_level, bmi, smoking_status]
    
    # Print the resulting dataframe
    #print(input_data)

    # convert categorical columns to numerical values
    input_data.replace({'gender':{'Male':0, 'Female':1, 'Other':2}, 'hypertension':{'No':0, 'Yes':1}, 'heart_disease':{'No':0, 'Yes':1}, 'ever_married':{'No':0, 'Yes':1}, 'work_type':{'Private':0, 'Self-employed':1, 'children':2, 'Govt_job':3, 'Never_worked':4}, 'Residence_type':{'Rural':0, 'Urban':1}, 'smoking_status':{'never smoked':0, 'Unknown':1, 'formerly smoked':2, 'smokes':3}}, inplace=True)

    # Print the resulting dataframe
    #print(input_data)

    # code for prediction
    diagnosis = ''
    
    # creating a button for prediction
    if st.button('Predict'):
        diagnosis = cspred(input_data)
  
    st.success(diagnosis)
    
    
    
if __name__ == '__main__':
    main()
  
    
  
    

    
