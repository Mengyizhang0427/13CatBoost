# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 11:06:55 2023

@author: Starchild
"""
import streamlit as st
import pandas as pd
import pickle
import shap
import numpy as np

# Title
st.header("CatBoost ACLF death prediction model")
#input
st.sidebar.text("Please enter each factor below according to the model range")
INR=st.sidebar.number_input("INR(Norm:0.8-1.2,model range:0.9-12)",min_value=0.9, max_value=12)
age=st.sidebar.number_input("age(year,model range:21-90)",min_value=21, max_value=90)
bilirubin=st.sidebar.number_input("Total bilirubin(mg/dL,Norm:0.3-1.9,model range:0-50)",min_value=0, max_value=50)
resp_rate=st.sidebar.number_input("resp_rate(bpm,Norm:12-20,model range:0-70)",min_value=0, max_value=70)
albumin=st.sidebar.number_input("albumin(g/dL,Norm:3.5-5,model range:0-10)",min_value=0, max_value=10)
sodium=st.sidebar.number_input("sodium(mEq/L,Nrom:135-145,model range:0-200)",min_value=0, max_value=200)
heart_rate=st.sidebar.number_input("heart rate(bpm,Norm:60-100,model range:0-300)",min_value=0, max_value=300)
sbp=st.sidebar.number_input("sbp(mmHg,Norm:90-120,model range:0-400)",min_value=0, max_value=400)
spo2=st.sidebar.number_input("spo2(%,Norm:95-100,model range:0-100)",min_value=0, max_value=100)
alt=st.sidebar.number_input("alt(IU/L,Norm:7-56,model range:10-400)",min_value=10, max_value=400)
temperature=st.sidebar.number_input("temperature(°C,Norm:36.5-37.5,model range:35-50)",min_value=35, max_value=50)
platelet_count=st.sidebar.number_input("platelet count(K/μL,Norm:150-450,model range:0-500)",min_value=0, max_value=500)

with open('13CatBoost.pkl', 'rb') as f:
    clf = pickle.load(f)
with open('13data_max.pkl', 'rb') as f:
    data_max = pickle.load(f)
with open('13data_min.pkl', 'rb') as f:
    data_min = pickle.load(f)
with open('13CatBoost_explainer.pkl', 'rb') as f:
    explainer = pickle.load(f)


# If button is pressed
if st.button("Submit"):
    
    # Unpickle classifier
    # Store inputs into dataframe
    columns = ['INR','age','bilirubin','resp_rate', 'albumin','sodium','heart_rate','sbp','spo2','alt','temperature','platelet_count']
    X = pd.DataFrame([[INR,age,bilirubin,resp_rate, albumin,sodium,heart_rate,sbp,spo2,alt,temperature,platelet_count]], 
                     columns =columns )
    st.write('Raw data:')
    st.dataframe(X)
    X = (X-data_min)/(data_max-data_min)
    st.write('Normalized data:')
    st.dataframe(X)
    # Get prediction
    prediction = clf.predict(X)
    pred=clf.predict_proba(X)[0][1]
    shap_values2 = explainer(X)
    
    # Output prediction
    
    st.text(f"The probability of death of the patient is {pred}.")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig=shap.plots.bar(shap_values2[0])
    st.pyplot(fig)
    
    
    
    
    
    
    
