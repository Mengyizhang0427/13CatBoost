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
from sklearn.metrics import precision_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

# Title
st.header("CatBoost ACLF death prediction model")
#example
df=pd.read_csv('example.csv')
example= df.to_csv(index=False)
st.download_button(label='Download example', data=example, file_name='example.csv', mime='csv')
#input
uploaded_file = st.file_uploader("Please upload the csv file(The file must contain the following columns：INR,age,bilirubin,resp_rate, albumin,sodium,heart_rate,sbp,spo2,alt,temperature,platelet_count,30day)", )

with open('13CatBoost.pkl', 'rb') as f:
    clf = pickle.load(f)
with open('13data_max.pkl', 'rb') as f:
    data_max = pickle.load(f)
with open('13data_min.pkl', 'rb') as f:
    data_min = pickle.load(f)
#with open('13CatBoost_explainer.pkl', 'rb') as f:
#    explainer = pickle.load(f)


# If button is pressed
if st.button("Submit"):
    data = pd.read_excel(uploaded_file)
    # Unpickle classifier
    # Store inputs into dataframe
    X = pd.DataFrame()
    X['INR']=data["INR"]
    X['age']=data['age']
    X['bilirubin']=data['bilirubin']
    X['resp_rate']=data['resp_rate']
    X['albumin']=data['albumin']
    X['sodium']=data['sodium']
    X['heart_rate']=data['heart_rate']
    X['sbp']=data['sbp']
    X['spo2']=data['spo2']
    X['alt']=data['alt']
    X['temperature']=data['temperature']
    X['platelet_count']=data['platelet_count']
    
    st.write('Raw data:')
    st.dataframe(X)
    X = (X-data_min)/(data_max-data_min)
    st.write('Normalized data:')
    st.dataframe(X)
    # Get prediction
    prediction = clf.predict(X)
    probas = clf.predict_proba(X)
    pred = probas[:, 1]
    best_threshold=0.4457025818600448
    y_pred = (pred >= best_threshold).astype(int)
    if '30day' in data.columns:
        y=data["30day"]
        # 计算混淆矩阵
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        # 计算特异性
        specificity = tn / (tn + fp)
        # 计算敏感性（召回率）
        sensitivity = tp / (tp + fn)
        #精确率，F1
        precision=precision_score(y, y_pred)
        F1=f1_score(y, y_pred)
        # 计算AUC
        auc = roc_auc_score(y, pred)    
        st.text(f"AUC value of prediction result: {auc}.")
    #shap_values2 = explainer(X)
    # Output prediction
    result=pd.DataFrame()
    result["Predicted_result"]=y_pred
    result["Predicted_probability"]=pred
    csv = result.to_csv(index=False)
    st.download_button(label='Download result', data=csv, file_name='result.csv', mime='text/csv')
    #st.set_option('deprecation.showPyplotGlobalUse', False)
    #fig=shap.plots.bar(shap_values2[0])
    #st.pyplot(fig)
    
    
    
    
    
    
    
