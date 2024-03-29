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
example = df.to_csv(index=False)
st.download_button(label='Download example', data=example, file_name='example.csv', mime='csv')
#input
st.write("The factors are explained as follows：")
st.write("INR(Norm:0.8-1.2,model range:0.8-12)")
st.write("age(year,model range:21-90)")
st.write("Total bilirubin(mg/dL,Norm:0.3-1.9,model range:0-50)")
st.write("resp rate(bpm,Norm:12-20,model range:0-70)")
st.write("albumin(g/dL,Norm:3.5-5,model range:0-10)")
st.write("sodium(mEq/L,Nrom:135-145,model range:0-200)")
st.write("heart rate(bpm,Norm:60-100,model range:0-300)")
st.write("sbp(mmHg,Norm:90-120,model range:0-400)")
st.write("spo2(%,Norm:95-100,model range:0-100)")
st.write("alt(IU/L,Norm:7-56,model range:10-4000)")
st.write("temperature(°C,Norm:36.5-37.5,model range:35-50)")
st.write("platelet count(K/μL,Norm:150-450,model range:0-500)")
st.write("30day represents the death within 30 days, 1 represents death, 0 represents survival")
#input
uploaded_file = st.file_uploader("Please upload the csv file(If the range of the data does not meet the model range, the prediction result is None)",type=['csv'])

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
    data = pd.read_csv(uploaded_file)
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
    result=pd.DataFrame()
    within_range=[]
    for i in range(len(X)):
        if X.iloc[i]['INR']<0.8 or X.iloc[i]['INR']>12 or\
        X.iloc[i]['age']<21 or X.iloc[i]['age']>90 or\
        X.iloc[i]['bilirubin']<0 or X.iloc[i]['bilirubin']>50 or\
        X.iloc[i]['resp_rate']<0 or X.iloc[i]['resp_rate']>70 or\
        X.iloc[i]['albumin']<0 or X.iloc[i]['albumin']>10 or\
        X.iloc[i]['sodium']<0 or X.iloc[i]['sodium']>200 or\
        X.iloc[i]['heart_rate']<0 or X.iloc[i]['heart_rate']>300 or\
        X.iloc[i]['sbp']<0 or X.iloc[i]['sbp']>400 or\
        X.iloc[i]['spo2']<0 or X.iloc[i]['spo2']>100 or\
        X.iloc[i]['alt']<10 or X.iloc[i]['alt']>4000 or\
        X.iloc[i]['temperature']<35 or X.iloc[i]['temperature']>50 or\
        X.iloc[i]['platelet_count']<0 or X.iloc[i]['platelet_count']>500 :
            h='NO'
            within_range.append(h)
        else:
            h='YES'
            within_range.append(h)
        
    st.write('Raw data:')
    st.dataframe(X)
    X1 = pd.DataFrame()
    X1 = (X-data_min)/(data_max-data_min)
    X1['is_within_range']=within_range
    st.write('Normalized data:')
    st.dataframe(X1)
    # Get prediction
    def make_prediction(X):
        pred=[]
        for i in range(len(X)):
            if X.iloc[i]['is_within_range']=='YES':
                prediction = clf.predict_proba(X.iloc[i, :-1])
                prob=prediction[1]
                pred.append(prob)     
            else:
                pred.append(None)
        return pred
    pred = make_prediction(X1)
    best_threshold=0.4457025818600448
    y_pred=[]
    for pre in pred:
        if pre is not None:
            if pre >=best_threshold:
                y_pred.append(1)
            else:
                y_pred.append(0)
        else:
            y_pred.append(None)
    if '30day' in data.columns:
        y=data["30day"]
        data_y=pd.DataFrame()
        data_y['y']=y
        data_y['y_pred']=y_pred
        df_y = data_y.dropna()
        # 计算混淆矩阵
        tn, fp, fn, tp = confusion_matrix(df_y['y'], df_y['y_pred']).ravel()
        # 计算特异性
        specificity = tn / (tn + fp)
        # 计算敏感性（召回率）
        sensitivity = tp / (tp + fn)
        #精确率，F1
        precision=precision_score(df_y['y'], df_y['y_pred'])
        F1=f1_score(df_y['y'], df_y['y_pred'])
        # 计算AUC
        auc = roc_auc_score(df_y['y'], df_y['y_pred'])    
        st.text(f"AUC value of prediction result: {auc}.")
    #shap_values2 = explainer(X)
    # Output prediction
    result['is_within_range']=X1['is_within_range']
    result["Predicted_result"]=y_pred
    result["Predicted_probability"]=pred
    csv = result.to_csv(index=False)
    st.download_button(label='Download result', data=csv, file_name='result.csv', mime='text/csv')
    #st.set_option('deprecation.showPyplotGlobalUse', False)
    #fig=shap.plots.bar(shap_values2[0])
    #st.pyplot(fig)
    
    
    
    
    
    
    
