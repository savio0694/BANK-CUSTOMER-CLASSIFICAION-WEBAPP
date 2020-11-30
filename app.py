import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mt
import streamlit as st
import streamlit.components.v1 as components
import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as IMPPIPE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

bankdata=pd.read_csv('bank.csv',sep=',')
VAL=0
VAL1=0

#--------------CREATE MODEL---SCALE-BALANCE CLASSES------------#
def make_model(bankdata):
    final_data_features=bankdata.iloc[:,[0,1,3,4]]
    final_data_target=bankdata.iloc[:,[2]]
    X_train, X_test, y_train, y_test = train_test_split(final_data_features,final_data_target,random_state=1,shuffle=True,test_size=0.1)

    over = SMOTE()
    under = RandomUnderSampler()

    steps = [('o', over), ('u', under)]
    SAMPLER = imblearn.pipeline.Pipeline(steps=steps)

    X_SMOTE, y_SMOTE = SAMPLER.fit_resample(X_train, y_train)

    pre_process = ColumnTransformer([('scale',StandardScaler(),['housing_loan','personal_loan','age','balance']) ])
                      
    RF=Pipeline([('PR',pre_process),('DECISON_TREE',RandomForestClassifier(random_state=0))])

    RF_MODEL=RF.fit(X_SMOTE,y_SMOTE)
    
     
    return RF_MODEL

#---------------------LAYOUT----------------------------#

st.title('CUSTOMER CLASSIFICATION')

st.text('''The data is related with direct marketing campaigns of a 
Portuguese banking institution. The marketing campaigns were based 
on phone calls. Often, more than one contact to the same client 
was required, in order to access if the product (bank term deposit) 
would be ('yes') or not ('no') subscribed.''')

st.subheader('To view the entire dataset with the original features,EDA,feature selecion,Adjustments forClass imbalance etc. view the jupyter noetebook on Github.')



st.dataframe(bankdata)

housing = st.sidebar.radio(
     "Does he customer have a housing loan?",
    ('YES', 'NO'))
    
if housing == 'YES':
     VAL=1
     
else:
     VAL=0
     
    
personal=st.sidebar.radio(
     "Does he customer have a personal loan?",
    ('YES', 'NO'))
    
if personal == 'YES':
     VAL1=1
     
else:
     VAL1=0
     
age = st.sidebar.slider('How old is the customer', 18, 100, 25)

balance= st.sidebar.slider('What is the customer balance?', -500, 2000, 250)

#---------------LAYOUT-------------------------------#


#--------------MAKE-TEST_DATAFRAME-------------#     
def make_frame(VAL,VAL1,age,balance):
      data={"housing_loan" : VAL, 
      "personal_loan" : VAL1, 
      "age" : age,
      "balance":balance}
      df = pd.DataFrame(data,index=[1])
      return df
      
 

df = make_frame(VAL,VAL1,age,balance)



MODEL=make_model(bankdata)
PREDICTION=MODEL.predict(df)


#---FINAL DECISION-------#
def final_decision(PREDICTION):
    if PREDICTION[0]==0:
       return " The customer is UNLIKELY to have a TERM DEPOSIT"
    else:
        return " The customer is LIKELY to have a TERM DEPOSIT"
 
st.button(final_decision(PREDICTION))