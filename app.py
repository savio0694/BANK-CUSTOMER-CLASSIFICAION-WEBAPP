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
from PIL import Image


bankdata_clean=pd.read_csv('run-1608626939481-part-r-00000',sep=',')
bankdata=bankdata_clean[(bankdata_clean['balance']<2000)&(bankdata_clean['balance']>-1000)]
bankdata1=pd.read_csv('bank2.csv',sep=',')
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


st.subheader('*To view the entire dataset with the original features,Exploratory data analysis,feature selecion,Adjustments forClass imbalance etc. view the')
components.html("""<a href="https://nbviewer.jupyter.org/github/savio0694/BANK-CUSTOMER-CLASSIFICATION-WEBAPP/blob/main/customer-classification.ipynb" target="_blank">JUPYTER NOTEBOOK</a> """)


st.text('')

st.subheader('A sample of raw data')
st.dataframe(bankdata1.head(10))


st.subheader('The raw data is initially kept in the in folder of an amazon S3 bucket,Amazon S3 is a simple web services interface that you can use to store and retrieve any amount of data, at any time, from anywhere on the web.')
st.image('https://github.com/savio0694/BANK-CUSTOMER-CLASSIFICATION-WEBAPP/blob/main/images/Capture2.PNG?raw=true')

st.subheader('The raw data is read and transformed using AWS GLUE,AWS Glue is a serverless data integration service that makes it easy to discover, prepare, and combine data for analytics, machine learning, and application development.')
st.image('https://github.com/savio0694/BANK-CUSTOMER-CLASSIFICATION-WEBAPP/blob/main/images/Capture6.png?raw=true')

st.image('https://github.com/savio0694/BANK-CUSTOMER-CLASSIFICATION-WEBAPP/blob/main/images/Capture3.PNG?raw=true')
st.image('https://github.com/savio0694/BANK-CUSTOMER-CLASSIFICATION-WEBAPP/blob/main/images/Capture1.PNG?raw=true')

st.subheader('We create a crawler to crawl the s3 bucket and define an ETL job to run our extrac,transform,load SPARK script given below.Apache Spark is a unified analytics engine for big data processing, with built-in modules for streaming, SQL, machine learning and graph processing.')

st.text('''#########################################
### IMPORT LIBRARIES AND SET VARIABLES
#########################################

#Import pyspark modules
from pyspark.context import SparkContext
import pyspark.sql.functions as f
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
#Import glue modules
from awsglue.utils import getResolvedOptions
from awsglue.context import GlueContext
from awsglue.dynamicframe import DynamicFrame
from awsglue.job import Job

#Initialize contexts and session
spark_context = SparkContext.getOrCreate()
glue_context = GlueContext(spark_context)
session = glue_context.spark_session

#Parameters
glue_db = "db1"
glue_tbl = "in"
s3_write_path = "s3://customers123/out/"

#########################################
### EXTRACT (READ DATA)
#########################################

dynamic_frame_read = glue_context.create_dynamic_frame.from_catalog(database = glue_db, table_name = glue_tbl)

#Convert dynamic frame to data frame to use standard pyspark functions
data = dynamic_frame_read.toDF()

#########################################
### TRANSFORM (MODIFY DATA)
#########################################
df=data.select('age','balance','housing','loan','y')
indexers=[StringIndexer(inputCol=columns,outputCol=columns+'-out') for columns in ['housing','loan','y']]

pipeline=Pipeline(stages=indexers)
df=pipeline.fit(df).transform(df)

df=df.select('age','balance','housing-out','loan-out','y-out')
df=df.withColumnRenamed('housing-out','housing_loan') \
   .withColumnRenamed('loan-out','personal_loan') \
   .withColumnRenamed('y-out','Term_deposit')

#df=df.filter((df.balance>1-1000)&(df.balance<1800)).collect()



#########################################
### LOAD (WRITE DATA)
#########################################


#Convert back to dynamic frame
dynamic_frame_write = DynamicFrame.fromDF(df, glue_context, "dynamic_frame_write")

#Write data back to S3
glue_context.write_dynamic_frame.from_options(
frame = dynamic_frame_write,
connection_type = "s3",
connection_options = {
"path": s3_write_path,

},
format = "csv"
)'''
)

st.subheader('The spark program creates he output file in the out folder in S3')
st.image('https://github.com/savio0694/BANK-CUSTOMER-CLASSIFICATION-WEBAPP/blob/main/images/Capture5.PNG?raw=true')


st.text('I will not access the file directly from S3 so as to avoid sharing my aws keys publicly,instead I have downloaded the file.')

st.subheader('Final_data_features')
st.dataframe(bankdata)

st.header('FINAL RESULT AFTER RUNNING THROUGH  RandomForestClassifier')

housing = st.sidebar.radio(
     "Does the customer have a housing loan?",
    ('YES', 'NO'))

if housing == 'YES':
     VAL=1

else:
     VAL=0


personal=st.sidebar.radio(
     "Does the customer have a personal loan?",
    ('YES', 'NO'))

if personal == 'YES':
     VAL1=1

else:
     VAL1=0

age = st.sidebar.slider('How old is the customer', 18, 100, 25)

balance= st.sidebar.slider('What is the customer balance?', -500, 2000, 250)



#---------------LAYOUT-------------------------------#


MODEL=make_model(bankdata)

#--------------MAKE-TEST_DATAFRAME-------------#


def make_frame(VAL,VAL1,age,balance):
      data={
      "age" : age,
      "balance":balance,
      "housing_loan" : VAL,
      "personal_loan" : VAL1
      }
      df = pd.DataFrame(data,index=[1])
      return df





df = make_frame(VAL,VAL1,age,balance)


PREDICTION=MODEL.predict(df)


#---FINAL DECISION-------#
def final_decision(PREDICTION):
    if PREDICTION[0]==0:
       return " The customer is UNLIKELY to have a TERM DEPOSIT"
    else:
        return " The customer is LIKELY to have a TERM DEPOSIT"

st.button(final_decision(PREDICTION))
