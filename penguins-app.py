#import Library
import numpy as np
import pandas as pd
import streamlit as st 
import pickle

#App config
st.title('Credit Card Defaulter Prediction System')
#st.progress(progress_variable_1_to_100)
#Sidebar

# loading the trained model classifier is the model name
pickle_in_dt_tunned_classifier = open(r'C:\Users\KASISH\Documents\ML2 Projects\dt_tuned_classifier.pkl', 'rb')
pickle_in_gnb_classifier = open(r'C:\Users\KASISH\Documents\ML2 Projects\gnb_classifier.pkl', 'rb')
pickle_in_knn_tunned_classifier = open(r'C:\Users\KASISH\Documents\ML2 Projects\knn_tuned_classifier.pkl', 'rb')
pickle_in_logit_classifier = open(r'C:\Users\KASISH\Documents\ML2 Projects\logit_classifier.pkl', 'rb')
pickle_in_dt_stack_model_classifier = open(r'C:\Users\KASISH\Documents\ML2 Projects\stack_model_classifier_updated.pkl')

model_pkl=st.selectbox("Models",("DT Tunned", "GNB","KNN","Logit","Stack"))

@st.cache()
  
# defining the function which will make the prediction using the data which the user inputs 
def prediction(EDUCATION,SEX,MARRIAGE, PAY_0,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6,PAY_AMT_1):   
 
    # Pre-processing user input    
    if SEX == "Male":
        SEX = 1
    else:
        SEX = 2
 
    if MARRIAGE == "Married":
        MARRIAGE = 1
    elif MARRIAGE == "Single":  
        MARRIAGE = 2
    else:
        MARRIAGE = 3
        
    if EDUCATION == "Graduate School":
        EDUCATION = 1
    elif EDUCATION == "University":  
        EDUCATION = 2
    elif EDUCATION == "High School":
        EDUCATION = 3    
    else:
        EDUCATION = 4    
 
    
    #PAY_0 = Last_payment_day
    #PAY_AMT_1 = Last_bill_amount
    classifier = pickle.load(pickle_in_dt_tunned_classifier)
    if model_pkl == "DT Tunned":
      classifier = pickle.load(pickle_in_dt_tunned_classifier)
      if model_pkl =="GNB":
            classifier = pickle.load(pickle_in_gnb_classifier)
      elif model_pkl=="KNN":
            classifier = pickle.load(pickle_in_knn_tunned_classifier)
      elif model_pkl=="Logit":
            classifier = pickle.load(pickle_in_logit_classifier)
      elif model_pkl=="Stack":
            classifier = pickle.load(pickle_in_dt_stack_model_classifier)

    # Making predictions 
    prediction = classifier.predict( 
        [[EDUCATION,SEX,MARRIAGE,PAY_0,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6,PAY_AMT_1]])
     
    if prediction == 0:
        pred = 'not be a defaulter'
    else:
        pred = 'be a defaulter'
    return pred
      
  
# this is the main function in which we define our webpage  
def main(): 
     # front end elements of the web page 
      
     # display the front end aspect
     
     #following lines create boxes in which user can enter data required to make prediction 
     EDUCATION = st.selectbox('Custommer Highest Education',("Graduate School","University","High School","Others"))
     SEX = st.selectbox('Gender',("Male","Female"))
     MARRIAGE = st.selectbox('Marital Status',("Married","Single","Others")) 
     PAY_0 = st.select_slider('First Payment', options=[-2,-1,0,1,2])
     PAY_2 = st.select_slider('Second Payment', options=[-2,-1,0,1,2])
     PAY_3 = st.select_slider('Third Payment', options=[-2,-1,0,1,2])
     PAY_4 = st.select_slider('Fourth Payment', options=[-2,-1,0,1,2])
     PAY_5 = st.select_slider('Fifth Payment', options=[-2,-1,0,1,2])
     PAY_6 = st.select_slider('Sixth Payment', options=[-2,-1,0,1,2])
     PAY_AMT_1 = st.number_input("Pay Amount")
     result =""
      
     # when 'Predict' is clicked, make the prediction and store it  
     if st.button("Predict"):
          result = prediction(EDUCATION,SEX,MARRIAGE,PAY_0,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6,PAY_AMT_1) 
          st.success('This customer may {}'.format(result))
          print(PAY_AMT_1)

if __name__=='__main__': 
    main()
