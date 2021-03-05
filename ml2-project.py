#import Library
import numpy as np
import pandas as pd
import streamlit as st 
import pickle
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import streamlit.components.v1 as components



import matplotlib.pyplot as plt
import seaborn as sns

# to suppress warnings 
from warnings import filterwarnings
filterwarnings('ignore')

# display all columns of the dataframe
pd.options.display.max_columns = None

# display all rows of the dataframe
pd.options.display.max_rows = None
 
# to display the float values upto 6 decimal places     
pd.options.display.float_format = '{:.6f}'.format

# import train-test split 
from sklearn.model_selection import train_test_split

# import StandardScaler to perform scaling
from sklearn.preprocessing import StandardScaler 

# import various functions from sklearn
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,precision_score,recall_score,cohen_kappa_score,roc_curve,auc

# import the functions for visualizing the decision tree
import pydotplus
from IPython.display import Image  


#App config
st.title('')
#st.progress(progress_variable_1_to_100)
#Sidebar

df=st.sidebar.text_input('File')

try:
    with open(df, "r+") as data:
       data = pd.DataFrame(data)
except FileNotFoundError:
    st.error('File not found.')

page_select = st.sidebar.selectbox("Modules",("Introduction", "EDA","Modeling","Prdiction","Insights"))
if page_select == "Modeling":
      sampling_tech=st.sidebar.selectbox("Sampling Techniques",("Randon", "Stratified","SMOTE"))
      if sampling_tech=="Randon":
            rand=st.sidebar.slider('Sample', min_value=0, max_value=1000)
      elif sampling_tech=="SMOTE":
            smt=st.sidebar.select_slider('Random State', options=[1,'2'])


st.sidebar.button('RUN')
HtmlFile = open(r'E:\MTech\ML 2\profile_report.html', 'rb')
source_code = HtmlFile.read() 

# loading the trained model classifier is the model name
pickle_in_dt_tunned_classifier = open('C:\Users\KASISH\Documents\ML2 Projects\dt_tunned_classifier.pkl', 'rb')
pickle_in_gnb_classifier = open('C:\Users\KASISH\Documents\ML2 Projects\gnb_classifier.pkl', 'rb')
pickle_in_knn_tunned_classifier = open('C:\Users\KASISH\Documents\ML2 Projects\knn_tunned_classifier.pkl', 'rb')
pickle_in_logit_classifier = open('C:\Users\KASISH\Documents\ML2 Projects\logit_classifier.pkl', 'rb')
pickle_in_ml2_classifier = open('C:\Users\KASISH\Documents\ML2 Projects\ml2_classifier.pkl', 'rb')
pickle_in_dt_stack_model_classifier = open('C:\Users\KASISH\Documents\ML2 Projects\stack_model_classifier_updated.pkl', 'rb')

classifier = pickle.load(pickle_in)
 
@st.cache()
  
# defining the function which will make the prediction using the data which the user inputs 
def prediction(Gender, Married, Education, Last_payment_day, Last_bill_amount):   
 
    # Pre-processing user input    
    if Gender == "Male":
        SEX = 1
    else:
        SEX = 2
 
    if Married == "Unmarried":
        MARRIAGE = 1
    elif Married == "Married":  
        MARRIAGE = 2
    else:
        MARRIAGE = 3
        
    if Education == "Graduate School":
        EDUCATION = 1
    elif Education == "University":  
        EDUCATION = 2
    elif Education == "High School":
        EDUCATION = 3    
    else:
        EDUCATION = 4    
 
    
    PAY_0 = Last_payment_day
    BILL_AMT1 = Last_bill_amount
  
 
    # Making predictions 
    prediction = classifier.predict( 
        [[SEX, MARRIAGE, EDUCATION, PAY_0, BILL_AMT1]])
     
    if prediction == 0:
        pred = 'not A Defaulter'
    else:
        pred = 'a Defaulter'
    return pred
      
  
# this is the main function in which we define our webpage  
def main(): 
    if page_select =="Prdiction": 
    	# front end elements of the web page 
    	html_temp = """ 
    	<div style ="background-color:white;padding:13px"> 
    	<h1 style ="color:black;text-align:center;">Credit Card Defaulter Prediction</h1> 
    	</div> 
    	"""
      
# display the front end aspect
    	#st.markdown(html_temp, unsafe_allow_html = True)
	#model_pkl=st.selectbox("Models",("DT Tunned", "GNB","KNN","Logit","Stack"))
	#following lines create boxes in which user can enter data required to make prediction 
    	#Education = st.selectbox("Applicants Education",("Graduate School","University","High School","Others"))
        #Gender = st.selectbox('Gender',("Male","Female"))
        Married = st.selectbox('Marital Status',("Unmarried","Married","Others")) 
    	PAY_0 = st.text("First Payment",("Yes","No"))
    	PAY_1 = st.selectbox("Second Payment",("Yes","No"))
    	PAY_2 = st.selectbox("Third Payment",("Yes","No"))
    	PAY_3 = st.selectbox("Fourth Payment",("Yes","No"))
    	PAY_4 = st.selectbox("Fifth Payment",("Yes","No"))
    	PAY_5 = st.selectbox("Sixth Payment",("Yes","No"))
    	PAY_6 = st.selectbox("Seventh Payment",("Yes","No"))
    	Last_bill_amount = st.number_input("Last Pay Amount")
    	result =""
      
    	# when 'Predict' is clicked, make the prediction and store it 
    	if st.button("Predict"): 
        	result = prediction(Education,Gender, Married, PAY_0,PAY_1,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6,Last_bill_amount) 
        	st.success('Customer is {}'.format(result))
    elif page_select == "EDA":
    	print(source_code)
    	components.html(source_code, height = 1200)
    elif page_select == "Introduction":
        #pr = ProfileReport(data, explorative=True)
        #pr.to_file('E:\MTech\ML 2\profile_report.html')
        #st_profile_report(pr)
        st.write(data)


if __name__=='__main__': 
    main()