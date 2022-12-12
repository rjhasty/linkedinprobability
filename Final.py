#### 
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from PIL import Image

st.set_page_config(layout="wide")

st.markdown("""
<style>
.big-font {
    font-size:25px !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">Ryan Hasty - Programming II - LinkedIn Probability Final Project</p>', unsafe_allow_html=True)

image = Image.open('LI_photo.png')

st.image(image, caption='Image taken from Google')


st.title('What is the Probability that you are a LinkedIn User?')



#Education Dropdown
educ = st.selectbox("Education Level", 
             options = ["Less than High School Diploma",
                        "High School - Incomplete",
                        "High School - Graduate",
                        "Some College - No Degree",
                        "Two-Year Associate's Degree",
                        "Four Year Bachelors Degree",
                        "Some Post-Graduate or Professional Schooling - No Degree",
                        "Post-Graduate or Professional Degree"
                        ])


if educ == "Less than High School Diploma":
    educ = 1
elif educ == "High School - Incomplete":
    educ = 2
elif educ == "High School - Graduate":
    educ = 3
elif educ == "Some College - No Degree":
    educ = 4
elif educ == "Two-Year Associate's Degree":
    educ = 5
elif educ == "Four Year Bachelors Degree":
    educ = 6
elif educ == "Some Post-Graduate or Professional Schooling - No Degree":
    educ = 7
else: 
    educ = 8


#Income Dropdown
income = st.selectbox("Gross Household Income Level", 
             options = ["Less than $10,000",
                        "$10,000 to under $20,000",
                        "$20,000 to under $30,000",
                        "$30,000 to under $40,000",
                        "$40,000 to under $50,000",
                        "$50,000 to under $75,000",
                        "$75,000 to under $100,000",
                        "$100,000 to under $150,000",
                        "$150,000 or more?"
                        ])



if income == "Less than $10,000":
    income = 1
elif income == "$10,000 to under $20,000":
    income = 2
elif income == "$20,000 to under $30,000":
    income = 3
elif income == "$30,000 to under $40,000":
    income = 4
elif income == "$40,000 to under $50,000":
    income = 5
elif income == "$50,000 to under $75,000":
    income = 6
elif income == "$75,000 to under $100,000":
    income = 7
elif income == "$100,000 to under $150,000":
    income = 8
else: 
    income = 9


#Parental Status Dropdown

child = st.selectbox("Parental Status",
            options= ["Yes",
                      "No",
                        ])


if child == "Yes":
    child = 1
else:
    child = 0 

#Marital Status Dropdown

marriage = st.selectbox("Marital Status",
            options= ["Yes",
                      "No",
                        ])

if marriage == "Yes":
    marriage = 1
else:
    marriage = 0 



#Gender Dropdown
gender = st.selectbox("Gender",
            options= ["Male",
                      "Female",
                        ])

if gender == "Female":
    gender = 1
else:
    gender = 0 




#Age Dropdown

age = st.number_input('Enter Your Age',
                min_value= 1,
                max_value= 99,
                value=30)



s = pd.read_csv(r"social_media_usage.csv")

def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return(x)
                     



ss = pd.DataFrame({
    "income": np.where(s["income"] <= 9,s["income"],np.nan),
    "education": np.where(s["educ2"] <= 8, s["educ2"], np.nan),
    "parent": np.where(s["par"]==1,1,0),
    "married": np.where(s["marital"]==1,1,0),
    "gender": np.where(s["gender"] ==2,1,0),
    "age": np.where(s["age"] <= 98, s["age"], np.nan),
    "sm_li": clean_sm(s["web1h"])
})

ss = ss.dropna()


y = ss["sm_li"]
x = ss[["income","education","parent","married","gender","age"]]

  
x_train, x_test, y_train, y_test = train_test_split(x,
                                                    y,
                                                    stratify=y,       
                                                    test_size=0.3,    
                                                    random_state=750)


lr = LogisticRegression(class_weight="balanced")
lr.fit(x_train, y_train)  

y_pred = lr.predict(x_test)
y_pred_prob = lr.predict_proba(x_test)


prediction1 = pd.DataFrame({
    "income":[income],
    "education":[educ],
    "parent":[child], 
    "married": [marriage],
    "gender": [gender],
    "age":[age]
})
prob = round( lr.predict_proba(prediction1)[0,1] * 100 , 2 )

outcome = lr.predict(prediction1)

st.markdown(f"Results {prob}%")

if prob > 70:

    label = "Most Likely"

elif prob > 50:

    label = "Potentially"

elif prob < 49.9:

    label = "Probably Not"


st.markdown(f"You are ***{label}*** a LinkedIn User" )

fig = go.Figure(go.Indicator(

    mode = "gauge+number",

    value = prob,

    title = {'text': f"Probability Gauge: Are you a LinkedIn User? {label}"},

    gauge = {'axis': {"range": [0, 100], 'tickwidth': 1, 'tickcolor': "black"},
             'bar':{"color":"blue"},
             'bgcolor': "black",
             'borderwidth': 5,
             'bordercolor': "black", 
            'steps': [
                {"range": [0, 33.33], "color":"red"},
                {"range": [33.34, 66.66], "color":"yellow"},
                {"range": [66.67, 100], "color":"green"}],
             
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': prob}}))  
            

st.plotly_chart(fig)
