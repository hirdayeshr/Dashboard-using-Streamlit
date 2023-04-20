import streamlit as st
import  numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit as st
import datetime

from PIL import Image

image = Image.open("SPICEJET.png")
st.sidebar.image(image)
st.header(" Airline(spicejet) EDA and Dashboard with parameters ")
st.sidebar.header("USER INPUT PARAMETER")
data_1=pd.read_csv("C:\\Users\\Saroj Raghav\\PycharmProjects\\Computer_Vision_2023\\Airlinewise Monthly International Air Traffic To And From The Indian Territory.csv")
data_3=pd.read_csv("C:\\Users\\Saroj Raghav\\PycharmProjects\\Computer_Vision_2023\\Countrywise Quarterly International Air Traffic To And From The Indian Territory.csv")

def user_input_feature():

    options = st.sidebar.multiselect(
        'Select Carrier Type',
        data_1["CARRIER TYPE"].unique(), default="DOMESTIC")

    #st.write('You selected:', options)
    CT = data_1[data_1["CARRIER TYPE"].isin(options)]
    st.write(CT)


    options_1 = st.sidebar.multiselect(
        'Select Country',
        data_3["COUNTRY NAME"].unique())

    filtered = data_3[data_3["COUNTRY NAME"].isin(options_1)]

    st.write(filtered)
    fig = px.scatter(filtered, x="COUNTRY NAME", y="PASSENGERS FROM INDIA")
    st.plotly_chart(fig)

    options_2 = st.sidebar.multiselect(
        'Select Airline name',
        data_1["AIRLINE NAME"].unique())

    filtered_1 = data_1[data_1["AIRLINE NAME"].isin(options_2)]

    st.write(filtered_1)
    fig_2 = px.scatter(filtered_1, x="FREIGHT TO INDIA", y="FREIGHT FROM INDIA")
    st.plotly_chart(fig_2)

    options_3 = st.sidebar.multiselect(
        'Select by QUARTER',
        data_1["QUARTER"].unique())

    filtered_2 = filtered_1[filtered_1["QUARTER"].isin(options_3)]

    st.write(filtered_2)
    fig_2 = px.scatter(filtered_2, x="FREIGHT TO INDIA", y="FREIGHT FROM INDIA",color="AIRLINE NAME")
    st.plotly_chart(fig_2)


    options_3 = st.sidebar.multiselect(
        'Select by YEAR',
        data_1["YEAR"].unique())

    filtered_3 = filtered_2[filtered_2["YEAR"].isin(options_3)]
    st.write(filtered_3)
    fig_2 = px.scatter(filtered_3, x="FREIGHT TO INDIA", y="FREIGHT FROM INDIA", color="AIRLINE NAME")
    st.plotly_chart(fig_2)






    numofAirlines = st.sidebar.slider("Select the top used Airlines")
    airline_passengers = data_1.groupby('AIRLINE NAME')['PASSENGERS FROM INDIA'].sum()
    sorted_airline_passengers = airline_passengers.sort_values(ascending=False)

    top_5_airlines = sorted_airline_passengers.head(numofAirlines)

    #st.write(airline_passengers)
    #st.write(sorted_airline_passengers)
    st.write(top_5_airlines)

    #fig = px.scatter(top_5_airlines, x="PASSENGERS TO INDIA", y="AIRLINE NAME", color="CARRIER TYPE")
    #st.plotly_chart(fig)


    fig,ax = plt.subplots()
    ax.hist(top_5_airlines)
    ax.set_title("Popularity through Airline Name")
    ax.set_xlabel("Airline")
    st.pyplot(fig)






user_input_feature()

# st.write(df)
# iris= datasets.load_iris()
# X=iris.data
# y=iris.target
# clf=RandomForestClassifier()
# clf.fit(X,y)
# prediction=clf.predict(df)
# prediction_proba=clf.predict_proba(df)
#
# st.subheader('Class label and their corresponding index number')
# st.write(iris.target_names)
#
# st.subheader('prediction')
# st.write(iris.target_names[prediction])
# #st.write(prediction)
#
# st.subheader("Prediction Probability")
# st.write(prediction_proba)
