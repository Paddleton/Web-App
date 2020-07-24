# Decscription: If someone has diabetes using  machine learning and python

# Import the libraries
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

# Create a title and subtitle
st.write(""" # Diabetes Detection
""")
# Open and display an image
image = Image.open('C:/Users/admin/Desktop/Project2/Imageforpage.png')
st.image(image, caption='ML', use_column_width=True)
# Get data
df = pd.read_csv('C:/Users/admin/Desktop/Project2/diabetes.csv')

# Set a subheader
st.subheader('Data Information:')
# Show the data as table
st.dataframe(df)
# Show statistics
st.write(df.describe())
# Show the data as chart
chart = st.bar_chart(df)
# Split the data independent 'X' and dependent 'y'
X = df.iloc[:, 0:7].values
Y = df.iloc[:, -1].values
# Split into training and testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)


# Feature input from the user
def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('glucose', 0, 199, 117)
    blood_pressure = st.sidebar.slider('blood_pressure', 0, 122, 72)
    skin_thickness = st.sidebar.slider('skin_thickness', 0, 17, 3)
    insulin = st.sidebar.slider('insulin', 0.0, 846.0, 30.0)
    BMI = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    DPF = st.sidebar.slider('DPF', 0.078, 2.43, 0.3725)
    age = st.sidebar.slider('age', 21, 81, 29)

    # Store a dictionary into a variable
    user_data = {'pregnancies': pregnancies,
                 'glucose': glucose,
                 'blood_pressure': blood_pressure,
                 'skin_thickness': skin_thickness,
                 'BMI': BMI,
                 'DPF': DPF,
                 'age': age
                 }
    # Transform the data to data frame
    features = pd.DataFrame(user_data, index=[0])
    return features


# Store the user input in a variable
user_input = get_user_input()

# Set a sub header and display user input
st.subheader('User Input:')
st.write(user_input)
# Create and train the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)
# Show the model's metrics
st.subheader('Model Test Accuracy Score:')
st.write(str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100) + '%')

# Store the model's predictions in a variable
prediction = RandomForestClassifier.predict(user_input)

# Set a subheader and display the classification
st.subheader('Classification: ')
st.write(prediction)
