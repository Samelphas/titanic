import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Load the trained model
with open('titanic.pkl', 'rb') as file:
    model = pickle.load(file)
y_pred = model.predict([[1,2,3,4,5,6,7]])
print(y_pred)

# Streamlit app
st.title('Titanic Survival Prediction')
st.write('This app predicts whether a passenger survived the Titanic disaster or not based on their features.')

# Input fields
pclass = st.selectbox('Pclass', [1, 2, 3])
sex = st.selectbox('Sex', ['male', 'female'])
age = st.slider('Age', 0, 100, 25)
sibsp = st.slider('Siblings/Spouses Aboard', 0, 8, 0)
parch = st.slider('Parents/Children Aboard', 0, 6, 0)
fare = st.slider('Fare', 0.0, 500.0, 10.0)
embarked = st.selectbox('Embarked', ['C', 'Q', 'S'])

# Convert input to DataFrame
input_data = {
    'Pclass': pclass,
    'Sex': sex,
    'Age': age,
    'SibSp': sibsp,
    'Parch': parch,
    'Fare': fare,
    'Embarked': embarked
}
input_df = pd.DataFrame([input_data])

# Preprocess the input data
input_df['Sex'] = label_encoder.fit_transform(input_df['Sex'])
input_df['Embarked'] = label_encoder.fit_transform(input_df['Embarked'])

# Make prediction
if st.button('Predict'):
    prediction = model.predict(input_df)
    if prediction[0] == 1:
        st.success('Survived')
    else:
        st.error('Did not survive')
