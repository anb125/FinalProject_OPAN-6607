import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_social_media_usage.csv
data = load_social_media_usage.csv()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logReg = LogisticRegression(class_weight='balanced', random_state=42)
logReg.fit(x_train, y_train)

st.title("Predicting LinkedIn Users")


sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)
st.subheader("Make a Prediction")

user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

if st.button("Predict"):
    prediction = logReg.predict(user_input)
    st.write(f"And the Prediction is: {prediction[0]}")

st.subheader("Model Data Data Model")
st.write(f"x_train shape: {x_train.shape}")
st.write(f"x_test shape: {x_test.shape}")
st.write(f"y_train shape: {y_train.shape}")
st.write(f"y_test shape: {y_test.shape}")

import joblib
joblib.dump(logReg, 'logistic_regression_model.pkl')

logReg = joblib.load('logistic_regression_model.pkl')



