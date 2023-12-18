#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install scikit-learn')


# In[2]:


pip install streamlit


# In[3]:


pip install numpy


# In[4]:


pip install pandas


# In[5]:


pip install tensorflow


# In[ ]:


import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('weights.best.hdf5')
    return model

model = load_model()

# Streamlit App Title
st.write("# Bike Sharing Demand Prediction")

# CSV File Uploader
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

# Initialize a StandardScaler instance

scaler = StandardScaler()

# Preprocess the input and make a prediction
def preprocess_data(df):
    X = df.drop(columns=["cnt", "dteday"])  # Adjust the columns as per your data
    X_scaled = scaler.fit_transform(X)  # Scaling the features
    return X_scaled

def make_prediction(data, model):
    prediction = model.predict(data)
    return prediction

if uploaded_file is not None:
    # Read and preprocess the data
    input_df = pd.read_csv(uploaded_file)
    preprocessed_data = preprocess_data(input_df)

    if st.sidebar.button('Predict Demand'):
        prediction = make_prediction(preprocessed_data, model)
        # Display the prediction
        st.subheader("Bike Sharing Demand Prediction")
        st.write(f"Predicted Bike Demand: {prediction[0][0]:.2f} bikes")
else:
    st.sidebar.write("Please upload a CSV file to make predictions.")


# In[ ]:




