import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np


# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('weights.best.hdf5')
    return model

model = load_model()

# Streamlit App Title
st.write("# Bike Sharing Demand Prediction")

# CSV File Uploader
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

def make_prediction(data, model):
    prediction = model.predict(data)
    return prediction

if uploaded_file is not None:
    # Use pandas to read the uploaded CSV file
    df = pd.read_csv(uploaded_file)

    # Display the data
    st.dataframe(df)

else:
    st.sidebar.write("Please upload a CSV file to make predictions.")
