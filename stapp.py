import streamlit as st
import tensorflow as tf
import pandas as pd

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

# Define a function to preprocess the data

def make_prediction(data, model):
    prediction = model.predict(data)
    return prediction
