pip install scikit-learn
import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Streamlit App Title
st.write("# Bike Sharing Demand Prediction")

# CSV File Uploader
uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])

# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('weights.best.hdf5')
    return model

model = load_model()

def preprocess_data(input_df):
    # Assuming your data preprocessing steps here
    # Make sure it's compatible with your training data preprocessing
    # You can use the same preprocessing steps you used for your training data
    return input_df

def make_prediction(data, model):
    prediction = model.predict(data)
    return prediction

if uploaded_file is not None:
    try:
        # Use pandas to read the uploaded CSV file
        df = pd.read_csv(uploaded_file)

        # Preprocess the data (make sure it matches your training data preprocessing)
        df = preprocess_data(df)

        # Extract feature columns (X) and target values (Y)
        X = df['TAX'].values.reshape(-1, 1)
        Y = df['MEDV'].values

        # Scale the input features using the same scaler as in training
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Split the data into training and testing sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=7)

        # Train the model
        model.fit(X_train, Y_train, epochs=150, batch_size=10, verbose=0)

        # Evaluate the model on the test data
        scores = model.evaluate(X_test, Y_test)
        st.subheader("Model Evaluation")
        st.write(f"Mean Squared Error: {scores}")

        # Make predictions
        if st.sidebar.button('Predict Demand'):
            prediction = make_prediction(X_test, model)
            st.subheader("Bike Sharing Demand Prediction")
            st.write(f"Predicted Bike Demand: {prediction[0][0]:.2f} bikes")

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.sidebar.write("Please upload a CSV file to make predictions.")
