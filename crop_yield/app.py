import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import requests

model_url = 'https://github.com/Sai-Santhosh/DataScienceProjects/raw/main/crop_yield/lstm_crop_yield.h5'
transformer_url = 'https://github.com/Sai-Santhosh/DataScienceProjects/raw/main/crop_yield/power_transformer_yeojohnson.pkl' 

# Download the file
r = requests.get(model_url)
with open('lstm_crop_yield.h5', 'wb') as f:
    f.write(r.content)

# Power Transformer Download the file
r = requests.get(transformer_url)
with open('power_transformer_yeojohnson.pkl', 'wb') as f:
    f.write(r.content)

# Load the saved LSTM model and PowerTransformer
model = load_model('lstm_crop_yield.h5')
power_transformer = joblib.load('power_transformer_yeojohnson.pkl')

# Streamlit page configuration
st.title('Crop Yield Prediction')
st.write('Please enter the details for prediction:')
training_columns=['Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Crop_Arhar/Tur',
       'Crop_Bajra', 'Crop_Banana', 'Crop_Barley', 'Crop_Black pepper',
       'Crop_Cardamom', 'Crop_Cashewnut', 'Crop_Castor seed', 'Crop_Coconut ',
       'Crop_Coriander', 'Crop_Cotton(lint)', 'Crop_Cowpea(Lobia)',
       'Crop_Dry chillies', 'Crop_Garlic', 'Crop_Ginger', 'Crop_Gram',
       'Crop_Groundnut', 'Crop_Guar seed', 'Crop_Horse-gram', 'Crop_Jowar',
       'Crop_Jute', 'Crop_Khesari', 'Crop_Linseed', 'Crop_Maize',
       'Crop_Masoor', 'Crop_Mesta', 'Crop_Moong(Green Gram)', 'Crop_Moth',
       'Crop_Niger seed', 'Crop_Oilseeds total', 'Crop_Onion',
       'Crop_Other  Rabi pulses', 'Crop_Other Cereals',
       'Crop_Other Kharif pulses', 'Crop_Other Summer Pulses',
       'Crop_Peas & beans (Pulses)', 'Crop_Potato', 'Crop_Ragi',
       'Crop_Rapeseed &Mustard', 'Crop_Rice', 'Crop_Safflower',
       'Crop_Sannhamp', 'Crop_Sesamum', 'Crop_Small millets', 'Crop_Soyabean',
       'Crop_Sugarcane', 'Crop_Sunflower', 'Crop_Sweet potato', 'Crop_Tapioca',
       'Crop_Tobacco', 'Crop_Turmeric', 'Crop_Urad', 'Crop_Wheat',
       'Crop_other oilseeds', 'Season_Kharif     ', 'Season_Rabi       ',
       'Season_Summer     ', 'Season_Whole Year ', 'Season_Winter     ',
       'State_Arunachal Pradesh', 'State_Assam', 'State_Bihar',
       'State_Chhattisgarh', 'State_Delhi', 'State_Goa', 'State_Gujarat',
       'State_Haryana', 'State_Himachal Pradesh', 'State_Jammu and Kashmir',
       'State_Jharkhand', 'State_Karnataka', 'State_Kerala',
       'State_Madhya Pradesh', 'State_Maharashtra', 'State_Manipur',
       'State_Meghalaya', 'State_Mizoram', 'State_Nagaland', 'State_Odisha',
       'State_Puducherry', 'State_Punjab', 'State_Sikkim', 'State_Tamil Nadu',
       'State_Telangana', 'State_Tripura', 'State_Uttar Pradesh',
       'State_Uttarakhand', 'State_West Bengal']
# Create input fields for the features
# Replace these with dropdowns or appropriate input methods for your categorical features
crop = st.selectbox('Crop', ['Rice', 'Maize', 'Moong(Green Gram)', 'Urad', 'Groundnut', 'Sesamum',
       'Potato', 'Sugarcane', 'Wheat', 'Rapeseed &Mustard', 'Bajra', 'Jowar',
       'Arhar/Tur', 'Ragi', 'Gram', 'Small millets', 'Cotton(lint)', 'Onion',
       'Sunflower', 'Dry chillies', 'Other Kharif pulses', 'Horse-gram',
       'Peas & beans (Pulses)', 'Tobacco', 'Other  Rabi pulses', 'Soyabean',
       'Turmeric', 'Masoor', 'Ginger', 'Linseed', 'Castor seed', 'Barley',
       'Sweet potato', 'Garlic', 'Banana', 'Mesta', 'Tapioca', 'Coriander',
       'Niger seed', 'Jute', 'Coconut ', 'Safflower', 'Arecanut', 'Sannhamp',
       'Other Cereals', 'Cowpea(Lobia)', 'Cashewnut', 'Black pepper',
       'other oilseeds', 'Moth', 'Khesari', 'Cardamom', 'Guar seed',
       'Oilseeds total', 'Other Summer Pulses'])  # Replace with actual crop names
crop_year = st.number_input('Crop Year', min_value=1997, max_value=2020, step=1)
season = st.selectbox('Season', ['Kharif     ', 'Rabi       ', 'Whole Year ', 'Summer     ',
       'Autumn     ', 'Winter     '])  # Replace with actual seasons
state = st.selectbox('State', ['Karnataka', 'Andhra Pradesh', 'West Bengal', 'Chhattisgarh', 'Bihar',
       'Madhya Pradesh', 'Uttar Pradesh', 'Tamil Nadu', 'Gujarat',
       'Maharashtra', 'Uttarakhand', 'Odisha', 'Assam', 'Nagaland',
       'Puducherry', 'Meghalaya', 'Haryana', 'Jammu and Kashmir',
       'Himachal Pradesh', 'Kerala', 'Manipur', 'Tripura', 'Mizoram',
       'Telangana', 'Punjab', 'Arunachal Pradesh', 'Jharkhand', 'Goa',
       'Sikkim', 'Delhi'])  # Replace with actual states

# Numeric features
area = st.number_input('Area', min_value=0.5, format="%.2f")
production = st.number_input('Production', min_value=0)
annual_rainfall = st.number_input('Annual Rainfall', min_value=0.0, format="%.2f")
fertilizer = st.number_input('Fertilizer', min_value=0.0, format="%.2f")
pesticide = st.number_input('Pesticide', min_value=0.0, format="%.2f")

# Predict button
if st.button('Predict'):
    # Create a DataFrame from the input data
    ## Compile user inputs into a DataFrame
    input_data = pd.DataFrame([[crop, season, state, area, production, annual_rainfall, fertilizer, pesticide]],
    columns=['Crop', 'Season', 'State', 'Area', 'Production', 'Annual_Rainfall', 'Fertilizer', 'Pesticide'])

    # One-hot encode the categorical features
    input_data = pd.get_dummies(input_data, columns=['Crop', 'Season', 'State'])

    # Align columns with the training data columns
    for col in training_columns:
        if col not in input_data.columns:
            input_data[col] = 0  # Add missing columns with default value of 0
    input_data = input_data.reindex(columns=training_columns, fill_value=0)


     # Transform the input data using the saved PowerTransformer
    input_transformed = power_transformer.transform(input_data)

    # Reshape input for LSTM model
    input_reshaped = input_transformed.reshape((1, 1, -1))

    # Make prediction
    prediction = model.predict(input_reshaped)

    # Display the prediction
    st.write('Predicted Crop Yield:', prediction[0][0])
