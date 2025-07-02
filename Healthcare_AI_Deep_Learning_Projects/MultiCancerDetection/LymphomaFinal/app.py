
import os
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np
from io import BytesIO
import requests

model_url = 'https://github.com/Sai-Santhosh/DataScienceProjects/raw/main/Healthcare_AI_Deep_Learning_Projects/MultiCancerDetection/LymphomaFinal/Ensemble.h5'

# Download the file
r = requests.get(model_url)
with open('Ensemble.h5', 'wb') as f:
    f.write(r.content)

# Now you can load your model
model = load_model('Ensemble.h5')

def preprocess_image(image):
    image = image.resize((224, 224)) 
    image = np.array(image)
    image = np.expand_dims(image, axis=0)  
    return image

# Function to display the introduction section
def show_introduction():
    st.title('Lymphoma Classification using Ensemble of CNN Models')
    st.markdown("""
    This application is designed to classify lymphoma using an ensemble of CNN models.
    Simply upload an image and the model will predict the type of lymphoma.
    For more information of this project, visit the [IEEE published paper](https://www.google.com).
    """)

# Function to display the prediction section
def show_prediction():
    st.header("Prediction")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        if st.button('Predict'):
            preprocessed_image = preprocess_image(image)
            prediction = model.predict(preprocessed_image)
            class_names = ["Chronic Lymphocytic Leukemia", "Follicular Lymphoma", "Mantle Cell Lymphoma"]
            predicted_class_index = np.argmax(prediction)
            pred_class = class_names[predicted_class_index]
            pred_score = prediction[0][predicted_class_index]
            st.write(f"The image given has {pred_class} and it has been predicted with a confidence score of {pred_score:.2f}")


# Function to display the accuracy graphs section
def show_accuracy_graphs():
    st.header("Accuracy Graphs")

    # URLs for the graphs
    graph_urls = {
        'Accuracy': 'https://github.com/Sai-Santhosh/DataScienceProjects/raw/main/Healthcare_AI_Deep_Learning_Projects/MultiCancerDetection/LymphomaFinal/Accuracy.png',
        'Loss': 'https://github.com/Sai-Santhosh/DataScienceProjects/raw/main/Healthcare_AI_Deep_Learning_Projects/MultiCancerDetection/LymphomaFinal/Loss.png',
        'Confusion Matrix': 'https://github.com/Sai-Santhosh/DataScienceProjects/raw/main/Healthcare_AI_Deep_Learning_Projects/MultiCancerDetection/LymphomaFinal/confmatrix.png'
    }

    for graph_name, graph_url in graph_urls.items():
        # Download the image
        response = requests.get(graph_url)
        image = Image.open(BytesIO(response.content))

        # Display the image
        st.image(image, caption=graph_name, use_column_width=True)
    

# Function to check if the user is logged in
def is_user_logged_in():
    return 'logged_in' in st.session_state and st.session_state.logged_in

# # Function to display the login form
def login_form():
    st.title("Lymphoma Classification System")
    form = st.form(key='login_form')
    username = form.text_input("Username")
    password = form.text_input("Password", type="password")
    login_button = form.form_submit_button("Login")
    if login_button:
        if username == "admin" and password == "password":  
            st.session_state.logged_in = True
            st.experimental_rerun()
        else:
            st.error("Incorrect Username/Password")

# Main app
def main_app():
    st.sidebar.title("Lymphoma Classification")
    app_mode = st.sidebar.radio("Go to", ["Introduction", "Prediction", "Validation of Model"])

    if app_mode == "Introduction":
        show_introduction()
    elif app_mode == "Prediction":
        show_prediction()
    elif app_mode == "Validation of Model":
        show_accuracy_graphs()

# Main
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if is_user_logged_in():
    main_app()
else:
    login_form()
