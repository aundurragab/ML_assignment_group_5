import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import joblib
from joblib import load
import os
import xgboost as xgb

 
#st.set_page_config(layout="wide")

def home():
    st.title("Welcome to Health App Hub! 🏥")
    st.write("Explore and try our health prediction apps. Choose wisely, stay healthy!")

    # Images with descriptions, increased size, and layout changes
    col1, col2, col3, col4 = st.columns(4)
    # Add buttons for each app with the corresponding display name
    with col1:
        image = Image.open("media/heart.jpg")
        st.image(image, width=300, use_column_width=True)
        st.markdown("<h3 style='text-align: center; color: #3cb371;'>Heart Attack Prediction</h3>", unsafe_allow_html=True)
        st.write("The **Heart Attack Prediction App** assesses the likelihood of a heart attack based on factors like age, blood pressure, cholesterol levels, and lifestyle choices. Users input their health parameters to **receive a risk estimation**, empowering them to make informed decisions about their cardiovascular health")

    with col2:
        image = Image.open("media/neumonia.jpg")
        st.image(image, width=300, use_column_width=True)
        st.markdown("<h3 style='text-align: center; color: #3cb371;'>Pneumonia Prediction</h3>", unsafe_allow_html=True)
        st.write("The **Pneumonia Prediction App** analyzes chest X-ray images to determine the **likelihood of pneumonia**. By processing these images, the app provides a **quick and accurate assessment**, aiding in timely medical intervention and treatment decisions")

    with col3:
        image = Image.open("media/smoking.jpg")
        st.image(image, width=300, use_column_width=True)
        st.markdown("<h3 style='text-align: center; color: #3cb371;'>Smoking Habits Detection</h3>", unsafe_allow_html=True)
        st.write("The **Smoking Habits Detection App** identifies smoking behaviors and evaluates related health risks. By analyzing user inputs, it **provides insights into the impact of smoking on overall health**, aiding individuals in understanding the risks associated with tobacco use and encouraging smoking cessation for improved well-being")

    with col4:
        image = Image.open("media/health.jpg")
        st.image(image, width=300, use_column_width=True)
        st.markdown("<h3 style='text-align: center; color: #3cb371;'>Healthy Habits Classification</h3>", unsafe_allow_html=True)
        st.write("The **Healthy Habits Classification App** categorizes individuals' health levels by considering physical performance metrics and personal details. It offers **personalized insights, guiding users to make informed decisions about their well-being** and encouraging healthier lifestyle choices")
    st.subheader("Disclaimer:")
    st.write("These apps provide general predictions based on the input data. Consult healthcare professionals for accurate diagnoses.")

side_bar = st.sidebar

side_bar.header("Please select one of the following apps.")
selectbox = side_bar.selectbox('Choose an app:',("Home", 'Heart Attack Prediction App', 'Pneumonia Prediction App', 'Healthy Habits Prediction App', 'Smoking Detection App', 'Feedback'))


def heart():
    
	# Load the pickled model
    with open('model.pkl', 'rb') as f:
    	classifier = pickle.load(f)

    def prediction(PhysicalHealthDays, GeneralHealth, RemovedTeeth, HadAngina, HadStroke, HadCOPD, HadKidneyDisease, HadArthritis, HadDiabetes, DeafOrHardOfHearing,
	               DifficultyWalking, SmokerStatus, ChestScan, AgeCategory, PneumoVaxEver):
        PhysicalHealthDays = PhysicalHealthDays
        GeneralHealth_Fair = 1 if GeneralHealth == "Fair" else 0
        GeneralHealth_Poor = 1 if GeneralHealth == "Poor" else 0
        RemovedTeeth_6_or_more_but_not_all = 1 if RemovedTeeth == "6 or more but not all" else 0
        RemovedTeeth_All = 1 if RemovedTeeth == "All" else 0
        RemovedTeeth_None_of_them = 1 if RemovedTeeth == "None of them" else 0
        HadAngina_Yes = 1 if HadAngina == "Yes" else 0
        HadStroke_Yes = 1 if HadStroke == "Yes" else 0
        HadCOPD_Yes = 1 if HadCOPD == "Yes" else 0
        HadKidneyDisease_Yes = 1 if HadKidneyDisease == "Yes" else 0
        HadArthritis_Yes = 1 if HadArthritis == "Yes" else 0
        HadDiabetes_Yes = 1 if HadDiabetes == "Yes" else 0
        DeafOrHardOfHearing_Yes = 1 if DeafOrHardOfHearing == "Yes" else 0
        DifficultyWalking_Yes = 1 if DifficultyWalking == "Yes" else 0
        SmokerStatus_Never_smoked = 1 if SmokerStatus == "Never smoked" else 0
        ChestScan_Yes = 1 if ChestScan == "Yes" else 0
        AgeCategory_Age_80_or_older = 1 if AgeCategory == "Age 80 or older" else 0
        PneumoVaxEver_Yes = 1 if PneumoVaxEver == "Yes" else 0

        # Making predictions 
        prediction = classifier.predict_proba([[PhysicalHealthDays, GeneralHealth_Fair, GeneralHealth_Poor,
        	RemovedTeeth_6_or_more_but_not_all, RemovedTeeth_All,
        	RemovedTeeth_None_of_them, HadAngina_Yes, HadStroke_Yes,
        	HadCOPD_Yes, HadKidneyDisease_Yes, HadArthritis_Yes,
        	HadDiabetes_Yes, DeafOrHardOfHearing_Yes, DifficultyWalking_Yes,
        	SmokerStatus_Never_smoked, ChestScan_Yes,
        	AgeCategory_Age_80_or_older, PneumoVaxEver_Yes]])[0]

        probability_of_heart_attack = prediction[1] * 100

        if probability_of_heart_attack < 0.5:
        	st.success(f'Great news! Based on the information provided you have a probability of {(probability_of_heart_attack):.2f}% to have a heart attack. Keep maintaining a healthy lifestyle.')
        	st.image('media/heart-success.webp', width=400)
        else:
        	st.error(f'Important: The prediction indicates a potential risk of a heart attack with a probability of {(probability_of_heart_attack):.2f}%. It is crucial to consult a healthcare professional.')
        	st.image('media/heart-attack.jpg', width=400)

        return probability_of_heart_attack

    st.markdown("<h1 style='text-align: center;'>🚑 Heart Attack App ❤️</h1>", unsafe_allow_html=True)

    # Centered image
    col1, col2, col3 = st.columns(3)  # Create three columns for layout
    with col2:  # Use the middle column for the centered image
        st.image('media/heart-logo.jpg', width=400, use_column_width=True)

    description = """
    *Welcome to the Heart Attack Prediction App!*
    This application is designed to provide you with a quick and simple assessment of your potential risk of experiencing a heart attack based on your current health status and lifestyle choices. Our goal is to empower you with knowledge so that you can take proactive steps towards maintaining your heart health.

    *How does it work?*
    1. *Input Your Information:* You will be asked to provide specific details about your health and lifestyle. These include your physical health days, general health condition, smoking status, and more. All of these factors play a significant role in determining your heart health.
    2. *Get Your Risk Assessment:* Once you submit your information, our advanced prediction model, which has been trained on a wide range of health data, will analyze your inputs and calculate your potential risk of having a heart attack.

    *Please Note:*
    - This tool is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
    - The prediction is based on general trends and patterns in health data, and individual variations may exist.

    *Take Charge of Your Heart Health Today!*
    By understanding your risk and taking steps to live a healthier lifestyle, you can reduce your chances of heart disease and lead a longer, healthier life.
    """
    st.markdown(description)

    st.subheader('Please enter your details:')

    PhysicalHealthDays = st.slider('How many days during the past 30 days was your physical health not good? (last 30 days)', 0, 30, 1)
    GeneralHealth = st.selectbox('Would you say that in general your health is:', ('Excellent', 'Very Good', 'Good', "Fair", "Poor"))
    RemovedTeeth = st.selectbox('How many of your permanent teeth have been removed because of tooth decay or gum disease?', ('None of them', '1', '2', '3', '4', '5', '6 or more but not all', 'All'))
    HadAngina = st.selectbox('Ever told you had angina or coronary heart disease?',('Yes', 'No'))
    HadStroke = st.selectbox('Ever told you had a stroke',('Yes', 'No'))
    HadCOPD = st.selectbox('Ever told you had C.O.P.D. (chronic obstructive pulmonary disease), emphysema or chronic bronchitis?',('Yes', 'No'))
    HadKidneyDisease = st.selectbox('Not including kidney stones, bladder infection or incontinence, were you ever told you had kidney disease?',('Yes', 'No'))
    HadArthritis = st.selectbox('Ever told you had some form of arthritis, rheumatoid arthritis, gout, lupus, or fibromyalgia?',('Yes', 'No'))
    HadDiabetes = st.selectbox('Ever told you had diabetes?',('Yes', 'No'))
    DeafOrHardOfHearing = st.selectbox('Are you deaf or do you have serious difficulty hearing?',('Yes', 'No'))
    DifficultyWalking = st.selectbox('Do you have serious difficulty walking or climbing stairs?',('Yes', 'No'))
    SmokerStatus = st.selectbox('Smoker Status', ('Never smoked', 'Former smoker', 'Current smoker'))
    ChestScan = st.selectbox(' Have you ever had a CT or CAT scan of your chest area?',('Yes', 'No'))
    AgeCategory = st.selectbox('Age Category', ('18-24', '25-34', '35-44', '45-54', '55-64', '65-79', 'Age 80 or older'))
    PneumoVaxEver = st.selectbox('Have you ever had a pneumonia shot also known as a pneumococcal vaccine?', ('Yes', 'No'))

    if st.button("Predict"): 
    	prediction(PhysicalHealthDays, GeneralHealth, RemovedTeeth, HadAngina, HadStroke, HadCOPD, HadKidneyDisease, HadArthritis, HadDiabetes, DeafOrHardOfHearing,
    			            DifficultyWalking, SmokerStatus, ChestScan, AgeCategory, PneumoVaxEver)




def pneumonia():
    model = load_model('pneumonia_model.h5')

    st.markdown("<h1 style='text-align: center;'>Pneumonia Prediction App </h1>", unsafe_allow_html=True)

# Centered image
    col1, col2, col3 = st.columns(3)  # Create three columns for layout
    with col2:  # Use the middle column for the centered image
        st.image('media/lung-logo.jpg', width=400, use_column_width=True)

    with st.expander("ℹ️ - About this app", expanded=True):
        st.write("""
        Welcome to the Pneumonia Prediction App! This application utilizes a deep learning model to analyze chest X-ray images 
        and predict the likelihood of pneumonia. Our goal is to provide a quick and preliminary analysis to aid healthcare professionals.
        - *How to use:* Upload a chest X-ray image in JPEG format, and the model will analyze the image and provide a prediction.
        - *Note:* This app should not be used as a sole diagnostic tool. Consult a healthcare professional for an accurate diagnosis.
        """)

    uploaded_file = st.file_uploader("Please upload an X-ray image of you chest", type=["jpeg", "png", "tiff", "webp"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded X-ray image.', use_column_width=True)

        img = img.convert('L').resize((128, 128))
        img = img.resize((128, 128))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img /= 255.

        message_placeholder = st.empty()
        message_placeholder.text("Classifying...")

        message_placeholder.empty()

        prediction = model.predict(img)
        prediction_value = prediction[0][0] * 100

        st.write('Result:')

        if prediction_value < 50:
            st.success(f"Good news! The X-ray image is classified as Normal. The model is {100 - prediction_value:.2f}% confident.")
            st.image('media/lung-success.jpg', width = 400)
        else:
            st.error(f"Bad news. The X-ray image is classified as Pneumonia. The model is {prediction_value:.2f}% confident and we strongly suggest you to consult a healthcare professional.")
            st.image('media/lung-checkup.jpg', width = 400)



def smoking03():
    # Load the XGBoost model
    #with open('ML_Group_xgBoost_Smoking_model.final.model', 'rb') as f:
    #   model = pickle.load(f)

    model = xgb.Booster()
    model.load_model('ML_Group_xgBoost_Smoking_Model_final.model') 


    def transform_data(data):
        # Transform continuous age into age bucket
        data['age_bucket'] = pd.cut(data['age'], bins=range(20, 130, 5), right=False)

        # Define a function to map continuous age to age bucket
        def map_age_to_bucket(age):
            return int((age // 5) * 5)

        # Apply the transformation to the 'age' column
        data['age'] = data['age'].apply(map_age_to_bucket)
        data.drop('age_bucket', axis=1, inplace=True)

        # Calculate 'heightXhemoglobin' without scaling
        data['heightXhemoglobin'] = data['height(cm)'] * data['hemoglobin']

        return data

    # Function to predict using the XGBoost model
    def predict_smoker(data):
        data = np.array(data).reshape(1, -1)
        dmatrix = xgb.DMatrix(data)
        prediction = model.predict(dmatrix)
        return prediction[0]

    st.markdown("<h1 style='text-align: center;'> 🚬Smoking Prediction App 📊</h1>", unsafe_allow_html=True)

    # Centered image
    col1, col2, col3 = st.columns(3)  # Create three columns for layout
    with col2:  # Use the middle column for the centered image
        st.image('media/smoking_.jpg', width=400, use_column_width=True)




    st.write("Tobacco kills up to 8 million people annually worldwide.\
        Smoking causes a range of diseases like cancer, strokes and several lung and heart diseases.\
        Smoking also increases the risk for tuberculosis, certain eye diseases, and problems of the immune system, including rheumatoid arthritis.\
        Since smoking leads to such a vast number of health problems, these problems are easily visible in a person's health data.\
        This app lets you predict if a patient is a smoker based on a few simple health metrics.")
    st.write()

    st.write("*The default inputs are the average values of non-smokers.*")

    col1, col2 = st.columns(2)  # Split the page into two columns for better visualization

    # First column of input form
    with col1:
        age = st.number_input("Age", min_value=20, max_value=100, value=45)
        height = st.number_input("Height (cm)", min_value=120, max_value=250, value=161)
        weight = st.number_input("Weight (kg)", min_value=30, max_value=250, value=63)
        waist = st.number_input("Waist (cm)", min_value=50, max_value=200, value=80)
        eyesight_left = st.number_input("Eyesight (left)", min_value=0, max_value=10, value=1)
        eyesight_right = st.number_input("Eyesight (right)", min_value=0, max_value=10, value=1)
        hearing_left = st.selectbox("Hearing (left)", [1, 2], index=0)
        hearing_right = st.selectbox("Hearing (right)", [1, 2], index=0)
        systolic = st.number_input("Systolic", min_value=0, max_value=250, value=120)
        relaxation = st.number_input("Relaxation", min_value=0, max_value=150, value=75)
        fasting_blood_sugar = st.number_input("Fasting Blood Sugar", min_value=0, max_value=500, value=98)

    with col2:
        cholesterol = st.number_input("Cholesterol", min_value=0, max_value=600, value=197)
        triglyceride = st.number_input("Triglyceride", min_value=0, max_value=600, value=113)
        hdl = st.number_input("HDL", min_value=0, max_value=500, value=59)
        ldl = st.number_input("LDL", min_value=0, max_value=2000, value=116)
        hemoglobin = st.number_input("Hemoglobin", min_value=0, max_value=25, value=14)
        serum_creatinine = st.number_input("Serum Creatinine", min_value=0, max_value=12, value=1)
        ast = st.number_input("AST", min_value=0, max_value=1500, value=25)
        alt = st.number_input("ALT", min_value=0, max_value=3500, value=25)
        gtp = st.number_input("GTP", min_value=0, max_value=1000, value=30)
        dental_caries = st.selectbox("Dental Caries", [0, 1], index=0)  # Dropdown menu

    # User input dictionary
    user_input = {
        'age': age,
        'height(cm)': height,
        'weight(kg)': weight,
        'waist(cm)': waist,
        'eyesight(left)': eyesight_left,
        'eyesight(right)': eyesight_right,
        'hearing(left)': hearing_left,
        'hearing(right)': hearing_right,
        'systolic': systolic,
        'relaxation': relaxation,
        'fasting blood sugar': fasting_blood_sugar,
        'Cholesterol': cholesterol,
        'triglyceride': triglyceride,
        'HDL': hdl,
        'LDL': ldl,
        'hemoglobin': hemoglobin,
        'serum creatinine': serum_creatinine,
        'AST': ast,
        'ALT': alt,
        'Gtp': gtp,
        'dental caries': dental_caries,
    }

    # Transform user input
    transformed_input = transform_data(pd.DataFrame([user_input]))

    # Prediction
    if st.button("Predict"):
        prediction = predict_smoker(transformed_input)
        st.write(f"**The patient is a {'Smoker' if prediction > 0.5 else 'Non-Smoker'}**.")



def healthy_habits():
    
    # Load the model
    model = joblib.load('trained_health_SVClassification_model.sav')
    # Function to take inputs

    def user_input_features():

        gender = st.radio('Gender', ('Female', 'Male'))
        col1, col2 = st.columns(2)

        with col1:
            # One-hot encode gender
            gender_encoded = 1 if gender == 'Male' else 0
            age = st.number_input('Age', value=30)
            height_cm = st.number_input('Height in cm', value=170)
            weight_kg = st.number_input('Weight in kg', value=70)
            body_fat_percent = st.number_input('Body fat percentage', value=25)
            diastolic = st.number_input('Diastolic blood pressure', value=80)
        
        with col2:
            systolic = st.number_input('Systolic blood pressure', value=120)
            gripForce = st.number_input('Grip Force', value=30)
            sit_and_bend_forward_cm = st.number_input('Sit and bend forward in cm', value=15)
            sit_ups_counts = st.number_input('Sit-ups count', value=20)
            broad_jump_cm = st.number_input('Broad jump in cm', value=200)

        # Create input data as a dictionary with correct feature names
        data = {
            'ohe__gender_F': [1 - gender_encoded],  # Complement of gender_Male
            'ohe__gender_M': [gender_encoded],
            'scaler__age': [age],
            'scaler__height_cm': [height_cm],
            'scaler__weight_kg': [weight_kg],
            'scaler__body fat_%': [body_fat_percent],
            'scaler__diastolic': [diastolic],
            'scaler__systolic': [systolic],
            'scaler__gripForce': [gripForce],
            'scaler__sit and bend forward_cm': [sit_and_bend_forward_cm],
            'scaler__sit-ups counts': [sit_ups_counts],
            'scaler__broad jump_cm': [broad_jump_cm],
        }
        features = pd.DataFrame(data)
        return features

    st.markdown("<h1 style='text-align: center;'>🍏 Healthy Habits Classificator 🏋️‍♂️</h1>", unsafe_allow_html=True)
    
    # Centered image
    col1, col2, col3 = st.columns(3)  # Create three columns for layout
    with col2:  # Use the middle column for the centered image
        st.image('media/healthy.jpg', width=400, use_column_width=True)

    st.write("The **Healthy Habits Classification App** is a sophisticated and user-centric platform that employs a comprehensive approach to assessing individuals' health. By meticulously analyzing a wide array of physical performance metrics, such as cardiovascular endurance, muscle strength, flexibility, and more, in conjunction with personal details like age, gender, and medical history, the app delivers a **precise evaluation** of one's health status. This in-depth assessment forms the foundation for the app's intelligent algorithms to generate **highly personalized insights**. These insights go beyond generic health recommendations, tailoring their guidance to each user's unique profile. Whether it's optimizing workout routines, suggesting dietary adjustments, or offering stress management techniques, the app empowers users with a holistic understanding of their well-being, helping them make **informed decisions** to improve their health.")

    st.write("At its core, the **Healthy Habits Classification App** is designed not only to provide valuable health assessments but also to inspire **positive change** in users' lives. By promoting **healthier lifestyle choices**, the app becomes a dedicated partner in enhancing users' overall health and quality of life. It recognizes that true well-being extends beyond just physical health and delves into the realm of **mental and emotional wellness**. Through its user-friendly interface and actionable recommendations, the app motivates individuals to embrace a balanced and holistic approach to their health journey. Whether it's encouraging regular exercise, suggesting **nutritious meal plans**, or offering mindfulness practices, the app instills a sense of **empowerment**, helping users take control of their health and embark on a path towards a **happier, healthier, and more fulfilling life**.")

    
    # Get user input
    input_df = user_input_features()

    # Prediction button
    if st.button("Predict"):
        # Predict
        prediction = model.predict(input_df)

        # Map prediction to corresponding category
        categories = {
            1: 'A: Apex Vitality (The Pinnacle Performers)',
            2: 'B: Robust Wellness (The Steady Strivers)',
            3: 'C: Moderate Health (The Health Conscious)',
            4: 'D: Developing Health (The Path to Progress)'
        }

        predicted_category = categories[prediction[0]]

        # Display prediction
        st.write("## Prediction")
        st.write(predicted_category)
        st.write("Recommended Actions:")
        if prediction == 1:
            st.write("- Maintain regular exercise routines to sustain peak performance.")
            st.write("- Focus on balanced nutrition and hydration.")
            st.write("- Encourage participation in advanced fitness activities to challenge and enhance abilities.")
        elif prediction == 2:
            st.write("- Engage in regular physical activities to maintain and improve current fitness levels.")
            st.write("- Emphasize flexibility and agility exercises to enhance overall mobility.")
            st.write("- Monitor and manage stress levels for holistic health.")
        elif prediction == 3:
            st.write("- Adopt a well-rounded exercise routine that includes both cardiovascular and strength-training exercises.")
            st.write("- Focus on improving specific areas of weakness identified by the model.")
            st.write("- Explore healthier dietary choices and portion control.")
        elif prediction == 4:
            st.write("- Consult with fitness professionals or healthcare providers to create a personalized fitness plan.")
            st.write("- Start with low-impact exercises and gradually increase intensity.")
            st.write("- Embrace a balanced diet, emphasizing fruits, vegetables, and whole grains.")

def feedback():
    st.title("Feedback and Suggestions 😊")
    st.write("We value your feedback and suggestions! Please share your thoughts with us below.")

    # Feedback input
    feedback_text = st.text_area("Your Feedback:")

    # Submit Feedback button
    if st.button("Submit Feedback"):
        # Logic to save feedback
        save_feedback(feedback_text)
        st.success("Thank you for your feedback! We appreciate your input. 🌟")

    st.subheader("Additional Requests 🚀")
    st.write("Which other ML algorithms or features would you like to see available in our platform? Let us know!")

    # Additional Models input
    desired_ml_models = st.text_input("Desired ML Models (comma-separated):")

    # Submit Additional Models button
    if st.button("Submit Additional Models"):
        # Logic to save additional models
        save_additional_models(desired_ml_models)
        st.success("Thank you for your additional model requests! We'll consider your suggestions. 🚀")

    # Line separator
    st.write("---")

    # Contact Information
    st.subheader("Contact Information 📧")
    st.markdown(
        """
        If you need further assistance or have urgent inquiries, feel free to reach out to us:
        
        **Email:** [support@mlmodels.com](mailto:support@example.com) ✉️
        **Phone:** +1 (123) 456-7890 ☎️
        """
    )

def save_feedback(feedback_text):
    # Define the file path for saving feedback
    feedback_file_path = 'feedback.txt'

    # Open the file in append mode and save the feedback
    with open(feedback_file_path, 'a') as file:
        file.write(f"Feedback: {feedback_text}\n")

    # Optionally, you can log the feedback to the console for confirmation
    print(f"Feedback saved: {feedback_text}")

def save_additional_models(desired_ml_models):
    # Define the file path for saving additional models
    additional_models_file_path = 'additional_models.txt'

    # Open the file in append mode and save the additional models
    with open(additional_models_file_path, 'a') as file:
        file.write(f"Additional Models: {desired_ml_models}\n")

    # Optionally, you can log the additional models to the console for confirmation
    print(f"Additional Models saved: {desired_ml_models}")



if selectbox == 'Home':
	home()
elif selectbox == 'Heart Attack Prediction App':
	heart()
elif selectbox == "Pneumonia Prediction App":
	pneumonia()
elif selectbox == "Smoking Detection App":
	smoking03()
elif selectbox == 'Healthy Habits Prediction App':
	healthy_habits()
elif selectbox == "Feedback":
	feedback() 

