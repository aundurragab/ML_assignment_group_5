import streamlit as st 
import numpy as np 
import matplotlib.pyplot as plt
import pickle
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

side_bar = st.sidebar

side_bar.header("Please select one of the following apps.")
selectbox = side_bar.selectbox('Choose an app:',('Heart Attack Predicton App', 'Pneumonia Prediction App'))

if selectbox == 'Heart Attack Predicton App':

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
        	st.image('heart-success.webp', width=650)
        else:
        	st.error(f'Important: The prediction indicates a potential risk of a heart attack with a probability of {(probability_of_heart_attack):.2f}%. It is crucial to consult a healthcare professional.')
        	st.image('heart-attack.jpg', width=650)

        return probability_of_heart_attack

    st.title('Heart Attack App')

    st.image('heart-logo.jpg', width = 700)

    # Application Description
    description = """
    **Welcome to the Heart Attack Prediction App!**\
    This application is designed to provide you with a quick and simple assessment of your potential risk of experiencing a heart attack based on your current health status and lifestyle choices. Our goal is to empower you with knowledge so that you can take proactive steps towards maintaining your heart health.

    **How does it work?**
    1. **Input Your Information:** You will be asked to provide specific details about your health and lifestyle. These include your physical health days, general health condition, smoking status, and more. All of these factors play a significant role in determining your heart health.
    2. **Get Your Risk Assessment:** Once you submit your information, our advanced prediction model, which has been trained on a wide range of health data, will analyze your inputs and calculate your potential risk of having a heart attack.

    **Please Note:**
    - This tool is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
    - The prediction is based on general trends and patterns in health data, and individual variations may exist.

    **Take Charge of Your Heart Health Today!**
    By understanding your risk and taking steps to live a healthier lifestyle, you can reduce your chances of heart disease and lead a longer, healthier life.
    """
    st.markdown(description)

    # Streamlit app layout
    st.subheader('Please enter your details:')

    # Collecting user input

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

    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
    	prediction(PhysicalHealthDays, GeneralHealth, RemovedTeeth, HadAngina, HadStroke, HadCOPD, HadKidneyDisease, HadArthritis, HadDiabetes, DeafOrHardOfHearing,
    			            DifficultyWalking, SmokerStatus, ChestScan, AgeCategory, PneumoVaxEver)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

if selectbox == 'Pneumonia Prediction App':

    # Load model
    model = load_model('pneumonia_model.h5')

    # Streamlit application
    st.title('Pneumonia Prediction App')

    st.image('lung-logo.jpg', width = 700)

    with st.expander("ℹ️ - About this app", expanded=True):
    	st.write("""
        Welcome to the Pneumonia Prediction App! This application utilizes a deep learning model to analyze chest X-ray images 
        and predict the likelihood of pneumonia. Our goal is to provide a quick and preliminary analysis to aid healthcare professionals.
        - **How to use:** Upload a chest X-ray image in JPEG format, and the model will analyze the image and provide a prediction.
        - **Note:** This app should not be used as a sole diagnostic tool. Consult a healthcare professional for an accurate diagnosis.
        """)

    #Upload image through streamlit
    uploaded_file = st.file_uploader("Please upload an X-ray image of you chest", type="jpeg")

    #Functionality of the app
    if uploaded_file is not None:
    	img = Image.open(uploaded_file)
    	st.image(img, caption='Uploaded X-ray image.', use_column_width=True)

    	# Preprocess the image to match model input
    	img = img.resize((128, 128))
    	img = image.img_to_array(img)
    	img = np.expand_dims(img, axis=0)
    	img /= 255.

    	# Create a placeholder for the "Classifying..." message
    	message_placeholder = st.empty()
    	message_placeholder.text("Classifying...")
    	# Preprocess and classify the image
    	# Clear the "Classifying..." message
    	message_placeholder.empty()

    	# Make prediction
    	prediction = model.predict(img)
    	prediction_value = prediction[0][0] * 100  # Convert to percentage
    	st.write('Result:')

    	if prediction_value < 50:
    		st.success(f"Good news! The X-ray image is classified as Normal. The model is {100 - prediction_value:.2f}% confident.")
    		st.image('lung-success.jpg', width = 700)
    	else:
    		st.error(f"Bad news. The X-ray image is classified as Pneumonia. The model is {prediction_value:.2f}% confident and we strongly suggest you to consult a healthcare professional.")
    		st.image('lung-checkup.jpg', width = 700)













































































































































