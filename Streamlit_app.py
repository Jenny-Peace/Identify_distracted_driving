import streamlit as st
import pandas as pd
import numpy as np
import cv2
import io
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from CNN_oneaction_model import make_average_predictions


#Load your model and check create the class_names list
Model_Path = 'model_MobileNetV2_Final.h5'

class_names = ['adjusting','calling','drinking','grooming','reaching','driving safe','talking','texting']
model = tf.keras.models.load_model(Model_Path)
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


st.set_page_config(
    page_title='IDENTIFY DRIVING DISTRACTION',
    layout='wide',
    initial_sidebar_state='auto',
)
menu = ['With Single Photo','With 1-action Video']
choice = st.sidebar.radio('Identify driving distractions using Deep Learning', menu)


# st.header('Identify driving distractions using Deep Learning')
    

if choice == 'With Single Photo':
    uploaded_photo = st.file_uploader('The requirements of the file', ['png', 'jpeg', 'jpg'])
    if uploaded_photo!=None:
        image_np = np.asarray(bytearray(uploaded_photo.read()), dtype=np.uint8)
        img = cv2.imdecode(image_np, 1)
        st.image(img, channels='BGR')

        # Resize the Image according with your model
        img = cv2.resize(img,(224,224),interpolation = cv2.INTER_AREA)
        # Expand dim to have img_array (1, Height, Width , Channel ) before plugging into the model
        img_array  = np.expand_dims(img, axis=0)
        
        # Prediction
        prediction = model.predict(img_array)
        a = np.argmax(prediction,axis=1)
        #st.write(a[0])
        result = class_names[int(a)]

        # Set threshold   
        threshold = 0.25
        if  prediction.max() < threshold:
            st.write('The driver is doing something')
        else: 
            st.write('The driver is',result)

        # Audio for distracted driving
        if (a != 5) or (prediction.max() < threshold):
            st.audio('Audio/alert audio.mp3')
        # st.write(prediction)     
        

elif choice == 'With 1-action Video':
    # st.write('With 1-action Video')

    st.write('')
    
    # Frame_rate setting
    frame_rate = 90
    # Create box to input video
    uploaded_video = st.file_uploader('The requirements of the file',['mp4'])
    image_height, image_width = 224,224
    predictions_frames_count = frame_rate

    if uploaded_video is not None:
        # Display video
        st.video(uploaded_video)

        # Read Video
        g = io.BytesIO(uploaded_video.read())            # BytesIO Object
        temporary_location = 'test_video/temp.mp4'      # save to temp file
        with open(temporary_location, 'wb') as out:     # Open temporary file as bytes
            out.write(g.read())                         # Read bytes into file
            out.close()                                 # close file
        input_video_file_path = temporary_location

        # Make prediction
        result = make_average_predictions(input_video_file_path, predictions_frames_count)

        df = pd.DataFrame.from_dict(result, orient='index', columns=['Probability'])
        key_result = list(result)
                
        # Set Threshold
        threshold_video = 0.23
        if  df.iloc[0,0] < threshold_video:
            st.write('The driver is doing something')
        else: 
            st.write('The driver is',key_result[0])
            # st.write(df) # Display proba_table
                
        # Audio for distracted driving
        # if key_result[0] != 'driving safe':
            # st.audio('Audio/alert audio.mp3')
        if key_result[0] == class_names[0] and df.iloc[0,0] > threshold_video:
            st.audio('Audio/audio_0.mp3')
        elif key_result[0] == class_names[1] and df.iloc[0,0] > threshold_video:
            st.audio('Audio/audio_1.mp3')
        elif key_result[0] == class_names[2] and df.iloc[0,0] > threshold_video:
            st.audio('Audio/audio_2.mp3')
        elif key_result[0] == class_names[3] and df.iloc[0,0] > threshold_video:
            st.audio('Audio/audio_3.mp3')
        elif key_result[0] == class_names[4] and df.iloc[0,0] > threshold_video:
            st.audio('Audio/audio_4.mp3')
        elif key_result[0] == class_names[5] and df.iloc[0,0] > threshold_video:
            st.audio('Audio/audio_5.mp3')
        elif key_result[0] == class_names[6] and df.iloc[0,0] > threshold_video:
            st.audio('Audio/audio_6.mp3')
        elif key_result[0] == class_names[7] and df.iloc[0,0] > threshold_video:
            st.audio('Audio/audio_6.mp3')
        elif df.iloc[0,0] < threshold_video:
            st.audio('Audio/audio_u.mp3')
    

        



