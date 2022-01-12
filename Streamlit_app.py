import streamlit as st
import tempfile
from pygame import mixer
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from statistics import mode


# LOAD MODEL
Model_Path = 'model_MobileNetV2_Final.h5'

# CREATE CLASS_NAMES LIST
class_names = ['adjusting','calling','drinking','grooming','reaching','driving safe','talking','texting','unknown']
model = tf.keras.models.load_model(Model_Path)

# SET UP TAB NAME
st.set_page_config(
    page_title='IDENTIFY DRIVING DISTRACTION',
    layout='wide',
    initial_sidebar_state='auto',
)

# SET UP MENU
menu = ['Video with UX-robot','Video with UX-Human']
choice = st.sidebar.radio('Identify driving distractions using Deep Learning', menu)


# MAIN 
if choice == 'Video with UX-robot':
    uploaded_video  = st.file_uploader('', ['mp4','mov'])

    FRAME_WINDOW = st.image([])
    count_wrong = 0
    scenario_list = []

    # SET THRESHOLD
    threshold = 0.25
    
    # Read the video
    if uploaded_video is not None: 
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video = cv2.VideoCapture(tfile.name)

        stframe = st.empty()

        count_frame = 0

        while video.isOpened(): 
            ret, frame = video.read()
            # Set 1 predict/5 frame
            if count_frame % 6 == 0: #30 frames/second
                
                if not ret:
                    print("Can't get the frame")
                    break
                        
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize the Image according with the model
                img = cv2.resize(image,(224,224),interpolation = cv2.INTER_AREA)
                # Expand dim to have img_array (1, Height, Width , Channel ) before plugging into the model
                img_array  = np.expand_dims(img, axis=0)

                
                # Make Prediction
                try:
                    # Prediction
                    prediction = model.predict(img_array)
                    a = np.argmax(prediction,axis=1)
                    result = class_names[int(a)]
                
                    # Text for prediction
                    text_pred = 'The driver is ' + result
                except:
                    text_pred = 'The driver is doing something'
                    
                
                # Make reminder
                prediction_round = round(prediction.max(),4)
                if (a != 5) or (prediction_round < threshold):
                    count_wrong += 1
                    if count_wrong == 5:
                        mixer.init() 
                        mixer.music.load('Audio/Robot.wav') 
                        mixer.music.play()
                        count_wrong = 0
                else:
                    count_wrong = 0        
            
            # Display the result
            prediction_round = round(prediction.max(),4)
            if  prediction_round < threshold:
                text_pred = 'The driver is doing something'
                cv2.putText(frame,text_pred+' '+str(prediction_round), (40,70), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,0,255), 2)
            else: 
                if a == 5:
                    cv2.putText(frame, text_pred+' '+str(prediction_round), (40,70), cv2.FONT_HERSHEY_SIMPLEX,  1, (255,0,0), 2) 
                else:
                    cv2.putText(frame, text_pred+' '+str(prediction_round), (40,70), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,0,255), 2) 
            
            # Display predicted frame
            FRAME_WINDOW.image(frame, channels='BGR')
            count_frame += 1



elif choice == 'Video with UX-Human':
    uploaded_video  = st.file_uploader('', ['mp4','mov'])

    FRAME_WINDOW = st.image([])
    count_wrong = 0
    scenario_list = []
    same_scenario = [5]

    # SET THRESHOLD
    threshold = 0.35

    # Read the video
    if uploaded_video is not None: 
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video = cv2.VideoCapture(tfile.name)

        stframe = st.empty()

        count_frame = 0 

        while video.isOpened(): 
            ret, frame = video.read()
            # SET 1 predict/5 frame
            if count_frame % 6 == 0:
                if not ret:
                    print("Can't get the frame")
                    break
                        
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize the Image according with the model
                img = cv2.resize(image,(224,224),interpolation = cv2.INTER_AREA)
                # Expand dim to have img_array (1, Height, Width , Channel ) before plugging into the model
                img_array  = np.expand_dims(img, axis=0)

                # MAKE PREDITION
                try:
                    # Prediction
                    prediction = model.predict(img_array)
                    a = np.argmax(prediction,axis=1)                                                                                                                                                                                                                                                                                                                                                                                 
                    result = class_names[int(a)]
                
                    # TEXT FOR PREDICTION
                    text_pred = 'The driver is ' + result
                except:
                    text_pred = 'The driver is doing something'
                           

                # MAKE AUDIO Reminder for each case
                # Detect case
                if (a != 5) and (prediction.max() > threshold):
                    class_pred = np.argmax(prediction,axis=1)
                    
                    scenario_list.append(int(class_pred))
                    
                    if len(scenario_list) == 12: 
                        final_class_pred = mode(scenario_list)
                        final_result = class_names[int(final_class_pred)]
                        # AUDIO FOR EACH CASE
                        count_mode_class = scenario_list.count(final_class_pred)
                        if count_mode_class <= round(len(scenario_list)*(2/3)):
                            final_class_pred = 8

                        # Choose suitable audio
                        same_scenario.append(int(final_class_pred))
                        mixer.init()
                        if same_scenario[-2] != same_scenario[-1]:
                            if count_mode_class <= round(len(scenario_list)*(2/3)):
                                mixer.music.load('Audio/audio_u.mp3')
                            else:
                                if final_result == class_names[0]:
                                    mixer.music.load('Audio/audio_0.mp3')
                                elif final_result == class_names[1]:
                                    mixer.music.load('Audio/audio_1.mp3')
                                elif final_result == class_names[2]:
                                    mixer.music.load('Audio/audio_2.mp3')
                                elif final_result == class_names[3]:
                                    mixer.music.load('Audio/audio_3.mp3')
                                elif final_result == class_names[4]:
                                    mixer.music.load('Audio/audio_4.mp3')
                                elif final_result == class_names[5]:
                                    mixer.music.load()
                                elif final_result == class_names[6]:
                                    mixer.music.load('Audio/audio_6.mp3')
                                elif final_result == class_names[7]:
                                    mixer.music.load('Audio/audio_7.mp3')
                                else:
                                    mixer.music.load('Audio/audio_u.mp3')
                            
                        else: # Audio for stronger reminder if the case is similar with the previous case
                            class_same = class_names[int(final_class_pred)]
                            # AUDIO for each case
                            if class_same == class_names[0]:
                                mixer.music.load('Audio/audio_20.mp3')
                            elif class_same == class_names[1]:
                                mixer.music.load('Audio/audio_21.mp3')
                            elif class_same == class_names[2]:
                                mixer.music.load('Audio/audio_22.mp3')
                            elif class_same == class_names[3]:
                                mixer.music.load('Audio/audio_23.mp3')
                            elif class_same == class_names[4]:
                                mixer.music.load('Audio/audio_24.mp3')
                            elif class_same == class_names[5]:
                                mixer.music.load()
                            elif class_same == class_names[6]:
                                mixer.music.load('Audio/audio_26.mp3')
                            elif class_same == class_names[7]:
                                mixer.music.load('Audio/audio_27.mp3')
                            else:
                                mixer.music.load('Audio/audio_2u.mp3')
                        
                        mixer.music.play()
                       
                        scenario_list = []
                else:
                    scenario_list = []
                
            # Display the result
            prediction_round = round(prediction.max(),4)
            if  prediction_round < threshold:
                text_pred = 'The driver is doing something'
                cv2.putText(frame,text_pred+' '+str(prediction_round), (40,70), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,0,255), 2)
            else: 
                if a == 5:
                    cv2.putText(frame, text_pred+' '+str(prediction_round), (40,70), cv2.FONT_HERSHEY_SIMPLEX,  1, (255,0,0), 2) 
                else:
                    cv2.putText(frame, text_pred+' '+str(prediction_round), (40,70), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,0,255), 2) 
            
            # Display predicted frame
            FRAME_WINDOW.image(frame, channels='BGR')
            count_frame += 1