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


#Load your model and check create the class_names list
Model_Path = 'model_MobileNetV2_Final.h5'

class_names = ['adjusting','calling','drinking','grooming','reaching','driving safe','talking','texting']
model = tf.keras.models.load_model(Model_Path)


st.set_page_config(
    page_title='IDENTIFY DRIVING DISTRACTION',
    layout='wide',
    initial_sidebar_state='auto',
)
# menu = ['Single Photo','Video with UX-1','Video with UX-2']
menu = ['Video with UX-1','Video with UX-2']
choice = st.sidebar.radio('Identify driving distractions using Deep Learning', menu)


# st.header('Identify driving distractions using Deep Learning')
    

if choice == 'Single Photo':
    uploaded_photo = st.file_uploader('The requirements of the file', ['png', 'jpeg', 'jpg'])
    if uploaded_photo!=None:
        image_np = np.asarray(bytearray(uploaded_photo.read()), dtype=np.uint8)
        img = cv2.imdecode(image_np, 1)
        # st.image(img, channels='BGR')

        # Resize the Image according with your model
        img = cv2.resize(img,(224,224),interpolation = cv2.INTER_AREA)
        # Expand dim to have img_array (1, Height, Width , Channel ) before plugging into the model
        img_array  = np.expand_dims(img, axis=0)
        
        # Prediction
        prediction = model.predict(img_array)
        a = np.argmax(prediction,axis=1)                                                                                                                                                                                                                                                                                                                                                                                                                                                      
        #st.write(a[0])
        result = class_names[int(a)]

        # TEXT FOR PREDICTION
        text_unknown = 'The driver is doing something'
        text_pred = 'The driver is ' + result

        # SET THRESHOLD AND DISPLAY THE RESULT
        img = cv2.imdecode(image_np, 1) 

        threshold = 0.25
        prediction_round = round(prediction.max(),3)
        if  prediction_round < threshold:
            cv2.putText(img, text_unknown, (40,35), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,0,255), 2)
            
        else: 
            if a == 5:
                cv2.putText(img, text_pred+' '+str(prediction_round), (40,70), cv2.FONT_HERSHEY_SIMPLEX,  1, (255,0,0), 2) #,prediction_round)
            else:
                cv2.putText(img, text_pred+' '+str(prediction_round), (40,70), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,0,255), 2) #,prediction_round)
            # st.write(text_pred,prediction_round)     
        st.image(img, channels='BGR')
        

        # # Audio for distracted driving
        # if (a != 5) or (prediction_round < threshold):
        #     # st.audio('Audio/alert audio.mp3')
        #     mixer.init() # initiate the mixer instance
        #     mixer.music.load('Audio/mixkit-game-show-wrong-answer-buzz-950.wav') # loads the music, can be also mp3 file.
        #     mixer.music.play()
        # # st.write(prediction)     
        
    

elif choice == 'Video with UX-1':
    uploaded_video  = st.file_uploader('', ['mp4','mov'])

    FRAME_WINDOW = st.image([])
    count_wrong = 0
    scenario_list = []
    # SET THRESHOLD
    threshold = 0.25

    
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
                        
                image                   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                

                # Resize the Image according with the model
                img = cv2.resize(image,(224,224),interpolation = cv2.INTER_AREA)
                # Expand dim to have img_array (1, Height, Width , Channel ) before plugging into the model
                img_array  = np.expand_dims(img, axis=0)

                
                # MAKE PREDITION
                try:
                    # Prediction
                    prediction = model.predict(img_array)
                    a = np.argmax(prediction,axis=1)                                                                                                                                                                                                                                                                                                                                                                                                                                                      
                    #st.write(a[0])
                    result = class_names[int(a)]
                    # st.write('The driver is',result)
                
                    # TEXT FOR PREDICTION
                    # text_unknown = 'The driver is doing something'
                    text_pred = 'The driver is ' + result
                except:
                    text_pred = 'The driver is doing something'
                    
                
                # MAKE REMINDER
                prediction_round = round(prediction.max(),3)
                if (a != 5) or (prediction_round < threshold):
                    count_wrong += 1
                    if count_wrong == 5:
                        mixer.init() # initiate the mixer instance
                        mixer.music.load('Audio/mixkit-game-show-wrong-answer-buzz-950.wav') # loads the music, can be also mp3 file.
                        mixer.music.play()
                        count_wrong = 0
                else:
                    count_wrong = 0        

                
            
            # DISPLAY THE RESULT
            prediction_round = round(prediction.max(),3)
            if  prediction_round < threshold:
                text_pred = 'The driver is doing something'
                cv2.putText(frame,text_pred+' '+str(prediction_round), (40,70), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,0,255), 2)
                # st.write(text_pred,prediction_round)
            else: 
                if a == 5:
                    cv2.putText(frame, text_pred+' '+str(prediction_round), (40,70), cv2.FONT_HERSHEY_SIMPLEX,  1, (255,0,0), 2) #,prediction_round)
                else:
                    cv2.putText(frame, text_pred+' '+str(prediction_round), (40,70), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,0,255), 2) #,prediction_round)
                # st.write(text_pred,prediction_round)
            
            # Display predicted frame
            FRAME_WINDOW.image(frame, channels='BGR')
            count_frame += 1
   


elif choice == 'Video with UX-2':
    uploaded_video  = st.file_uploader('', ['mp4','mov'])

    FRAME_WINDOW = st.image([])
    count_wrong = 0
    scenario_list = []

    # SET THRESHOLD
    threshold = 0.35

    #MAIN
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
                    #st.write(a[0])
                    result = class_names[int(a)]
                    # st.write('The driver is',result)
                
                    # TEXT FOR PREDICTION
                    text_pred = 'The driver is ' + result
                except:
                    text_pred = 'The driver is doing something'
                           

                # MAKE AUDIO Reminder for each case
                if (a != 5) and (prediction.max() > threshold):
                    class_pred = np.argmax(prediction,axis=1)
                    # st.write(class_pred)
                    
                    scenario_list.append(int(class_pred))
                    # st.write(scenario_list)
                    if len(scenario_list) == 12: #Test in 25 (2,5 times of fps)
                        # st.write(scenario_list)
                        final_class_pred = mode(scenario_list)
                        # st.write(final_class_pred)
                        final_result = class_names[int(final_class_pred)]
                        # st.write(final_result)
                        # AUDIO FOR EACH CASE
                        count_mode_class = scenario_list.count(final_class_pred)
                        # st.write(count_mode_class)
                        # initiate the mixer instance
                        mixer.init()
                        if count_mode_class <= round(len(scenario_list)*(2/3)):
                            # st.write('unknown')
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
                            elif final_result == class_names[6]:
                                mixer.music.load('Audio/audio_6.mp3')
                            elif final_result == class_names[7]:
                                mixer.music.load('Audio/audio_7.mp3')
                        mixer.music.play()
                        scenario_list = []
                else:
                    scenario_list = []
            
            # DISPLAY THE RESULT
            prediction_round = round(prediction.max(),3)
            if  prediction_round < threshold:
                text_pred = 'The driver is doing something'
                cv2.putText(frame,text_pred+' '+str(prediction_round), (40,70), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,0,255), 2)
                # st.write(text_pred,prediction_round)
            else: 
                if a == 5:
                    cv2.putText(frame, text_pred+' '+str(prediction_round), (40,70), cv2.FONT_HERSHEY_SIMPLEX,  1, (255,0,0), 2) #,prediction_round)
                else:
                    cv2.putText(frame, text_pred+' '+str(prediction_round), (40,70), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,0,255), 2) #,prediction_round)
                # st.write(text_pred,prediction_round)
            
            # Display predicted frame
            FRAME_WINDOW.image(frame, channels='BGR')
            count_frame += 1