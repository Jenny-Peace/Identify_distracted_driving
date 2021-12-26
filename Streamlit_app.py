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
from CNN_model import make_average_predictions


#Load your model and check create the class_names list
Model_Path = 'model_MobileNetV2_9418.h5'

class_names = ['adjusting','calling','drinking','grooming','reaching','safe','talking','texting']
model = tf.keras.models.load_model(Model_Path)
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


st.set_page_config(
    page_title="DISTRACTED DRIVERS RECOGNITION",
    layout='wide',
    initial_sidebar_state='auto',
)
menu = ['Predict Photo','Predict 1-action Video']
choice = st.sidebar.radio('Menu:', menu)


st.header("Drive safety with AI Companion!")
# if choice=='Home':
#     st.title("What content here??? ")

#     st.balloons()
    

if choice == 'Predict Photo':
    st.title('Upload Your Photo')
    uploaded_photo = st.file_uploader('Please take a look at the requirements of the file', ['png', 'jpeg', 'jpg'])
    if uploaded_photo!=None:
        image_np = np.asarray(bytearray(uploaded_photo.read()), dtype=np.uint8)
        img = cv2.imdecode(image_np, 1)
        st.image(img, channels='BGR')

        #st.write(uploaded_photo.size)
        #st.write(uploaded_photo.type)

        #Resize the Image according with your model
        img = cv2.resize(img,(224,224),interpolation = cv2.INTER_AREA)
        #Expand dim to make sure your img_array is (1, Height, Width , Channel ) before plugging into the model
        img_array  = np.expand_dims(img, axis=0)
        
        #JENNY code
        prediction = model.predict(img_array)
        a = np.argmax(prediction,axis=1)
        #st.write(a[0])
        result = class_names[int(a)]
        if a != 5:
            st.audio('Angry Birds Sound SMS-nhacchuong123.com.mp3')    
        st.write('The driver is',result)
        st.write(prediction)
        

elif choice == 'Predict 1-action Video':
    st.title('Predict 1-action Video')

    st.write('')
    # Create slider bar
    frame_rate = st.sidebar.slider(
        "Number of Frames",
        min_value=1,
        max_value=100,
        value=100,
        step=5,
        help="Number of frames that CNN uses to average",
    )

    # Create box to input video
    st.title('Upload your videos (.mp4)')
    uploaded_video = st.file_uploader(' ',['mp4'])
    image_height, image_width = 224,224
    predictions_frames_count = frame_rate

    # Adding columns to format video and prediction table
    col1,col2 = st.columns(2)

    if uploaded_video is not None:
        with col1:
            st.video(uploaded_video)

        with col2:
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
                st.write('The driver is',key_result[0])
                st.write(df)
    
    
    
    
    # # Extract frame


    # # Decode image
    # image_np = np.asarray(bytearray(file_uploaded.read()), dtype=np.uint8)
    # img = cv2.imdecode(image_np, 1)
    # st.image(img, channels='BGR')

    # #Resize the Image according with the model
    # img = cv2.resize(img,(224,224),interpolation = cv2.INTER_AREA)
    # #Expand dim to make sure your img_array is (1, Height, Width , Channel ) before plugging into the model
    # img_array  = np.expand_dims(img, axis=0)
   
    # #JENNY code for prediction
    # prediction = model.predict(img_array)
    # a = np.argmax(prediction,axis=1)
    # st.write(a)
    # result = class_names[int(a)]
    # st.write('The driver is',result)
        



