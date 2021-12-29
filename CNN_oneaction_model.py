import os
import cv2
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
import matplotlib.pyplot as plt

# Define subcategories
classes_list = ['adjusting','calling','drinking','grooming','reaching','driving safe','talking','texting']

# Define model output nodes
model_output_size = len(classes_list)

# Define input image height and width for mode
image_height, image_width = 224, 224


# Load the CNN model
Model_Path = 'model_MobileNetV2_Final.h5'

model = tf.keras.models.load_model(Model_Path)

# Function for prediction
def make_average_predictions(video_file_path, predictions_frames_count):
    probability = []
    idx_sorted =[]
    # Initializing the Numpy array which will store Prediction Probabilities
    predicted_labels_probabilities_np = np.zeros((predictions_frames_count, model_output_size), dtype = np.float)

    # Reading the Video File using the VideoCapture Object
    video_reader = cv2.VideoCapture(video_file_path)

    # Getting The Total Frames present in the video 
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculating The Number of Frames to skip Before reading a frame
    skip_frames_window = video_frames_count // predictions_frames_count

    for frame_counter in range(predictions_frames_count): 

        # Setting Frame Position
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

        # Reading The Frame
        _ , frame = video_reader.read() 

        # Resize the Frame to fixed Dimensions
        resized_frame = cv2.resize(frame, (image_height, image_width))
        
        # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
        normalized_frame = resized_frame #/ 255

        # Passing the Image Normalized Frame to the model and receiving Predicted Probabilities.
        predicted_labels_probabilities = model.predict(np.expand_dims(normalized_frame, axis = 0))[0]

        # Appending predicted label probabilities to the deque object
        predicted_labels_probabilities_np[frame_counter] = predicted_labels_probabilities

    # Calculating Average of Predicted Labels Probabilities Column Wise 
    predicted_labels_probabilities_averaged = predicted_labels_probabilities_np.mean(axis = 0)

    # Sorting the Averaged Predicted Labels Probabilities
    predicted_labels_probabilities_averaged_sorted_indexes = np.argsort(predicted_labels_probabilities_averaged)[::-1]

    probability = predicted_labels_probabilities_averaged.copy().tolist()
    idx_sorted = predicted_labels_probabilities_averaged_sorted_indexes.copy().tolist()

    result0 = classes_list[idx_sorted[0]]
    result1 = classes_list[idx_sorted[1]]
    result2 = classes_list[idx_sorted[2]]
    result3 = classes_list[idx_sorted[3]]
    result4 = classes_list[idx_sorted[4]]
    result5 = classes_list[idx_sorted[5]]
    result6 = classes_list[idx_sorted[6]]
    result7 = classes_list[idx_sorted[7]]
    
    # create a dictionary result
    dict_result = {result0:probability[idx_sorted[0]],
                   result1:probability[idx_sorted[1]],
                   result2:probability[idx_sorted[2]],
                   result3:probability[idx_sorted[3]],
                   result4:probability[idx_sorted[4]],
                   result5:probability[idx_sorted[5]],
                   result6:probability[idx_sorted[6]],
                   result7:probability[idx_sorted[7]]
                   }

    video_reader.release()

    return dict_result
    

