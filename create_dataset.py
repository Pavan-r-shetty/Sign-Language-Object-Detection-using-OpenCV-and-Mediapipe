import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands #8
mp_drawing = mp.solutions.drawing_utils #8
mp_drawing_styles = mp.solutions.drawing_styles #8

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3) #8

DATA_DIR = './data'

data = []
labels = []
for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
    # for img_path in os.listdir(os.path.join(DATA_DIR, dir_))[:1]:   #!8
        data_aux = []

        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))         #!
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #convert to rgb when working with mediapipe    #!
        # results = hands.process(img_rgb)   #8
        # if results.multi_hand_landmarks: #8
        #     for hand_landmarks in results.multi_hand_landmarks: #8
        #         mp_drawing.draw_landmarks(                             #8
        #             img_rgb, #image to draw #8
        #             hand_landmarks, # model output #8
        #             mp_hands.HAND_CONNECTIONS, #hand connections #8
        #             mp_drawing_styles.get_default_hand_landmarks_style(), #8
        #             mp_drawing_styles.get_default_hand_connections_style()) #8
 
        # plt.figure()   #!#!8
        # plt.imshow(img_rgb)   #!#!8
        # plt.show()  #!#!8

        results = hands.process(img_rgb)   #!8
        if results.multi_hand_landmarks: #!8
            for hand_landmarks in results.multi_hand_landmarks: #!8
                for i in range(len(hand_landmarks.landmark)): #!8
                    x = hand_landmarks.landmark[i].x   
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            labels.append(dir_)

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
