# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import imutils
from random import choice
import os
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from cvfpscalc import CvFpsCalc
from lis_classifier import LISClassifier
from lis_classifier_moving import LISClassifierMoving

from data_parsing import create_distance_vector

def main():
    cap_device = 0
    cap_width = 960
    cap_height = 540

    use_static_image_mode = False
    min_detection_confidence = 0.7
    min_tracking_confidence = 0.5

    use_brect = True

    # Camera preparation
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load 
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    lis_classifier = LISClassifier()
    lis_classifier_moving = LISClassifierMoving()

    # Read labels 
    with open(
            'lis_classifier_label.csv',
            encoding='utf-8-sig') as f:
        lis_classifier_labels = csv.reader(f)
        lis_classifier_labels = [
            row[0] for row in lis_classifier_labels
        ]
    with open(
            'lis_classifier_moving_label.csv',
            encoding='utf-8-sig') as f:
        lis_classifier_moving_labels = csv.reader(f)
        lis_classifier_moving_labels = [
            row[0] for row in lis_classifier_moving_labels
        ]

    #Test words
    test_words = ["spell your name", "spell grazie", "spell casa", "spell rana", "spell corsa"]
    word_to_spell = 0

    # Coordinate history ######
    history_length = 6
    point_history = deque(maxlen=history_length)

    #Hand gesture history
    hand_history_length = 10
    hand_gesture_history = deque(maxlen=hand_history_length)
    
    word = ["None"]
    last_let = ""

    mode = 0
    #  #############

    while True:

        # Process Key
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        if key == 49:
            mode = 1
        if key == 48:
            mode = 0 
        
        # Camera capture 
        ret, image = cap.read()
        image = imutils.resize(image, width=1000)
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation ##
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  #########
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)

                # Landmark calculation
                landmark_list = calc_landmark_list(hand_landmarks)
                processed_landmark_list = create_distance_vector(landmark_list,False)

                # Hand sign classification
                hand_sign_id, percentage = lis_classifier(processed_landmark_list)
                
                point_history.append(landmark_list)
                
                # Finger gesture classification
                finger_gesture_id = 1
                percentage_moving = 0
           
                sequence = list(point_history)[-7:]

                if len(sequence) == 6:
                    point_history_list = create_distance_vector(sequence,True)
                    finger_gesture_id, percentage_moving = lis_classifier_moving(
                        point_history_list)

                #Get the file with the guessed sign
                sign = get_template(hand_sign_id, 'sign_photos') 
                if finger_gesture_id == 0:
                    moving_sign = get_template(0, 'moving_sign_photos')
                elif finger_gesture_id == 3:
                    moving_sign = get_template(1, 'moving_sign_photos')
                else:
                    moving_sign = ""

                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, hand_landmarks)
                debug_image, letter = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    lis_classifier_labels[hand_sign_id],
                    lis_classifier_moving_labels[finger_gesture_id], 
                    percentage,
                    percentage_moving,
                    sign, 
                    moving_sign,
                    mode,
                    word_to_spell,
                    test_words
                )
                
                if len(word)==0:
                    word.append("None")

                if mode == 1:
                    #Construction of the word
                    hand_gesture_history.append(letter)
                    h_g = list(hand_gesture_history)
                    if len(h_g) > 0 and h_g.count(h_g[0]) == hand_history_length:
                        if letter == "si":
                            word = ["None"]
                            if mode == 1:
                                word_to_spell = choice([i for i in range(0,len(test_words)) if i != word_to_spell])
                            hand_gesture_history.clear()
                        elif last_let!= letter:
                            if letter == "no":
                                word = word[:-2]
                            else:
                                word.append(letter)
                            last_let=letter

                    debug_image = draw_word(word,debug_image)
        
        # Screen reflection 
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()

#calculate the bounding
def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

#calculate the list
def calc_landmark_list(landmarks):

    landmarks_ls = []
    for id, lm in enumerate(landmarks.landmark):
        landmarks_ls.append([lm.x,lm.y,lm.z])
    return landmarks_ls

#draws the hand landmarks
def draw_landmarks(image, landmark_point):
    mpDraw = mp.solutions.drawing_utils
    mpHands = mp.solutions.hands
    mpDraw.draw_landmarks(image, landmark_point, mpHands.HAND_CONNECTIONS)
            
    return image

#draws the rectangle around the hand
def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image

#Function that returns right image for the sign
def get_template(indx, path):
    signs = os.listdir(path)
    sign = signs[indx]
    return path+"/"+sign

#writes the word the user is spelling
def draw_word(word,image):
    w = ""
    for let in word:
        if let!="None":
            w+=let

    cv.putText(image,w, (400, 700), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0,0,0), 4, cv.LINE_AA)
    cv.putText(image,w, (400, 700), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)
    return image


#Display all the info on capture
def draw_info_text(image, brect, handedness, hand_sign_text, moving_text,
                    percentage, percentage_moving,sign, moving_sign,
                    mode, word_to_spell, test_words):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)
    info_text = handedness.classification[0].label[0:]

    #Determine if similarity threshold is enough
    h_text = "None" if percentage < 0.85 else hand_sign_text
    m_text = "None" if percentage_moving < 0.99 else moving_text

    #If not, remind how letter should be displayed
    try_text = ""
    try_m_text = ""

    if h_text == "None":
        try_text = "Do you mean:" + hand_sign_text +"?"
    if m_text == "None" and h_text == "None":
        try_m_text = "Do you mean:" + moving_text +"?"

    letter = ""

    #Set text to display
    if h_text != "":
        if m_text != "" and h_text=="None":# and (hand_sign_text=="g" or hand_sign_text=="i" or hand_sign_text=="None" or hand_sign_text=="o"):
            info_text = info_text + ':' + m_text
            letter = m_text
        else:
            info_text = info_text + ':' + h_text
            letter = h_text
    
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
                cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    
    #If mode is free speech 
    if mode == 0:
        cv.putText(image,try_text, (10, 70), cv.FONT_HERSHEY_SIMPLEX,
                1.0, (0,0,0), 4, cv.LINE_AA)
        cv.putText(image,try_text, (10, 70), cv.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv.LINE_AA)
        cv.putText(image,try_m_text, (470, 70), cv.FONT_HERSHEY_SIMPLEX,
                1.0, (0,0,0), 4, cv.LINE_AA)
        cv.putText(image,try_m_text, (470, 70), cv.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv.LINE_AA)
        
        
        #Display image of guessed sign
        sign = cv.imread(sign)
        if moving_sign!="":
            moving_sign = cv.imread(moving_sign)

        if h_text=="None":
            image[0:0+sign.shape[0], 300:300+sign.shape[1]] = sign
        if m_text=="None" and moving_sign!="" and try_m_text!="":
            image[0:0+moving_sign.shape[0], 740:740+moving_sign.shape[1]] = moving_sign

    
    #If mode is spell practice
    else:
        cv.putText(image,test_words[word_to_spell], (10, 70), cv.FONT_HERSHEY_SIMPLEX,
                1.0, (0,0,0), 4, cv.LINE_AA)
        cv.putText(image,test_words[word_to_spell], (10, 70), cv.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2, cv.LINE_AA)
    return image,letter

if __name__ == '__main__':
    main()
