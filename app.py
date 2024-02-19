# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import imutils
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


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    lis_classifier = LISClassifier()
    lis_classifier_moving = LISClassifierMoving()

    # Read labels ###########################################################
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

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 6
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################

    while True:

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        
        # Camera capture #####################################################
        ret, image = cap.read()
        image = imutils.resize(image, width=1190)
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
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
                '''
                if hand_sign_id==None:
                    point_history.append(landmark_list)
                else:
                    point_history.append([[0.1,0.1,0.1] for i in range (21)])
                '''

                point_history.append(landmark_list)

                # Finger gesture classification
                finger_gesture_id = 1
                percentage_moving = 0
           
                sequence = list(point_history)[-7:]
                if len(sequence) == 6:
                    point_history_list = create_distance_vector(sequence,True)
                    finger_gesture_id, percentage_moving = lis_classifier_moving(
                        point_history_list)
    
                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()
                
                sign = get_template(hand_sign_id) #get the file with the guessed sign
                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, hand_landmarks)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    lis_classifier_labels[hand_sign_id],
                    lis_classifier_moving_labels[finger_gesture_id], 
                    percentage,
                    percentage_moving,
                    sign
                )
        
        # Screen reflection #############################################################
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


def draw_landmarks(image, landmark_point):
    mpDraw = mp.solutions.drawing_utils
    mpHands = mp.solutions.hands
    mpDraw.draw_landmarks(image, landmark_point, mpHands.HAND_CONNECTIONS)
            
    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 1)

    return image

#Function that returns right image for the sign
def get_template(indx):
    signs = os.listdir('sign_photos')
    sign = signs[indx]
    return "sign_photos/"+sign

 #DIsolay all the info on capture
def draw_info_text(image, brect, handedness, hand_sign_text, moving_text, percentage, percentage_moving,sign):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)
    info_text = handedness.classification[0].label[0:]

    #Determine if similarity threshold is enough
    h_text = "None" if percentage < 0.75 else hand_sign_text
    m_text = "None" if percentage_moving < 0.99 else moving_text

    #If not, remind how letter should be displayed
    try_text = ""
    try_m_text = ""

    if h_text == "None":
        try_text = "Do you mean:" + hand_sign_text +"?"
    if m_text == "None":
        try_m_text = "Do you mean:" + moving_text +"?"

    #Set text to display
    if hand_sign_text != "":
        info_text = info_text + ':' + h_text
    if moving_text != "":
        info_text = info_text + ':' + m_text


    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    cv.putText(image,try_text, (10, 70), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0,0,0), 4, cv.LINE_AA)
    cv.putText(image,try_text, (10, 70), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)
    
    #Display image of guessed sign
    sign = cv.imread(sign)

    if h_text=="None":
        image[100:100+sign.shape[0], 10:10+sign.shape[1]] = sign
    return image

if __name__ == '__main__':
    main()
