import cv2
import mediapipe as mp
import os
from datetime import datetime
import json

#objects
cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

#capturing
while True:
    success, image = cap.read()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)
    # checking whether a hand is detected
    if results.multi_hand_landmarks:
        landmarks = []
        for handLms in results.multi_hand_landmarks: # working with each hand
            for id, lm in enumerate(handLms.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append((lm.x,lm.y,lm.z))
                #drawing landmarksS
                if id == 20 :
                    cv2.circle(image, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
            
            
            mpDraw.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)
            
            cv2.imshow("Output", image)
            letter = cv2.waitKey(1)
            if letter == ord('1'):
                break
            else:
                if(letter!=-1):
                    let = chr(letter)
                    timestamp = str(datetime.now().time()).replace(':','-')
                    with open('landmarks/'+let+'/'+timestamp[:timestamp.index('.')]+'.json','w', encoding='utf-8') as f:
                        f.write(json.dumps(landmarks))
                    