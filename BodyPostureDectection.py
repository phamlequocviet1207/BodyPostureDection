import cv2 as cv
import mediapipe as mp
import math as m
import numpy as np

def findLength(x,y):
    l = m.sqrt(())
    return 

def findDistance(x1,y1,x2,y2):
    dist = m.sqrt((x1-x2)**2 + ((y1-y2)**2))
    return dist

def findAngle(x1,y1,x2,y2):
    x3=x1
    y3=0
    #Calculate the vector_12 and vector_13 by using tuple
    vector_12 = (x2-x1, y2-y1)
    vector_13 = (x3-x1, y3-y1)

    #Calculate the length of vector_12 and vector_13
    l12 = m.sqrt((vector_12[0])**2 + (vector_12[1])**2)
    l13 = m.sqrt((vector_13[0])**2 + (vector_13[1])**2)

    numerator = vector_12[0] * vector_13[0] + vector_12[1] * vector_13[1]
    denominator = m.sqrt( (vector_12[0]**2) + (vector_12[1]**2) + (vector_13[0]**2) + (vector_13[1]**2))

    theta = m.acos(numerator / denominator)

    degree = int(180/m.pi)*theta

    return degree

def sendWarning(x):
    pass
 
# Initialize mediapipe pose class.
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv.VideoCapture(0)
screen_size = (120,720)
# with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
while cap.isOpened():
    ret, frame = cap.read()
    cv.imshow('Mediapipe Feed', frame)
    if (cv.waitKey(10) & 0xFF == ord('q')):
        break


    

cap.release
cv.destroyAllWindows





