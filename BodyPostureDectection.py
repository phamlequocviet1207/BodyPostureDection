import cv2 as cv
import mediapipe as mp
import math as m
import time

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

# Initialize frame counters.
good_frames = 0
bad_frames  = 0
 
# Font type.
font = cv2.FONT_HERSHEY_SIMPLEX
 
# Colors.
blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)
 
# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


