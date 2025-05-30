import cv2 as cv
import mediapipe as mp
import math as m
import numpy as np

def findLength(x,y):
    l = m.sqrt(())
    return 

def findDistance(x1,y1,x2,y2):
    # print("c")
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
    # denominator = m.sqrt( (vector_12[0]**2) + (vector_12[1]**2) + (vector_13[0]**2) + (vector_13[1]**2))
    denominator = l12 * l13

    print(numerator,denominator)

    theta = m.acos(numerator / denominator)

    degree = int(180/m.pi)*theta

    return degree
 
# Initialize mediapipe pose class.
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
(w,h) = (640,480)
red = (50, 50, 255)
green = (127, 255, 0)
yellow = (0, 255, 255)
pink = (255, 0, 255)
font = cv.FONT_HERSHEY_SIMPLEX

cap = cv.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        frame = cv.resize(frame, (w,h), interpolation=cv.INTER_AREA)

        results = pose.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

        landmarks = results.pose_landmarks.landmark
        lmPose = mp_pose.PoseLandmark

        # print("CC")
        # print(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x)

        # Left shoulder
        l_shldr_x = int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w)
        l_shldr_y = int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h)
        # print(l_shldr_x)
        # print(l_shldr_y)

        #Right shoulder
        r_shldr_x = int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w)
        r_shldr_y = int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h)
        # print(r_shldr_x)
        # print(r_shldr_y)

        # Left ear
        l_ear_x = int(landmarks[mp_pose.PoseLandmark.LEFT_EAR].x * w)
        l_ear_y = int(landmarks[mp_pose.PoseLandmark.LEFT_EAR].y * h)
        # print(l_ear_x)
        # print(l_ear_y)

        #Right ear
        r_ear_x = int(landmarks[mp_pose.PoseLandmark.RIGHT_EAR].x * w)
        r_ear_y = int(landmarks[mp_pose.PoseLandmark.RIGHT_EAR].y * h)
        # print(r_ear_x)
        # print(r_ear_y)

        # Hip
        l_hip_x = int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * w)
        l_hip_y = int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * w)
        # print(l_hip_x)
        # print(l_hip_y)
        
        offset = findDistance(l_shldr_x,l_shldr_y,r_shldr_x,r_shldr_y)
        # print(offset)

        if offset < 125:
            cv.putText(frame, str(int(offset)) + ' Aligned', (w - 250, 30), font, 0.9, green, 2)
        else:
            cv.putText(frame, str(int(offset)) + ' Not Aligned', (w - 250, 30), font, 0.9, red, 2)

        neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
        # torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)   

        # print(neck_inclination)
        # print(torso_inclination)

        # Draw landmarks.
        cv.circle(frame, (l_shldr_x, l_shldr_y), 7, yellow, -1)
        cv.circle(frame, (l_ear_x, l_ear_y), 7, yellow, -1)
        
        # Let's take y - coordinate of P3 100px above x1,  for display elegance.
        # Although we are taking y = 0 while calculating angle between P1,P2,P3.
        cv.circle(frame, (l_shldr_x, l_shldr_y - 100), 7, yellow, -1)
        cv.circle(frame, (r_shldr_x, r_shldr_y), 7, pink, -1)
        cv.circle(frame, (l_hip_x, l_hip_y), 7, yellow, -1)
        
        # Similarly, here we are taking y - coordinate 100px above x1. Note that
        # you can take any value for y, not necessarily 100 or 200 pixels.
        cv.circle(frame, (l_hip_x, l_hip_y - 100), 7, yellow, -1)
        
        # Put text, Posture and angle inclination.
        # Text string for display.
        # angle_text_string = 'Neck : ' + str(int(neck_inclination)) + '  Torso : ' + str(int(torso_inclination))

        
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # print(results)

        cv.imshow('Screen', frame)
        if (cv.waitKey(10) & 0xFF == ord('q')):
            break    

cap.release
cv.destroyAllWindows





