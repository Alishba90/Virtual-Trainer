import time
import cv2
from flask import Flask, render_template, Response
import mediapipe as mp
import numpy as np
import sys


app = Flask(__name__)

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return round(angle) 


def track():
    previous_time = 0
    # creating our model to draw landmarks
    mp_drawing = mp.solutions.drawing_utils
    # creating our model to detected our pose
    my_pose = mp.solutions.pose
    

    """Video streaming generator function."""
    cap = cv2.VideoCapture(0)
    with my_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, img = cap.read()
            
            # converting image to RGB from BGR cuz mediapipe only work on RGB
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgRGB.flags.writeable=False
            results = pose.process(imgRGB)
            imgRGB.flags.writeable=True
            imgRGB = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2BGR)
            
            
            
            
            try:            
                landmarks = results.pose_landmarks.landmark
                    
                
                
                shoulder = [landmarks[my_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[my_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[my_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[my_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[my_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[my_pose.PoseLandmark.LEFT_WRIST.value].y]
                
                angle = calculate_angle(shoulder, elbow, wrist)
                cv2.putText(img, str(angle), (70, 50), cv2.FONT_HERSHEY_TRIPLEX, 3, (255, 0, 0), 3)
            except:
                pass
            
            mp_drawing.draw_landmarks(img, results.pose_landmarks, my_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )      

            # checking video frame rate
            current_time = time.time()
            fps = 1 / (current_time - previous_time)
            previous_time = current_time

            
            cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)
            # Writing FrameRate on video
            cv2.putText(img, str(angle), (70, 50), cv2.FONT_HERSHEY_TRIPLEX, 3, (255, 0, 0), 3)

            #cv2.imshow("Pose detection", img)
            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            key = cv2.waitKey(20)
            if key == 27:
                break
        cap.release()
    

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(track(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__=="__main__":
    app.run(debug=True)







