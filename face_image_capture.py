import cv2
import dlib
import numpy as np
from JetsonCamera import Camera
from Focuser import Focuser
from AutoFocus import AutoFocus
import time
import sys
from FaceImageQuality import FaceImageQuality

# Load the face detector (Haar Cascade) and shape predictor
face_cascade = cv2.CascadeClassifier("shape_files/haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor("shape_files/shape_predictor_68_face_landmarks.dat")

# Define the camera's field of view (FOV) angles
camera_fov_horizontal = 96  # Adjust this value based on your camera's FOV
camera_fov_vertical = 33    # Adjust this value based on your camera's FOV

y_default_post = 165
x_default_pos  = 90

def init_focuser(focuser):
    focuser.set(Focuser.OPT_MOTOR_Y,y_default_post)
    focuser.set(Focuser.OPT_MOTOR_X,x_default_pos)

def autofous_first(camera,focuser):
    #open camera preview
    camera.start_preview()
    print("Focus value = %d\n" % focuser.get(Focuser.OPT_FOCUS))
    auto_focus = AutoFocus(focuser,camera)
    auto_focus.debug = True
    begin = time.time()
    max_index,max_value = auto_focus.startFocus()
    # max_index,max_value = auto_focus.startFocus2()
    print("total time = %lf"%(time.time() - begin))
    #Adjust focus to the best
    #time.sleep(1)
    print("max index = %d,max value = %lf" % (max_index,max_value))
    camera.stop_preview()

x_min,x_min = 60,120
y_min,y_max = 90,175 
def search_face(focuser):
    motor_step = 2
    global y_default_post

    current_position_y = focuser.get(Focuser.OPT_MOTOR_Y)
    if  current_position_y < y_default_post:
        focuser.set(Focuser.OPT_MOTOR_Y,current_position_y - motor_step)




# input

trial_id = input("Input Trial Id: ")
print(trial_id)

# Load the video file
camera = Camera()
focuser = Focuser(1)
init_focuser(focuser)
autofous_first(camera,focuser)

frame_count = 0
no_detection_count = 0

while True:
    #ret, frame = video_capture.read()
    #if not ret:
    #    break
    frame = camera.getFrame()
    frame_org = frame.copy()
    frame = cv2.resize(frame,(360,240))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    frame_count += 1
     
    # Detect faces using Haar Cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) > 0:
        no_detection_count = 0
    else:
        no_detection_count +=1

    # print("No deteciton", no_detection_count)

    # if no_detection_count > 1*30:
    #     print(no_detection_count)
    #     search_face()

    for (x, y, w, h) in faces:
        # Convert the OpenCV rectangle to a dlib rectangle
        dlib_rect = dlib.rectangle(x, y, x + w, y + h)

        # Get the facial landmarks
        landmarks = predictor(gray, dlib_rect)

        face_image_quality = FaceImageQuality()
        check_image_quality = face_image_quality.check_image_quality(frame, landmarks)
        print(check_image_quality)
            # Extract and plot the landmarks for the eyes
        #for i in range(36, 48):  # Landmarks 36-47 correspond to the eyes
        for i in range(0,67):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)

        # Find the center between the two eyes
        left_eye_x = np.mean([landmarks.part(i).x for i in range(36, 42)])
        left_eye_y = np.mean([landmarks.part(i).y for i in range(36, 42)])

        right_eye_x = np.mean([landmarks.part(i).x for i in range(42, 48)])
        right_eye_y = np.mean([landmarks.part(i).y for i in range(42, 48)])

        eye_center_x = int((left_eye_x + right_eye_x) / 2)
        eye_center_y = int((left_eye_y + right_eye_y) / 2)
        # Draw a circle at the center between the eyes
        cv2.circle(frame, (eye_center_x, eye_center_y), 2, (255, 255, 255), -1)

        
        # Calculate the pan and tilt angles relative to the center of the image
        center_x = frame.shape[1] / 2
        center_y = frame.shape[0] / 2
        pan_angle = (eye_center_x - center_x) / center_x * (camera_fov_horizontal / 2)
        tilt_angle = (center_y - eye_center_y) / center_y * (camera_fov_vertical / 2)

        print("panx:",pan_angle,'tilty:',tilt_angle)

        motor_step = 2 
        if abs(pan_angle) > 10:
            print(pan_angle, tilt_angle)
            if pan_angle < 0:
                focuser.set(Focuser.OPT_MOTOR_X,focuser.get(Focuser.OPT_MOTOR_X) + motor_step)
            if pan_angle > 0:
                focuser.set(Focuser.OPT_MOTOR_X,focuser.get(Focuser.OPT_MOTOR_X) - motor_step)

        if abs(tilt_angle) > 4:
            current_position_y = focuser.get(Focuser.OPT_MOTOR_Y)
            if tilt_angle < 0 and ((abs(current_position_y) + motor_step) < 175):
                focuser.set(Focuser.OPT_MOTOR_Y,current_position_y + motor_step)
            if tilt_angle > 0:
                focuser.set(Focuser.OPT_MOTOR_Y,current_position_y - motor_step)

        #take one
        break
    

    # Display the video frame with the center point between the eyes
    cv2.imshow("Keep face in the center", frame)
    k = cv2.waitKey(1) & 0xFF 
    if k == ord('q'):  # Press 'Esc' to exit the video
        break
    
    if check_image_quality[0] == True:
        cv2.imwrite(f'images/{str(trial_id)}_{str(frame_count)}.png',frame_org)
        print("Good Image saved")
    else:
        cv2.imwrite(f'rejected_images/{str(trial_id)}_{str(frame_count)}.png',frame_org)
        print("Bad Image saved")  

    # if k == ord('o'):
    #     focuser.set(Focuser.OPT_MOTOR_Y,current_position_y + 2)
    # if k == ord('l'):
    #     focuser.set(Focuser.OPT_MOTOR_Y,current_position_y - 2)


camera.close()
cv2.destroyAllWindows()




# for jetson nano, python 3.6 
#if dlib does not install then use this
# wget http://dlib.net/files/dlib-19.21.tar.bz2
# tar jxvf dlib-19.17.tar.bz2
# cd dlib-19.21/
# mkdir build
# cd build/
# #cmake ..
# cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1
# cmake --build .
# cd ../
# $ sudo python3 setup.py install
