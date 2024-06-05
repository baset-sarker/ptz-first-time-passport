import cv2
import numpy as np
import math
import time
from PIL import Image
from scipy.spatial import distance as dist
from imutils import face_utils
from numpy.linalg import norm
from dlib import get_frontal_face_detector,shape_predictor
import thresholds  as th
import messages as msg


class FaceImageQuality:
    
    def __init__(self):
        self.detector = None
        self.predictor = None
        #self.detector = get_frontal_face_detector()
        #self.predictor = shape_predictor("shape_files/shape_predictor_68_face_landmarks.dat")

        self.face_quality_param = {
            'brightness': {'result':False,'value': None, 'msg': None},
            'blur': {'result':False,'value': None, 'msg': None},
            'background_color': {'result':False,'value': None, 'msg': None},
            'washed_out': {'result':False,'value': None, 'msg': None},
            'pixelation': {'result':False,'value': None, 'msg': None},
            'face_present': {'result':False,'value': None, 'msg': None},
            'head_position': {'result':False,'value': None, 'msg': None},
            'pitch': {'result':False,'value': None, 'msg': None},
            'roll': {'result':False,'value': None, 'msg': None},
            'eye': {'result':False,'value': None, 'msg': None},
            'mouth': {'result':False,'value': None, 'msg': None},
            'red_eye': {'result':False,'value': None, 'msg': None},
        }

    def angle_between(self,p1, p2):
        ''' 
        Calculates the angle between two points in radians.
        
        Parameters:
        p1 (tuple): The coordinates of the first point.
        p2 (tuple): The coordinates of the second point.
        
        Returns:
        float: The angle between the two points in degrees.
        '''
        ang1 = np.arctan2(*p1[::-1])
        ang2 = np.arctan2(*p2[::-1])
        return np.rad2deg((ang1 - ang2) % (2 * np.pi))


    def dist_ratio(self,jaw,nose):
            '''
            Calculate the distance ratio between the jaw and nose landmarks.

            Parameters:
            jaw (list): List of jaw landmarks.
            nose (list): List of nose landmarks.

            Returns:
            float: The distance ratio between the jaw and nose landmarks.
            '''
            A = dist.euclidean(jaw[0], nose[0])
            B = dist.euclidean(nose[0], jaw[16])	
        
            if B == 0:
                return 0
            else:
                return A/B

    def get_dist_ratio(self,a,b,c,d):
            '''
            Calculate the distance ratio between two pairs of points.
            
            Parameters:
                a (tuple): The coordinates of the first point.
                b (tuple): The coordinates of the second point.
                c (tuple): The coordinates of the third point.
                d (tuple): The coordinates of the fourth point.
            
            Returns:
                float: The distance ratio between the two pairs of points.
            '''
            A = dist.euclidean(a,b)
            B = dist.euclidean(c,d)	
            if B == 0:
                return 0
            else:
                return A/B

    def getAngle(self, a, b, c):
        '''
        Calculate the angle between three points.

        Args:
            a (tuple): The coordinates of point a.
            b (tuple): The coordinates of point b.
            c (tuple): The coordinates of point c.

        Returns:
            float: The angle between points a, b, and c in degrees.
        '''
        ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
        return ang
    
    def cv2_to_PIL(self,imgOpenCV): 
        '''
        Converts an OpenCV image (in BGR format) to a PIL image (in RGB format).

        Parameters:
            imgOpenCV (numpy.ndarray): The OpenCV image to be converted.

        Returns:
            PIL.Image.Image: The converted PIL image.
        '''
        return Image.fromarray(cv2.cvtColor(imgOpenCV, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
    

    def get_brightness(self, img, lower_threshold, upper_threshold):
        '''
        Calculate the brightness value of an image.

        Args:
            img (numpy.ndarray): The input image.
            lower_threshold (float): The lower threshold for the brightness value.
            upper_threshold (float): The upper threshold for the brightness value.

        Returns:
            dict: A dictionary containing the result, value, and message of the brightness check.
                - result (bool): True if the brightness value is within the specified thresholds, False otherwise.
                - value (float): The calculated brightness value.
                - msg (str): A message indicating the result of the brightness check.

        '''
        brightness_value = 0
        if len(img.shape) == 3:
            brightness_value = np.average(norm(img, axis=2)) / np.sqrt(3)
        else:
            # Grayscale
            brightness_value = np.average(img)

        if brightness_value > lower_threshold and brightness_value < upper_threshold:
            return { 'result': True, 'value': brightness_value, 'msg': msg.BRIGHTNESS_MSG_OK }
        else:
            return { 'result': False, 'value': brightness_value, 'msg': msg.BRIGHTNESS_MSG_FAIL }

    # This function calculates the brightness value of an image and checks if it falls within the specified threshold.
        
    def check_blur_image(self, gray_image, lower_threshold, upper_threshold):
            '''
            Check if the given gray image is blurry based on the Laplacian variance.

            Parameters:
                gray_image (numpy.ndarray): The gray image to be checked.
                lower_threshold (float): The lower threshold for the Laplacian variance.
                upper_threshold (float): The upper threshold for the Laplacian variance.

            Returns:
                dict: A dictionary containing the result, value, and message of the blure check.
                    - result (bool): True if the Laplacian variance is within the specified thresholds, False otherwise.
                    - value (float): The calculated Laplacian variance.
                    - msg (str): A message indicating the result of the blur check.

            '''
            # Calculate the Laplacian variance of the gray image
            laplacian_value = cv2.Laplacian(gray_image, cv2.CV_64F).var()

            # Check if the Laplacian variance is within the specified thresholds
            if laplacian_value < lower_threshold or laplacian_value > upper_threshold:
                return { 'result': False, 'value': laplacian_value, 'msg': msg.BLUR_MSG_FAIL }
            else:
                return { 'result': True, 'value': laplacian_value, 'msg': msg.BLUR_MSG_OK }
        
        
    def check_pixelation(self, gray_image, lower_threshold, upper_threshold):
        '''
        Check if the given gray image is pixelated based on the mean gradient magnitude.

        Parameters:
            gray_image (numpy.ndarray): The gray image to be checked.
            lower_threshold (float): The lower threshold for the mean gradient magnitude.
            upper_threshold (float): The upper threshold for the mean gradient magnitude.

        Returns:
            dict: A dictionary containing the result, value, and message of the pixelation check.
                - result (bool): True if the mean gradient magnitude is within the specified thresholds, False otherwise.
                - value (float): The calculated mean gradient magnitude.
                - msg (str): A message indicating the result of the pixelation check.

        '''
        # Calculate the horizontal and vertical gradients using Sobel operator
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        # Combine the gradients
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        # Calculate the mean gradient magnitude
        mean_gradient = np.mean(gradient_magnitude)

        if mean_gradient < lower_threshold or mean_gradient > upper_threshold:
            return { 'result': False, 'value': mean_gradient, 'msg': msg.PIXELATION_MSG_FAIL }
        else:
            return { 'result': True, 'value': mean_gradient, 'msg': msg.PIXELATION_MSG_OK }
    
    def check_washed_out(self, img):
        """
        Check if an image is washed out.

        Args:
            img (numpy.ndarray): The input image.

        Returns:
            dict: A dictionary containing the result, value, and message.
                - result (bool): True if the image is not washed out, False otherwise.
                - value (None): Placeholder for additional information (not used in this case).
                - msg (str): A message indicating the result.

        """
        # Convert to grayscale if colored
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Calculate the histogram
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])

        # Normalize the histogram for better visualization
        hist_norm = hist.ravel() / hist.sum()

        # Thresholds can be adjusted based on image characteristics
        peak_threshold = th.WASHED_OUT_PEAK_THRESHOLD  # Threshold for peak concentration
        low_threshold = th.WASHED_OUT_LOW_THRESHOLD  # Threshold for low values at extremes
        washed_out = False

        # Check if peak is too high and extremes are too low
        if max(hist_norm) > peak_threshold and (hist_norm[0] < low_threshold or hist_norm[-1] < low_threshold):
            washed_out = True
        
        if washed_out == True:
            return {'result': False, 'value': None, 'msg': msg.WASHED_OUT_MSG_FAIL}
        else:
            return {'result': True, 'value': None, 'msg': msg.WASHED_OUT_MSG_OK}    
        

    def check_nose_is_in_middle(self, img, x, y):
        """
        Caluated head position based on the nose position.
        Check if the nose position is in the middle of the image.

        Args:
            img (numpy.ndarray): The input image.
            x (int): The x-coordinate of the nose position.
            y (int): The y-coordinate of the nose position.

        Returns:
            dict: A dictionary containing the result of the check, the value (which is None in this case), and a message.

        """
        h, w = img.shape[:2]
        x_percent = x * 100 / w
        y_percent = y * 100 / h

        # Check if the nose position is within the defined range
        if (x_percent > 42 and x_percent < 58) and (y_percent > 32 and y_percent < 40):
            return {'result': True, 'value': None, 'msg': msg.HEAD_POSITION_MSG_OK}
        else:
            return {'result': False, 'value': None, 'msg': msg.HEAD_POSITION_MSG_FAIL}
        
    def is_nose_is_in_middle(self,img,x,y):
        
        h, w = img.shape[:2]
        x_percent = x*100/w
        y_percent = y*100/h

        #print("Nose position in percent %",x_percent,y_percent)
        if (x_percent > 42 and x_percent < 58) and (y_percent > 32 and y_percent < 40):
            return {'result':True,'value':None,'msg':msg.HEAD_POSITION_MSG_OK}
        else:
            return {'result':False,'value':None,'msg':msg.HEAD_POSITION_MSG_FAIL}
    
    # required pil image so converted from cv2 first
    def check_background_color_white(self, cv2_image):
        '''
        Check if the background color of the image is white.

        Args:
            cv2_image (numpy.ndarray): The input image in OpenCV format.

        Returns:
            dict: A dictionary containing the result of the background color check.
                - 'result' (bool): True if the background color is white, False otherwise.
                - 'value' (None): Placeholder for additional information (not used in this function).
                - 'msg' (str): A message indicating the result of the background color check.

        '''
        # convert cv2 image to pil
        im = self.cv2_to_PIL(cv2_image)

        # Setting the points for cropped image
        left = 0
        top = 0
        right = im.width
        bottom = im.height - im.height * 97/100  # take only 3% from top

        # Cropped image of above dimension
        # (It will not change original image)
        im1 = im.crop((left, top, right, bottom))

        n, rgb = max(im1.getcolors(im1.size[0]*im1.size[1]))

        if rgb[0] > th.BACKGROUND_COLOR_THRESHOLD_LOW and rgb[1] > th.BACKGROUND_COLOR_THRESHOLD_LOW and rgb[2] > th.BACKGROUND_COLOR_THRESHOLD_LOW:
            return {'result': True, 'value': None, 'msg': msg.BACKGROUND_COLOR_MSG_OK}
        else:
            return {'result': False, 'value': None, 'msg': msg.BACKGROUND_COLOR_MSG_FAIL}
        

    def is_one_eye_red(self,eye):
        '''
        Check if there is red eye in the given eye image.

        Parameters:
        eye (numpy.ndarray): The eye image to be checked.

        Returns:
        bool: True if red eye is detected, False otherwise.
        '''
        b = eye[:, :, 0]
        g = eye[:, :, 1]
        r = eye[:, :, 2]
        # Add the green and blue channels.
        bg = cv2.add(b, g)
        # Simple red eye detector.
        mask = (r > 170) &  (r > bg)
        # Convert the mask to uint8 format.
        mask = mask.astype(np.uint8)*255

        mask_size = mask.size
        n_zeros = np.count_nonzero(mask==0)

        if  n_zeros < mask_size :
            #print("Red Eye detected")
            return True
        else:
            #print("No red eye detected")
            return False
        
    def detect_red_eye(self,photo,shape):

        x1=shape.part(36).x 
        x2=shape.part(39).x 
        y1=shape.part(37).y 
        y2=shape.part(40).y
        lefteye=photo[y1:y2,x1:x2]

        check_left_eye = self.is_one_eye_red(lefteye)

        x1=shape.part(42).x 
        x2=shape.part(45).x #43 46 #44 47 
        y1=shape.part(43).y 
        y2=shape.part(46).y 
        righteye=photo[y1:y2,x1:x2]

        check_right_eye = self.is_one_eye_red(righteye)

        if check_left_eye == False or check_right_eye == False:
            return {'result':True,'value':None,'msg':msg.RED_EYE_MSG_OK}
        else:
            return {'result':False,'value':None,'msg':msg.RED_EYE_MSG_FAIL}


    def check_pitch_value(self, jaw, nose, low_threshold, high_threshold):
        """
        Check the pitch value of the face image based on the jaw and nose coordinates.

        Args:
            jaw (float): The jaw coordinate value.
            nose (float): The nose coordinate value.
            low_threshold (float): The lower threshold for the pitch value.
            high_threshold (float): The upper threshold for the pitch value.

        Returns:
            dict: A dictionary containing the result of the pitch check, the pitch value, and a message.
                - 'result' (bool): True if the pitch value is within the threshold, False otherwise.
                - 'value' (float): The calculated pitch value.
                - 'msg' (str): A message indicating the result of the pitch check.
        """
        d_ratio = self.dist_ratio(jaw, nose)
        
        if d_ratio > high_threshold or d_ratio < low_threshold:
            return {'result': False, 'value': None, 'msg': msg.PITCH_MSG_FAIL}
        else:
            return {'result': True, 'value': None, 'msg': msg.PITCH_MSG_OK}
        
    
    def check_roll_angle(self, jaw):
            '''
            Check the roll angle of the jaw.

            Parameters:
            - jaw (list): A list of jaw landmarks.

            Returns:
            - dict: A dictionary containing the result of the roll angle check, the roll angle value, and a message.

            '''
            jaw_angle = self.getAngle(tuple(jaw[16]), jaw[0], (jaw[16][0], jaw[0][1]))
            if jaw_angle > th.ROLL_ANGLE or jaw_angle < -th.ROLL_ANGLE:
                return {'result': False, 'value': jaw_angle, 'msg': msg.ROLL_MSG_FAIL}
            else:
                return {'result': True, 'value': jaw_angle, 'msg': msg.ROLL_MSG_OK}
            
        
    def check_eye_open(self,shape):
            '''
            Check if the eyes in the given facial shape are open or closed.

            Parameters:
            - shape: A list of facial landmarks coordinates.

            Returns:
            - A dictionary with the following keys:
                - 'result': A boolean indicating if the eyes are open or closed.
                - 'value': None (not used in this function).
                - 'msg': A message indicating the result of the eye check.

            '''
            left_eye_distance_ratio = self.get_dist_ratio(shape[42],shape[45],shape[43],shape[47])
            right_eye_distance_ratio = self.get_dist_ratio(shape[36],shape[39],shape[37],shape[41])

            if left_eye_distance_ratio > th.EYE_DISTANCE_RATIO or right_eye_distance_ratio > th.EYE_DISTANCE_RATIO:
                return {'result': False, 'value': None, 'msg': msg.EYE_DISTANCE_MSG_FAIL}
            else:
                return {'result': True, 'value': None, 'msg': msg.EYE_DISTANCE_MSG_OK}
    
    def mouth_open(self,shape):
            '''
            Check if the mouth is open based on the distance ratio between specific facial landmarks.

            Parameters:
            shape (list): A list of facial landmarks coordinates.

            Returns:
            dict: A dictionary containing the result of the mouth open check, the distance ratio value, and a message.

            '''
            mouth_distance_ratio = self.get_dist_ratio(shape[48],shape[54],shape[62],shape[66])
            if mouth_distance_ratio < th.MOUTH_OPEN_LOW or mouth_distance_ratio > th.MOUTH_OPEN_HIGH :
                return {'result': False, 'value': mouth_distance_ratio, 'msg': msg.MOUTH_OPEN_MSG_FAIL}
            else:
                return {'result': True, 'value': mouth_distance_ratio, 'msg': msg.MOUTH_OPEN_MSG_OK}
        
    def check_if_face_present(self, image):
        '''
        Check if a face is present in the given image.

        Parameters:
        image (numpy.ndarray): The input image.

        Returns:
        tuple: A tuple containing the following elements:
            - bool: True if a face is present, False otherwise.
            - numpy.ndarray: The jaw coordinates of the detected face.
            - numpy.ndarray: The nose coordinates of the detected face.
            - numpy.ndarray: The shape of the detected face.
            - int: The x-coordinate of the nose.
            - int: The y-coordinate of the nose.
        '''
        (jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]
        (nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]

        rects = self.detector(image, 0)
        jaw = None
        if len(rects) > 0:
            rect = rects[0]
            shape = self.predictor(image, rect)
            shape_numpy = face_utils.shape_to_np(shape)
            jaw = shape_numpy[jStart:jEnd]
            nose = shape_numpy[nStart:nEnd]
            nose_x = nose[0][0]
            nose_y = nose[0][1]
            return True, jaw, nose, shape, shape_numpy, nose_x, nose_y
        else:
            return False, None, None, None, None, None, None
        
    def check_if_face_present_from_given_landmarks(self, shape):
        '''
        Check if a face is present in the given image.

        Parameters:
        shape (numpy.ndarray): The input image.

        Returns:
        tuple: A tuple containing the following elements:
            - bool: True if a face is present, False otherwise.
            - numpy.ndarray: The jaw coordinates of the detected face.
            - numpy.ndarray: The nose coordinates of the detected face.
            - numpy.ndarray: The shape of the detected face.
            - int: The x-coordinate of the nose.
            - int: The y-coordinate of the nose.
        '''
        (jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]
        (nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
        
        shape_numpy = face_utils.shape_to_np(shape)
        jaw = shape_numpy[jStart:jEnd]
        nose = shape_numpy[nStart:nEnd]
        jaw = shape_numpy[jStart:jEnd]
        nose = shape_numpy[nStart:nEnd]
        nose_x = nose[0][0]
        nose_y = nose[0][1]
        return True, jaw, nose, shape, shape_numpy, nose_x, nose_y


    def get_face_quality_params(self, image, shape=None):
        '''
        Calculates various quality parameters of a face image.

        Args:
            image (numpy.ndarray): The input face image.
            shape (numpy.ndarray): The facial landmarks of the face image
            if you want to pass from outside. it will not perform face detection again

        Returns:
            dict: A dictionary containing the calculated quality parameters.

        '''

        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate brightness value
        self.face_quality_param['brightness'] = self.get_brightness(
            image, th.BRIGHTNESS_THRESHOLD_LOW, th.BRIGHTNESS_THRESHOLD_HIGH)

        # Check for blur
        self.face_quality_param['blur'] = self.check_blur_image(
            gray, th.BLUR_THRESHOLD_LOW, th.BLUR_THRESHOLD_HIGH)

        # Check if background color is white
        self.face_quality_param['background_color'] = self.check_background_color_white(image)

        # Check for washed out image
        self.face_quality_param['washed_out'] = self.check_washed_out(image)

        # Check for pixelation
        self.face_quality_param['pixelation'] = self.check_pixelation(
            gray, th.PIXELATION_THRESHOLD_LOW, th.PIXELATION_THRESHOLD_HIGH)

        # Check if face is present
        if shape is not None:
            face_present, jaw, nose, shape, shape_numpy, nose_x, nose_y = self.check_if_face_present_from_given_landmarks(shape)
        else:
            face_present, jaw, nose, shape, shape_numpy, nose_x, nose_y = self.check_if_face_present(gray)
        
        if face_present == False:
            self.face_quality_param['face_present'] = {'result': face_present, 'value': None, 'msg': msg.FACE_PRESENT_MSG_FAIL}
        else:
            self.face_quality_param['face_present'] = {'result': face_present, 'value': None, 'msg': msg.FACE_PRESENT_MSG_OK}
            self.face_quality_param['head_position'] = self.check_nose_is_in_middle(image, nose_x, nose_y)
            self.face_quality_param['head_position'] = {'result':True, 'value':None,msg:msg.HEAD_POSITION_MSG_OK}
            self.face_quality_param['pitch'] = self.check_pitch_value(jaw, nose, th.PITCH_MIN, th.PITCH_MAX)
            self.face_quality_param['roll'] = self.check_roll_angle(jaw)
            self.face_quality_param['eye'] = self.check_eye_open(shape_numpy)
            self.face_quality_param['mouth'] = self.mouth_open(shape_numpy)
            self.face_quality_param['red_eye'] = self.detect_red_eye(image,shape)

        return self.face_quality_param


    def get_face_quality_results_only(self, image ,shape=None):
        '''
        Get the face quality results from the given image.

        Parameters:
            image (numpy.ndarray): The input image.
            shape (numpy.ndarray): The facial landmarks of the face image
            if you want to pass from outside. it will not perform face detection again

        Returns:
            list: A list of face quality results.

        '''
        # Get the face qualities from the image
        face_qualities = self.get_face_quality_params(image,shape)

        # Extract only the result from the face qualities
        results = []
        for key in face_qualities:
            results.append(face_qualities[key]['result'])

        return results

    def get_face_quality_values_only(self,image,shape=None):
        '''
         Get the face quality test values from the given image.

        Parameters:
            image (numpy.ndarray): The input image.
            shape (numpy.ndarray): The facial landmarks of the face image
            if you want to pass from outside. it will not perform face detection again

        Returns:
            list: A list of face quality values.
        '''

        face_qualities=self.get_face_quality_params(image,shape=None)
        # from the results ge only result give an arary
        values = []
        for key in face_qualities:
            values.append(face_qualities[key]['value'])
        return values
    
    def check_image_quality(self,image,shape=None):
        '''
        Check the quality of the input image pass/fail.
        returns the result and message of the image quality check.
        Parameters:
            image (numpy.ndarray): The input image.
            shape (numpy.ndarray): The facial landmarks of the face image
            if you want to pass from outside. it will not perform face detection again

        Returns:
            dict: A dictionary containing the results of the image quality checks.
        '''

        # Get the face quality parameters
        face_quality_params = self.get_face_quality_params(image)

        # Check if any of the face quality parameters failed
        # return that failed params result and msg

        for key in face_quality_params:
            if face_quality_params[key]['result'] == False:
                return face_quality_params[key]['result'],face_quality_params[key]['value'],face_quality_params[key]['msg']
        
        return True,None,"All face quality parameters are within the acceptable range"
import cv2
import numpy as np
import math
import time
from PIL import Image
from scipy.spatial import distance as dist
from imutils import face_utils
from numpy.linalg import norm
from dlib import get_frontal_face_detector,shape_predictor
import thresholds  as th
import messages as msg


class FaceImageQuality:
    
    def __init__(self):
        self.detector = get_frontal_face_detector()
        self.predictor = shape_predictor("shape_files/shape_predictor_68_face_landmarks.dat")

        self.face_quality_param = {
            'brightness': {'result':False,'value': None, 'msg': None},
            'blur': {'result':False,'value': None, 'msg': None},
            'background_color': {'result':False,'value': None, 'msg': None},
            'washed_out': {'result':False,'value': None, 'msg': None},
            'pixelation': {'result':False,'value': None, 'msg': None},
            'face_present': {'result':False,'value': None, 'msg': None},
            'head_position': {'result':False,'value': None, 'msg': None},
            'pitch': {'result':False,'value': None, 'msg': None},
            'roll': {'result':False,'value': None, 'msg': None},
            'eye': {'result':False,'value': None, 'msg': None},
            'mouth': {'result':False,'value': None, 'msg': None},
            'red_eye': {'result':False,'value': None, 'msg': None},
        }

    def angle_between(self,p1, p2):
        ''' 
        Calculates the angle between two points in radians.
        
        Parameters:
        p1 (tuple): The coordinates of the first point.
        p2 (tuple): The coordinates of the second point.
        
        Returns:
        float: The angle between the two points in degrees.
        '''
        ang1 = np.arctan2(*p1[::-1])
        ang2 = np.arctan2(*p2[::-1])
        return np.rad2deg((ang1 - ang2) % (2 * np.pi))


    def dist_ratio(self,jaw,nose):
            '''
            Calculate the distance ratio between the jaw and nose landmarks.

            Parameters:
            jaw (list): List of jaw landmarks.
            nose (list): List of nose landmarks.

            Returns:
            float: The distance ratio between the jaw and nose landmarks.
            '''
            A = dist.euclidean(jaw[0], nose[0])
            B = dist.euclidean(nose[0], jaw[16])	
        
            if B == 0:
                return 0
            else:
                return A/B

    def get_dist_ratio(self,a,b,c,d):
            '''
            Calculate the distance ratio between two pairs of points.
            
            Parameters:
                a (tuple): The coordinates of the first point.
                b (tuple): The coordinates of the second point.
                c (tuple): The coordinates of the third point.
                d (tuple): The coordinates of the fourth point.
            
            Returns:
                float: The distance ratio between the two pairs of points.
            '''
            A = dist.euclidean(a,b)
            B = dist.euclidean(c,d)	
            if B == 0:
                return 0
            else:
                return A/B

    def getAngle(self, a, b, c):
        '''
        Calculate the angle between three points.

        Args:
            a (tuple): The coordinates of point a.
            b (tuple): The coordinates of point b.
            c (tuple): The coordinates of point c.

        Returns:
            float: The angle between points a, b, and c in degrees.
        '''
        ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
        return ang
    
    def cv2_to_PIL(self,imgOpenCV): 
        '''
        Converts an OpenCV image (in BGR format) to a PIL image (in RGB format).

        Parameters:
            imgOpenCV (numpy.ndarray): The OpenCV image to be converted.

        Returns:
            PIL.Image.Image: The converted PIL image.
        '''
        return Image.fromarray(cv2.cvtColor(imgOpenCV, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
    

    def get_brightness(self, img, lower_threshold, upper_threshold):
        '''
        Calculate the brightness value of an image.

        Args:
            img (numpy.ndarray): The input image.
            lower_threshold (float): The lower threshold for the brightness value.
            upper_threshold (float): The upper threshold for the brightness value.

        Returns:
            dict: A dictionary containing the result, value, and message of the brightness check.
                - result (bool): True if the brightness value is within the specified thresholds, False otherwise.
                - value (float): The calculated brightness value.
                - msg (str): A message indicating the result of the brightness check.

        '''
        brightness_value = 0
        if len(img.shape) == 3:
            brightness_value = np.average(norm(img, axis=2)) / np.sqrt(3)
        else:
            # Grayscale
            brightness_value = np.average(img)

        if brightness_value > lower_threshold and brightness_value < upper_threshold:
            return { 'result': True, 'value': brightness_value, 'msg': msg.BRIGHTNESS_MSG_OK }
        else:
            return { 'result': False, 'value': brightness_value, 'msg': msg.BRIGHTNESS_MSG_FAIL }

    # This function calculates the brightness value of an image and checks if it falls within the specified threshold.
        
    def check_blur_image(self, gray_image, lower_threshold, upper_threshold):
            '''
            Check if the given gray image is blurry based on the Laplacian variance.

            Parameters:
                gray_image (numpy.ndarray): The gray image to be checked.
                lower_threshold (float): The lower threshold for the Laplacian variance.
                upper_threshold (float): The upper threshold for the Laplacian variance.

            Returns:
                dict: A dictionary containing the result, value, and message of the blure check.
                    - result (bool): True if the Laplacian variance is within the specified thresholds, False otherwise.
                    - value (float): The calculated Laplacian variance.
                    - msg (str): A message indicating the result of the blur check.

            '''
            # Calculate the Laplacian variance of the gray image
            laplacian_value = cv2.Laplacian(gray_image, cv2.CV_64F).var()

            # Check if the Laplacian variance is within the specified thresholds
            if laplacian_value < lower_threshold or laplacian_value > upper_threshold:
                return { 'result': False, 'value': laplacian_value, 'msg': msg.BLUR_MSG_FAIL }
            else:
                return { 'result': True, 'value': laplacian_value, 'msg': msg.BLUR_MSG_OK }
        
        
    def check_pixelation(self, gray_image, lower_threshold, upper_threshold):
        '''
        Check if the given gray image is pixelated based on the mean gradient magnitude.

        Parameters:
            gray_image (numpy.ndarray): The gray image to be checked.
            lower_threshold (float): The lower threshold for the mean gradient magnitude.
            upper_threshold (float): The upper threshold for the mean gradient magnitude.

        Returns:
            dict: A dictionary containing the result, value, and message of the pixelation check.
                - result (bool): True if the mean gradient magnitude is within the specified thresholds, False otherwise.
                - value (float): The calculated mean gradient magnitude.
                - msg (str): A message indicating the result of the pixelation check.

        '''
        # Calculate the horizontal and vertical gradients using Sobel operator
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        # Combine the gradients
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        # Calculate the mean gradient magnitude
        mean_gradient = np.mean(gradient_magnitude)

        if mean_gradient < lower_threshold or mean_gradient > upper_threshold:
            return { 'result': False, 'value': mean_gradient, 'msg': msg.PIXELATION_MSG_FAIL }
        else:
            return { 'result': True, 'value': mean_gradient, 'msg': msg.PIXELATION_MSG_OK }
    
    def check_washed_out(self, img):
        """
        Check if an image is washed out.

        Args:
            img (numpy.ndarray): The input image.

        Returns:
            dict: A dictionary containing the result, value, and message.
                - result (bool): True if the image is not washed out, False otherwise.
                - value (None): Placeholder for additional information (not used in this case).
                - msg (str): A message indicating the result.

        """
        # Convert to grayscale if colored
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Calculate the histogram
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])

        # Normalize the histogram for better visualization
        hist_norm = hist.ravel() / hist.sum()

        # Thresholds can be adjusted based on image characteristics
        peak_threshold = th.WASHED_OUT_PEAK_THRESHOLD  # Threshold for peak concentration
        low_threshold = th.WASHED_OUT_LOW_THRESHOLD  # Threshold for low values at extremes
        washed_out = False

        # Check if peak is too high and extremes are too low
        if max(hist_norm) > peak_threshold and (hist_norm[0] < low_threshold or hist_norm[-1] < low_threshold):
            washed_out = True
        
        if washed_out == True:
            return {'result': False, 'value': None, 'msg': msg.WASHED_OUT_MSG_FAIL}
        else:
            return {'result': True, 'value': None, 'msg': msg.WASHED_OUT_MSG_OK}    
        

    def check_nose_is_in_middle(self, img, x, y):
        """
        Caluated head position based on the nose position.
        Check if the nose position is in the middle of the image.

        Args:
            img (numpy.ndarray): The input image.
            x (int): The x-coordinate of the nose position.
            y (int): The y-coordinate of the nose position.

        Returns:
            dict: A dictionary containing the result of the check, the value (which is None in this case), and a message.

        """
        h, w = img.shape[:2]
        x_percent = x * 100 / w
        y_percent = y * 100 / h

        # Check if the nose position is within the defined range
        if (x_percent > 42 and x_percent < 58) and (y_percent > 32 and y_percent < 40):
            return {'result': True, 'value': None, 'msg': msg.HEAD_POSITION_MSG_OK}
        else:
            return {'result': False, 'value': None, 'msg': msg.HEAD_POSITION_MSG_FAIL}
        
    def is_nose_is_in_middle(self,img,x,y):
        
        h, w = img.shape[:2]
        x_percent = x*100/w
        y_percent = y*100/h

        #print("Nose position in percent %",x_percent,y_percent)
        if (x_percent > 42 and x_percent < 58) and (y_percent > 32 and y_percent < 40):
            return {'result':True,'value':None,'msg':msg.HEAD_POSITION_MSG_OK}
        else:
            return {'result':False,'value':None,'msg':msg.HEAD_POSITION_MSG_FAIL}
    
    # required pil image so converted from cv2 first
    def check_background_color_white(self, cv2_image):
        '''
        Check if the background color of the image is white.

        Args:
            cv2_image (numpy.ndarray): The input image in OpenCV format.

        Returns:
            dict: A dictionary containing the result of the background color check.
                - 'result' (bool): True if the background color is white, False otherwise.
                - 'value' (None): Placeholder for additional information (not used in this function).
                - 'msg' (str): A message indicating the result of the background color check.

        '''
        # convert cv2 image to pil
        im = self.cv2_to_PIL(cv2_image)

        # Setting the points for cropped image
        left = 0
        top = 0
        right = im.width
        bottom = im.height - im.height * 97/100  # take only 3% from top

        # Cropped image of above dimension
        # (It will not change original image)
        im1 = im.crop((left, top, right, bottom))

        n, rgb = max(im1.getcolors(im1.size[0]*im1.size[1]))

        if rgb[0] > th.BACKGROUND_COLOR_THRESHOLD_LOW and rgb[1] > th.BACKGROUND_COLOR_THRESHOLD_LOW and rgb[2] > th.BACKGROUND_COLOR_THRESHOLD_LOW:
            return {'result': True, 'value': None, 'msg': msg.BACKGROUND_COLOR_MSG_OK}
        else:
            return {'result': False, 'value': None, 'msg': msg.BACKGROUND_COLOR_MSG_FAIL}
        

    def is_one_eye_red(self,eye):
        '''
        Check if there is red eye in the given eye image.

        Parameters:
        eye (numpy.ndarray): The eye image to be checked.

        Returns:
        bool: True if red eye is detected, False otherwise.
        '''
        b = eye[:, :, 0]
        g = eye[:, :, 1]
        r = eye[:, :, 2]
        # Add the green and blue channels.
        bg = cv2.add(b, g)
        # Simple red eye detector.
        mask = (r > 170) &  (r > bg)
        # Convert the mask to uint8 format.
        mask = mask.astype(np.uint8)*255

        mask_size = mask.size
        n_zeros = np.count_nonzero(mask==0)

        if  n_zeros < mask_size :
            #print("Red Eye detected")
            return True
        else:
            #print("No red eye detected")
            return False
        
    def detect_red_eye(self,photo,shape):

        x1=shape.part(36).x 
        x2=shape.part(39).x 
        y1=shape.part(37).y 
        y2=shape.part(40).y
        lefteye=photo[y1:y2,x1:x2]

        check_left_eye = self.is_one_eye_red(lefteye)

        x1=shape.part(42).x 
        x2=shape.part(45).x #43 46 #44 47 
        y1=shape.part(43).y 
        y2=shape.part(46).y 
        righteye=photo[y1:y2,x1:x2]

        check_right_eye = self.is_one_eye_red(righteye)

        if check_left_eye == False or check_right_eye == False:
            return {'result':True,'value':None,'msg':msg.RED_EYE_MSG_OK}
        else:
            return {'result':False,'value':None,'msg':msg.RED_EYE_MSG_FAIL}


    def check_pitch_value(self, jaw, nose, low_threshold, high_threshold):
        """
        Check the pitch value of the face image based on the jaw and nose coordinates.

        Args:
            jaw (float): The jaw coordinate value.
            nose (float): The nose coordinate value.
            low_threshold (float): The lower threshold for the pitch value.
            high_threshold (float): The upper threshold for the pitch value.

        Returns:
            dict: A dictionary containing the result of the pitch check, the pitch value, and a message.
                - 'result' (bool): True if the pitch value is within the threshold, False otherwise.
                - 'value' (float): The calculated pitch value.
                - 'msg' (str): A message indicating the result of the pitch check.
        """
        d_ratio = self.dist_ratio(jaw, nose)
        
        if d_ratio > high_threshold or d_ratio < low_threshold:
            return {'result': False, 'value': None, 'msg': msg.PITCH_MSG_FAIL}
        else:
            return {'result': True, 'value': None, 'msg': msg.PITCH_MSG_OK}
        
    
    def check_roll_angle(self, jaw):
            '''
            Check the roll angle of the jaw.

            Parameters:
            - jaw (list): A list of jaw landmarks.

            Returns:
            - dict: A dictionary containing the result of the roll angle check, the roll angle value, and a message.

            '''
            jaw_angle = self.getAngle(tuple(jaw[16]), jaw[0], (jaw[16][0], jaw[0][1]))
            if jaw_angle > th.ROLL_ANGLE or jaw_angle < -th.ROLL_ANGLE:
                return {'result': False, 'value': jaw_angle, 'msg': msg.ROLL_MSG_FAIL}
            else:
                return {'result': True, 'value': jaw_angle, 'msg': msg.ROLL_MSG_OK}
            
        
    def check_eye_open(self,shape):
            '''
            Check if the eyes in the given facial shape are open or closed.

            Parameters:
            - shape: A list of facial landmarks coordinates.

            Returns:
            - A dictionary with the following keys:
                - 'result': A boolean indicating if the eyes are open or closed.
                - 'value': None (not used in this function).
                - 'msg': A message indicating the result of the eye check.

            '''
            left_eye_distance_ratio = self.get_dist_ratio(shape[42],shape[45],shape[43],shape[47])
            right_eye_distance_ratio = self.get_dist_ratio(shape[36],shape[39],shape[37],shape[41])

            if left_eye_distance_ratio > th.EYE_DISTANCE_RATIO or right_eye_distance_ratio > th.EYE_DISTANCE_RATIO:
                return {'result': False, 'value': None, 'msg': msg.EYE_DISTANCE_MSG_FAIL}
            else:
                return {'result': True, 'value': None, 'msg': msg.EYE_DISTANCE_MSG_OK}
    
    def mouth_open(self,shape):
            '''
            Check if the mouth is open based on the distance ratio between specific facial landmarks.

            Parameters:
            shape (list): A list of facial landmarks coordinates.

            Returns:
            dict: A dictionary containing the result of the mouth open check, the distance ratio value, and a message.

            '''
            mouth_distance_ratio = self.get_dist_ratio(shape[48],shape[54],shape[62],shape[66])
            if mouth_distance_ratio < th.MOUTH_OPEN_LOW or mouth_distance_ratio > th.MOUTH_OPEN_HIGH :
                return {'result': False, 'value': mouth_distance_ratio, 'msg': msg.MOUTH_OPEN_MSG_FAIL}
            else:
                return {'result': True, 'value': mouth_distance_ratio, 'msg': msg.MOUTH_OPEN_MSG_OK}
        
    def check_if_face_present(self, image):
        '''
        Check if a face is present in the given image.

        Parameters:
        image (numpy.ndarray): The input image.

        Returns:
        tuple: A tuple containing the following elements:
            - bool: True if a face is present, False otherwise.
            - numpy.ndarray: The jaw coordinates of the detected face.
            - numpy.ndarray: The nose coordinates of the detected face.
            - numpy.ndarray: The shape of the detected face.
            - int: The x-coordinate of the nose.
            - int: The y-coordinate of the nose.
        '''
        (jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]
        (nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]

        rects = self.detector(image, 0)
        jaw = None
        if len(rects) > 0:
            rect = rects[0]
            shape = self.predictor(image, rect)
            shape_numpy = face_utils.shape_to_np(shape)
            jaw = shape_numpy[jStart:jEnd]
            nose = shape_numpy[nStart:nEnd]
            nose_x = nose[0][0]
            nose_y = nose[0][1]
            return True, jaw, nose, shape, shape_numpy, nose_x, nose_y
        else:
            return False, None, None, None, None, None, None
        
    def check_if_face_present_from_given_landmarks(self, shape):
        '''
        Check if a face is present in the given image.

        Parameters:
        shape (numpy.ndarray): The input image.

        Returns:
        tuple: A tuple containing the following elements:
            - bool: True if a face is present, False otherwise.
            - numpy.ndarray: The jaw coordinates of the detected face.
            - numpy.ndarray: The nose coordinates of the detected face.
            - numpy.ndarray: The shape of the detected face.
            - int: The x-coordinate of the nose.
            - int: The y-coordinate of the nose.
        '''
        (jStart, jEnd) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]
        (nStart, nEnd) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
        
        shape_numpy = face_utils.shape_to_np(shape)
        jaw = shape_numpy[jStart:jEnd]
        nose = shape_numpy[nStart:nEnd]
        jaw = shape_numpy[jStart:jEnd]
        nose = shape_numpy[nStart:nEnd]
        nose_x = nose[0][0]
        nose_y = nose[0][1]
        return True, jaw, nose, shape, shape_numpy, nose_x, nose_y


    def get_face_quality_params(self, image, shape=None):
        '''
        Calculates various quality parameters of a face image.

        Args:
            image (numpy.ndarray): The input face image.
            shape (numpy.ndarray): The facial landmarks of the face image
            if you want to pass from outside. it will not perform face detection again

        Returns:
            dict: A dictionary containing the calculated quality parameters.

        '''

        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Calculate brightness value
        self.face_quality_param['brightness'] = self.get_brightness(
            image, th.BRIGHTNESS_THRESHOLD_LOW, th.BRIGHTNESS_THRESHOLD_HIGH)

        # Check for blur
        self.face_quality_param['blur'] = self.check_blur_image(
            gray, th.BLUR_THRESHOLD_LOW, th.BLUR_THRESHOLD_HIGH)

        # Check if background color is white
        self.face_quality_param['background_color'] = self.check_background_color_white(image)

        # Check for washed out image
        self.face_quality_param['washed_out'] = self.check_washed_out(image)

        # Check for pixelation
        self.face_quality_param['pixelation'] = self.check_pixelation(
            gray, th.PIXELATION_THRESHOLD_LOW, th.PIXELATION_THRESHOLD_HIGH)

        # Check if face is present
        if shape is not None:
            face_present, jaw, nose, shape, shape_numpy, nose_x, nose_y = self.check_if_face_present_from_given_landmarks(shape)
        else:
            face_present, jaw, nose, shape, shape_numpy, nose_x, nose_y = self.check_if_face_present(gray)
        
        if face_present == False:
            self.face_quality_param['face_present'] = {'result': face_present, 'value': None, 'msg': msg.FACE_PRESENT_MSG_FAIL}
        else:
            self.face_quality_param['face_present'] = {'result': face_present, 'value': None, 'msg': msg.FACE_PRESENT_MSG_OK}
            #self.face_quality_param['head_position'] = self.check_nose_is_in_middle(image, nose_x, nose_y)
            self.face_quality_param['head_position'] = {'result':True, 'value':None,msg:msg.HEAD_POSITION_MSG_OK}
            self.face_quality_param['pitch'] = self.check_pitch_value(jaw, nose, th.PITCH_MIN, th.PITCH_MAX)
            self.face_quality_param['roll'] = self.check_roll_angle(jaw)
            self.face_quality_param['eye'] = self.check_eye_open(shape_numpy)
            self.face_quality_param['mouth'] = self.mouth_open(shape_numpy)
            self.face_quality_param['red_eye'] = self.detect_red_eye(image,shape)

        return self.face_quality_param


    def get_face_quality_results_only(self, image ,shape=None):
        '''
        Get the face quality results from the given image.

        Parameters:
            image (numpy.ndarray): The input image.
            shape (numpy.ndarray): The facial landmarks of the face image
            if you want to pass from outside. it will not perform face detection again

        Returns:
            list: A list of face quality results.

        '''
        # Get the face qualities from the image
        face_qualities = self.get_face_quality_params(image,shape)

        # Extract only the result from the face qualities
        results = []
        for key in face_qualities:
            results.append(face_qualities[key]['result'])

        return results

    def get_face_quality_values_only(self,image,shape=None):
        '''
         Get the face quality test values from the given image.

        Parameters:
            image (numpy.ndarray): The input image.
            shape (numpy.ndarray): The facial landmarks of the face image
            if you want to pass from outside. it will not perform face detection again

        Returns:
            list: A list of face quality values.
        '''

        face_qualities=self.get_face_quality_params(image,shape=None)
        # from the results ge only result give an arary
        values = []
        for key in face_qualities:
            values.append(face_qualities[key]['value'])
        return values
    
    def check_image_quality(self,image,shape=None):
        '''
        Check the quality of the input image pass/fail.
        returns the result and message of the image quality check.
        Parameters:
            image (numpy.ndarray): The input image.
            shape (numpy.ndarray): The facial landmarks of the face image
            if you want to pass from outside. it will not perform face detection again

        Returns:
            dict: A dictionary containing the results of the image quality checks.
        '''

        # Get the face quality parameters
        face_quality_params = self.get_face_quality_params(image)

        # Check if any of the face quality parameters failed
        # return that failed params result and msg

        for key in face_quality_params:
            if face_quality_params[key]['result'] == False:
                return face_quality_params[key]['result'],face_quality_params[key]['value'],face_quality_params[key]['msg']
        
        return True,None,"All face quality parameters are within the acceptable range"

