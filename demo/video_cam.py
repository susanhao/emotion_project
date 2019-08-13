
import cv2
import numpy as np
from model_func import Face_Model

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self, model):
        ret, fr = self.video.read()

        if ret == True:
            gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            faces = facec.detectMultiScale(gray_fr, 1.3, 5)

            for (x, y, w, h) in faces:
                fc = gray_fr[y:y+h, x:x+w]

                roi = cv2.resize(fc, (48, 48))
                pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])

                emote = cv2.resize(self.PICS_LIST[pred], (w, h))
                fr[ x : x + w , y : y + h ] = emote

            _, jpeg = cv2.imencode('.jpg', fr)
            return jpeg.tobytes()
        else:
            print ('ERROR')
