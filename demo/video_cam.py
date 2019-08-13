
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
            faces = facec.detectMultiScale(fr, 1.3, 5)

            for (x, y, w, h) in faces:
                fc = fr[y:y+h, x:x+w]
                roi = cv2.resize(fc, (224, 224))
                pred = model.predict_emotion_class(roi.reshape(1, 224, 224, 3))

                cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
                cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)
                # emote = cv2.resize(self.PICS_LIST[pred], (w, h))
                # fr[ x : x + w , y : y + h ] = emote

            _, jpeg = cv2.imencode('.jpg', fr)
            return jpeg.tobytes()
        else:
            print ('ERROR')
