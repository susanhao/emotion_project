
import cv2
import numpy as np
from model_func import Face_Model
import matplotlib.pyplot as plt
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self, model):
        ret, fr = self.video.read()

        faces = facec.detectMultiScale(fr, 1.3, 5)

        #predicting emotion for each face and outputting emoji and graph
        for index, (x, y, w, h) in enumerate(faces):
            fc = fr[y:y+h, x:x+w]
            roi = cv2.resize(fc, (224, 224))
            #predict emotion
            pred = model.predict_emotion_class(roi.reshape(1, 224, 224, 3))
            
            #only graphs it for the first 3 faces.
            if index < 4:
                #graph output
                fig = plt.figure(figsize= (2,1.5), dpi = 100)
                plt.bar(Face_Model.EMOTE_DICT.keys(), model.preds[0])
                plt.xticks(fontsize = 6.5, rotation = 90)
                plt.ylim([0, 1])
                plt.yticks(fontsize = 6.5)
                plt.title('Face %d'%(index + 1))
                plt.tight_layout()
                fig.savefig('plot.png', dpi = 100)
                plt_img = cv2.imread('plot.png')
                fr[index*200:index*200 + 150, 1080:1080 + 200] = plt_img

            #output emoji overlay 
            emote = cv2.resize(model.load_emote_pics(), (w, h))
            added_emote = cv2.addWeighted(fr[y: y+h, x:x+w], 0.5, emote, 0.4, 0)
            fr[y: y+h, x: x+w] = added_emote

        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()
