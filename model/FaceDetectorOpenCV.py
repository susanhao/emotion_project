import cv2

class FaceDetector(object):
    def __init__(self, xml_path):
        self.classifier = cv2.CascadeClassifier(xml_path)
    
    def detect(self, image, biggest_only=True):
      
        # the following parameters do not need to be changed unless the classifer is bad
        # how much the image size is reduced at each image scale
        scale_factor = 1.2
        # Parameter specifying how many neighbors each candidate rectangle should have to retain it.
        min_neighbors = 5 
        # Minimum possible object size. Objects smaller than that are ignored.
        min_size = (30, 30) 
        
        biggest_only = True
        faces_coord = self.classifier.detectMultiScale(image,
                                                       scaleFactor=scale_factor,
                                                       minNeighbors=min_neighbors,
                                                       minSize=min_size,
                                                       flags=cv2.CASCADE_SCALE_IMAGE)
        return faces_coord


def cut_faces(image, faces_coord, zoom_ratio = 0):
    # crop face
    faces = []
    
    for (x, y, w, h) in faces_coord:
        w_rm = int(zoom_ratio * w / 2)
        faces.append(image[y: y + h, x + w_rm: x + w - w_rm])
         
    return faces

def resize(images, size=(224, 224)):
    # resize the cropped face
    images_norm = []
    for image in images:
        if image.shape < size:
            image_norm = cv2.resize(image, size, 
                                    interpolation=cv2.INTER_AREA)
        else:
            image_norm = cv2.resize(image, size, 
                                    interpolation=cv2.INTER_CUBIC)
        images_norm.append(image_norm)

    return images_norm



def normalize_faces(image, faces_coord, resized_shape, zoom_ratio):
    # a wrapper for cropping and resizing
    faces = cut_faces(image, faces_coord, zoom_ratio)
    faces = resize(faces,size = resized_shape)
    
    return faces
  