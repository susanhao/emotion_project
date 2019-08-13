from keras.models import Sequential, Model, model_from_json
from keras.callbacks import ModelCheckpoint,TensorBoard

#runs the model specified
#train data follos format [x, y].  Same with val_data
def run_model(model, model_name, epochs, train_data, val_data, batch_size):
    # save best weights
    log_dir="logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    checkpointer = ModelCheckpoint(filepath=working_dir + 'checkpoints/' + '%s.h5'%model_name, verbose=1)
    tensorboard_callback = TensorBoard(log_dir=log_dir)

    # run model
    history = model.fit(train_data[0], train_data[1], epochs=epochs,
                   shuffle=True,
                   batch_size=batch_size, validation_data=(val_data[0], val_data[1]),
                   callbacks=[checkpointer,tensorboard_callback], verbose=2)
    return history

#saves model to json file saves model weights to h5 file
def save_model(model, model_name, working_dir):

    # save model to json
    model_json = model.to_json()
    with open(working_dir + 'models/' + "%s.json"%model_name, "w") as json_file:
        json_file.write(model_json)

    #save model weights
    #save weights
    model.save_weights(working_dir +'models/' + '%s_weights.h5'%model_name)

class Face_Model():


  def __init__(self, model_info_path, model_name, weight_name):
    self.model = self.load_model(model_info_path + model_name)
    self.model = self.load_weights(self.model, model_info_path + weight_name)
    EMOTIONS_LIST = {0:"Angry", 1:"Disgust", 2:"Fear", 3:"Happy", 4:"Neutral", 5:"Sad", 6:"Surprise"}
    PICS_LIST = None

  def load_emote_pics(self)
  """ This method of the Face_Model class loads emotion pictures from memory
      into a dict whose indices match the EMOTION_LIST indices. This dict is
      saved as an attribute of the class instance which calls it
  """
   
   self.PICS_LIST = dict()
   
   self.PICT_LIST[0] = cv2.imread('../pics/angry.jpeg')
    
   self.PICT_LIST[1] = cv2.imread('../pics/disgusted.jpeg')
    
   self.PICT_LIST[2] = cv2.imread('../pics/fear.jpeg')
    
   self.PICT_LIST[3] = cv2.imread('../pics/happy.jpeg')
    
   self.PICT_LIST[4] = cv2.imread('../pics/neutral.jpeg')
    
   self.PICT_LIST[5] = cv2.imread('../pics/sad.jpeg')
    
   self.PICT_LIST[6] = cv2.imread('../pics/surprise.jpeg')
    
    
  def load_model(self, path, json=True):
    #load model
    if json == False:
      print ('havent gotten that far yet')
    else:
      print (path)
      with open (path, 'r') as f:
        model = model_from_json(f.read())
      return model
   
  def load_weights (self, model, path, h5 = True):
    #loads weights
    if h5 == False:
      print ('havent gotten that far yet')
    else:
      model.load_weights(path)
      return model

  def predict_emotion_class(self, model, path, h5 = True):
    #loads weights
    if h5 == False:
      print ('havent gotten that far yet')
    else:
      self.preds = self.model.predict(img)
      return Face_Model.EMOTIONS_LIST[np.argmax(self.preds)]
