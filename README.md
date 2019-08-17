# Real Time Face Emotion Detection #

This is a demo for the Berkeley GDSO 2019 DataScience Workshop.  Team members include Susan Hao, Zhimin Chen, Daniel Wooten, and Kilean Hwang.  We were mentored by Frank Cleary.

This demo uses open cv to read in your webcam stream, and runs each frame through a emotion recognition model.  The demo then takes an emoji associated with the predicted emotion of your face and overlays it on top of your face.  It additionally outputs a probability bar graph with the probability of each emotion outputted from the model.

**Requirements**
- Please make sure you install all the requirments for python as described in the requirements.txt file.  
- You will need to create a face_models folder in the root directory.  Put the .h5 and .json files in that directory.
  - change in the run_demo.py file the model names
  
**To Run**
- go into the demo folder and run 'python run_demo.py'
- in your web browser, go to 'localhost:2000'

**Pre-trained model**
- We used transfer learning with pre-trained VGGFace in Tensorflow Keras. The first 134 layers were frozen and the last 40 layers were set to be trainable. Adam optimizer and a learning rate of 0.001 were used. The best performing model achieved a validation accuracy of 64.4% and a validation loss of 0.98.
- Model: https://drive.google.com/open?id=1SSNYGsOaGDvnk1PPk4TE89rGFi0Krk4C
- Weights: https://drive.google.com/open?id=1vyUOonGTS4Jhy-xQmChatx6VSVn3jgwZ


**Training and validation data**
- FER dataset (~25K cropped faces with 7 emotion categories; 48x48x1): https://drive.google.com/open?id=1sD2WdIzrHSlbL88emQAXD0fF8ogi85dC
- Collapsed across  multiple datasets (-80k cropped faces with 7 emotion categories; 224x224x3): https://drive.google.com/open?id=1au4YEdJOTZeyGUxZfPIzWwcTf-DV-Uon; The same dataset but resized to 48x48x1: https://drive.google.com/open?id=1sKHM8txGFycJTZ2BoD6qnS0ju6YHjUUz

**References**
- we used the following sources as references for our demo.
https://towardsdatascience.com/from-raw-images-to-real-time-predictions-with-deep-learning-ddbbda1be0e4
