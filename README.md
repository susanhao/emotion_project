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

**References**
- we used the following sources as references for our demo.
https://towardsdatascience.com/from-raw-images-to-real-time-predictions-with-deep-learning-ddbbda1be0e4
