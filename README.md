# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, I used CNN to train a simulator to drive in an autonomous mode.
I created a CNN Lenet model using Keras. The model outputs a steering angle that it predicts from the simulator scene images.

Files 
---
The project includes the following files: 
* python_venv_reqs.txt - output of `pip freeze`.  The file lists the libraries that are _required_ to train/drive the project. 
Run the following commands to create a virtual environment `ml` and install the dependencies:
~~~
 sudo pip install virtualenv
 virtualenv ml
 source ml/bin/activate
 pip -r python_venv_reqs.txt
~~~
* `model.py` - main script. You can't run this, as I haver not included my data here. You can use your own data by modifying
 the file's read_data() and setting SteeringAnglePredictor(prev_model=None).  Then, execute `python model.py`
* `steering_angle_predictor.py` - File containing the class that calls on lenet.py model to train.
* `lenet.py` - CNN model used to predict steering angles
* `utils.py` - contains a method to read data from collected data/image directories.  Also I created a method to resize images 
and save them to the file system; I was concerned with the training performance.  
* `drive.py` - provided script to drive the car.  I added a line to resize images before predicting.
* `model.h5.zip` - a trained Keras model.  I zipped it due to github's puish size limit.  The file is 150 Mb unzipped.
* `2nd_track.h5` - previous model from which I transferred learning to `model.h5`  
* `writeiup_report.md` - markdown report
* `video.mp4` - a video recording of the vehicle driving autonomously around the track one full lap

Running the project
---
* download a simulator: 
[Mac](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4594_mac-sim.app/mac-sim.app.zip) 
[Windows](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4419_windows-sim/windows-sim.zip)
[Linux](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae46bb_linux-sim/linux-sim.zip)
* Activate `ml` environment (above)
* `unzip model.h5.zip`
* Run `python drive.py model.h5`
* Start the simulator and select the 1st track.  The model runs on the 2nd track as well, but it falls off at the end of the run.

