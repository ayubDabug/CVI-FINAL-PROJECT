# CVI-FINAL-PROJECT
The core idea: Feed the model images from a car's front camera, and have it output the correct steering angle to keep the car on the road.

How the project flows:

Data collection – Use the Udacity self-driving car simulator in "Training Mode." You manually drive the car around a track (both forward and backward directions, ~5 laps each way) while it records center/left/right camera images plus steering angle, throttle, brake, and speed into a driving_log.csv file.
Balance the dataset – Plot a histogram of steering angles to check the data isn't overly skewed (e.g., too many "drive straight" samples).
Data augmentation – Artificially diversify the images via flipping (remember to negate the steering angle when you flip), brightness changes, zooming, panning, rotation — applied randomly to only part of the data, and only on the training set.
Preprocessing – Crop out the sky/hood from images, convert to YUV color space, resize to 200×66 (matching Nvidia's model input), plus normalization and optional Gaussian blur.
Batching – Write a function to feed data to the model in batches.
Build & train the CNN – Follow the Nvidia architecture shown in the doc: 5 convolutional layers → flatten → fully-connected layers (1164 → 100 → 50 → 10) → single output (steering angle). Plot training curves to evaluate performance and save the trained model.
Test it – Run a TestSimulation.py script that feeds the model live images from the simulator (in "Autonomous Mode") and watch whether the car stays on the track.

Deliverables: all your Python scripts (preprocessing, training, inference, augmentation), a screen recording of the model driving successfully, a Git repo with commit history, and documentation/setup instructions. Groups of up to 3 are allowed.
