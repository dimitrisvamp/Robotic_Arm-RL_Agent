# Robotic_Arm-Machine_Learning_Agent

## University Diploma Thesis
My diploma thesis for Computer Science and Engineering department of the University of Ioannina (UOI).

## Description
Imitation learning of an intelligent agent for object approach by robotic arm.

The imitation learning algorithm implemented for training is Behavioral Cloning from Observation (BCO).

The simulation of training and test procedure take place at mujoco-py (https://openai.github.io/mujoco-py/build/html/index.html) wrapper-simulator.

Robomimic framework (https://robomimic.github.io/docs/datasets/robosuite.html) provides a proficient-human dataset for robosuite enviroment. The enviroment used for this project
is ***"Lift"*** using low dimensional observation. 

Neural Networks implemented by using Tensorflow framework (https://www.tensorflow.org/).

## Requirements
Special requirements:
1) Install mujoco-py wrapper.
2) Install robosuite.
3) Install robomimic.
4) Install low dimensional observation robosuite dataset from robomimic.
5) Install Tensorflow.

## Uploaded Files
The ***data*** folder contains all data that train and test procedures need, after data management, in form of txt files.

There is also ***src*** folder that contains five python files:
1) ***BCO.py*** (BCO implementation for agent's training and testing).
2) ***extract_data_from_cube_pos.py*** (extracting data according to cube's position, left-center-right side position).
3) ***extract_demos_from_hdf5_to_txt.py*** (extracting data from robomimic to txt files).
4) ***mma.py*** (calculating means and std for each dimension, required for z-score normalization).
5) ***utils.py*** (input parsing and shuffling function).

## How to run
For training mode run the following command at cmd: <br />
    &emsp; &emsp; &emsp; &emsp; ***BCO.py --state_dataset=state_cube_right.txt --nstate_dataset=nstate_cube_right.txt --action_dataset=action_cube_right.txt 
    --mode=train --trained_model_dir=trained_model_right-64/ --max_episodes=1000 --print_freq=50***

For testing mode run the following command at cmd: <br />
    &emsp; &emsp; &emsp; &emsp; ***BCO.py --state_dataset=state_cube_right.txt --nstate_dataset=nstate_cube_right.txt --action_dataset=action_cube_right.txt 
    --mode=test --trained_model_dir=trained_model_right-64/ --max_episodes=1000 --print_freq=50***

## Special Notes
For better agent's performance, it is suggested to train an agent for each cube's side position (left-side-agent, center-side-agent, right-side-agent).
