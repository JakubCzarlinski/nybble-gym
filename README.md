# Nybble Gym
A gym reinforcement learning environment for the Nybble OpenCat robot based on Stable-Baselines3 and PyBullet, using Python 3.9.2.

## Setup
1. You should use a virtual environment (`python3 -m venv .venv`). I personally create a conda environment. 
2. `pip install -r requirements.txt ` 
3. You may need to install torch with CUDA for GPU usage. https://pytorch.org/get-started/locally/ 

## Usage
This repo comes with a pretrained model. So run the demo you can `python ./nybble_demo.py`. This will either run the visual simulation or attempt to connect to Nybble through bluetooth.

You can start training a new model for Nybble using `python ./nybble_train.py` or use the Google Colab example train-colab.ipynb. 

The `train-colab.ipynb` is not kept upto data and needs reviewing.

## TODO
Implement https://arxiv.org/pdf/1812.11103.pdf. This is basically done.

Either domain adaptation or motion tracking for reality gap.

## References
For more information on the reinforcement training implementation: https://stable-baselines3.readthedocs.io/en/master/index.html  
And for the simulation environment please refer to: https://pybullet.org/wordpress/

https://arxiv.org/pdf/1812.11103.pdf
https://xbpeng.github.io/projects/Robotic_Imitation/index.html
