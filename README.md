[//]: # (Image References)
[image_0]: ./assets/car.jpg

# Discretization using Uniform-Space Grid

![alt text][image_0]

Sometimes, we are facing with environments which have continuous state and action spaces. We'll deal with those environments by discretizing them. This will enable us to apply reinforcement learning algorithms that are only designed to work with discrete spaces. In this project, we'll use [OpenAI Gym](https://gym.openai.com/) environments to test and develop our algorithms. These simulate a variety of classic as well as contemporary reinforcement learning tasks.  

We're going to use a MountainCar environment that has a continuous state space, but a discrete action space. If you want to find more information and code implementation, check the provided notebook in this repository.

## Dependencies

You'll need Python 3, Jupyter Notebooks installed to do this project. The best way to get setup with these if you are not already is to install [Anaconda](https://www.anaconda.com). You'll also need [OpenAI Gym](https://gym.openai.com/) installation.

## Installation

Download this repository and run the following command to test the project. You'll see the agent learning to reach the goal location. The simulation environment is rendered after each 1000 episodes.

```
python main.py
```
