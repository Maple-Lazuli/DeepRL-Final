# Solving Tennis with Deep Reinforcement Learning

![image-20220527123215957](images/image-20220527123215957.png)

## Project Details 

This repository contains python code that implements _deep reinforcement learning_ to solve the problem of maintaining a volley with two tennis rackets.

### Environment Details

The environment consists of 2 agents, one for each racket, to hit a tennis ball. For each agent, the environment keeps track of 8 variables provide data for:

1. Tennis Ball Velocity
1. Racket Velocity

Additionally, the action space for each agent consists of 2 movements, jumping and horizontal movement.

Finally, the agents are rewarded with a +0.1 every time it hits the ball over the net and received a -0.01 if they let the ball hit the ground or go out of bounds.

## Getting Started

Follow the steps below to setup an environment to run the project.

Start by creating a new environment with python 3.6 (**Note:** python 3.6 is necessary for tensorflow 1.7.1 and the unity environment will not work without this version.)

```bash
conda create --name drlnd python=3.6
source activate drlnd
```

Next, clone the __deep-reinforcement-learning__ repository to install the _unity agent_ dependency.

```bash
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```

Then, use _pip_ to install the requirements detailed in the requirements text file.

```bash
pip install -r requirments.txt
```

Finally, source the conda environment with

```bash
conda activate drlnd
```

## Execution Instructions

The agents can be trained by executing __train_agents.py__ with the command from the src directory:

```python
python train_agents.py
```

Running train_agents.py will start up the unity environment and begin training the model.

![image-20220527123015082](images/image-20220527123015082.png)

Confirmation of a successful start is seeing the print out "Starting Epoch 0".
