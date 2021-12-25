# HierNet-SC2

The codes, data, models, and logs for the paper "On Reinforcement Learning for Full-length Game of StarCraft".

## Codes

The below table shows the scripts and codes in the project. We provide codes for the main training of a strong agent, playing to generate replays, and extracting in replays to generate macros. 

Content | Description
------------ | -------------
./main.py | the script for the main training code
./mine_from_replay.py | the script for extracting from replays to generate macro-actions
./play_for_replay.py | the script to play the game for experts to generate replays 
./multi_agent.py | the codes for the agent in the RL training
./new_network.py | the codes for the neural network model in the agent
./param.py | the hyper-parameters in the neural network
./algo/ | the algorithms like ppo
./lib/ | the library functions


## Data

The below table shows the data for training and generating content.

Content | Description
------------ | -------------
./data/replay/ | 30 replays played by the expert for generating macro actions
./data/replay_data/ | using data mining algorithms to mine macro actions from the above replays

## Logs

The below table shows the logs for training on difficult level-8, level-9, and level-10.

Content | Description
------------ | -------------
./logs/lv10-0.20_to_0.90/ | the training log in lv-10 which takes the win-rate from 0.20 to 0.90
./logs/lv10-0.90_to_0.94/ | the training log in lv-10 which takes the  win-rate from 0.90 to 0.94
./logs/lv8-0_to_0.960/ | the training log in lv-8 which takes the  win-rate to 0.960, restore=lv10-0.20_to_0.90
./logs/lv9-0_to_0.967/ | the training log in lv-9 which takes the  win-rate to 0.967, restore=lv10-0.20_to_0.90

## Model

We provide the trained 0.94 win-rate model in lv-10 as below.

Content | Description
------------ | -------------
./model/lv10-0.94/ | the model which gets win-rate 0.94 in lv-10

### Requirements

- python==3.5
- tensorflow==1.5
- future==0.18.2
- pysc2==1.2
- matplotlib==3.3.4
- scipy==1.1.0
- prefixspan==0.5.2

### Install

A simple straightway is, you can first use conda like:
```
conda create -n tf_1_5 python=3.5 tensorflow-gpu=1.5
```
to install Tensorflow-gpu 1.5 (with accompanied CUDA and cudnn).

Next, you should activate the conda environment, like:
```
conda activate tf_1_5
```

Then you can install other python packages by pip, e.g., the command is:
```
pip install -r requirements.txt
```

### Usage

Run main.py to train an agent (P vs. T) against the built-in bot (lv-10) in StarCraft II. 