# HierNet-SC2

This repository contains the codes, models, training logs, and data for an extended paper of the original conference paper on AAAI 2019, "On Reinforcement Learning for Full-length Game of StarCraft".

The extended journal paper is named "On Efficient Reinforcement Learning for Full-length Game of StarCraft II", and the arxiv link is [here](https://arxiv.org/abs/2209.11553).

## Codes

The below table shows the codes in the project. We provide codes for the training of agents, generating replays, and mining macro actions. 

Content | Description
------------ | -------------
./main.py | the codes for the training code
./mine_from_replay.py | the codes for mining macro-actions
./play_for_replay.py | the codes for generating replays 
./multi_agent.py | the codes for the agent
./new_network.py | the codes for the neural network model
./param.py | the hyper-parameters
./algo/ | the algorithm of ppo
./lib/ | the library functions


## Data

The below table shows the data for training agent and generating macro actions.

Content | Description
------------ | -------------
./data/replay/ | 30 replays played by the expert for generating macro actions
./data/replay_data/ | using data mining algorithms to mine macro actions from the above replays

## Logs

The below table shows the logs of training on cheating level built-in AIs.

Content | Description
------------ | -------------
./logs/lv10-0.20_to_0.90/ | the training log in lv-10 which takes the win-rate from 0.20 to 0.90
./logs/lv10-0.90_to_0.94/ | the training log in lv-10 which takes the  win-rate from 0.90 to 0.94
./logs/lv8-0_to_0.960/ | the training log in lv-8 which takes the  win-rate to 0.960, restore=lv10-0.20_to_0.90
./logs/lv9-0_to_0.967/ | the training log in lv-9 which takes the  win-rate to 0.967, restore=lv10-0.20_to_0.90

## Model

The below table shows the model of a 0.94 win rate in lv-10.

Content | Description
------------ | -------------
./model/lv10-0.94/ | the model which gets win-rate 0.94 in lv-10

## Requirements

- python==3.5
- tensorflow==1.5
- future==0.18.2
- pysc2==1.2
- matplotlib==3.3.4
- scipy==1.1.0
- prefixspan==0.5.2

## Install

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

## Usage

Run main.py to train an agent against the most difficult built-in bot (lv-10) in StarCraft II. 

Run mine_from_replay.py to mine macro actions from replays. 

Run play_for_replay.py to generate replays by your selves. 