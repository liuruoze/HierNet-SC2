# HierNet-SC2

Codes, data, models, and the associated logs for the paper "On Reinforcement Learning for Full-length Game of StarCraft".

## Codes

The below table shows the scripts and codes in the project. We provide codes for the main training of a strong agent, playing to generate replays, and extracting in replays to generate macros. 

Content | Description
------------ | -------------
./main.py | script for the main training code
./mine_from_replay.py | script for extracting from replays to generate macro-actions
./multi_agent.py | the codes for the agent in the RL training
./new_network.py | the codes for the neural network model in the agent
./param.py | hyper-parameters in the neural network
./play_for_replay.py | script to play the game for experts to generate replays 
./algo/ | used algorithms like ppo
./lib/ | used library functions


## Data

The below table shows the data for training and generating content.

Content | Description
------------ | -------------
./data/replay/ | 30 replays played by the expert for generating macro actions
./data/replay_data/ | using data mining algorithms to mine macro actions from the above replays

## Logs

The below table shows the logs for training for difficult level-8, level-9, and level-10.

Content | Description
------------ | -------------
./logs/lv10-0.20_to_0.90/ | trainging in lv-10 win-rate from 0.20 to 0.90, lr=1e-4
./logs/lv10-0.90_to_0.94/ | trainging in lv-10 win-rate from 0.90 to 0.94, lr=1e-5
./logs/lv8-0_to_0.960/ | trainging in lv-10 win-rate to 0.960, lr=1e-5, restore=lv10-0.20_to_0.90
./logs/lv9-0_to_0.967/ | trainging in lv-10 win-rate to 0.967, lr=1e-5, restore=lv10-0.20_to_0.90

## Model

We provide the 0.94 win-rate model trained in lv-10 as below.

Content | Description
------------ | -------------
./model/lv10-0.94/ | trainging in lv-10 win-rate, final win rate is 0.94

### Requirements

- python==3.5
- tensorflow==1.5
- future==0.16
- pysc2==1.2
- matplotlib==2.1
- scipy==1.0

**Notes:**

If you install pysc2==1.2 and find this error "futures requires Python '>=2.6, <3' but the running Python is 3.5.6", then try first install futures as follow
```
pip install futures==3.1.1
```
then install pysc2==1.2, and this problem is solved.

### Usage

Run main.py to train an agent (P vs. T) against the built-in bot (lv-10) in StarCraft II. 