# MARL-in-SR-networks (SR-iLLC)

We propose a framework based on __Deep Reinforcement Learning__ to proactively and autonomously take under control links loads in __Segment Routing (SR) networks__.
The main idea is to monitor local link loads and, in case of anomalous situation, to execute local routing changes at milliseconds timescale.

The solution proposed is based on a __Multi Agent Reinforcement Learning__ (MARL) approach: a subset of nodes is equipped with a local agent, powered
by a __Deep Q-Network__ (DQN) algorithm, referred to as __SRv6__ rerouting for Local In-network Link Load Control (SR-LILLC).

To know all the arguments in the code:

```
python main_marl.py --help
```
It returns the following list:

usage: main_marl.py [-h] [--netw NETW] [--agents_file AGENTS_FILE]
                    [--train_flag TRAIN_FLAG] [--save_folder SAVE_FOLDER]
                    [--n_act N_ACT]

optional arguments:
  -h, --help            show this help message and exit
  --netw NETW           network you want: [nobel, germany]
  --agents_file AGENTS_FILE
                        csv containing the agents you want to introduce
  --train_flag TRAIN_FLAG
                        is you want train; put false for testing
  --save_folder SAVE_FOLDER
                        folder in which you eant to save your execution
  --n_act N_ACT         number of action for each agent
```

To run the code, according to the specific example provided in this folder type:

- Train

```
python main_marl.py --agents_file aaa.csv --train_flag train --save_folder training_agents --n_act 1
```

- Test

```
python main_marl.py --agents_file aaa.csv --train_flag test --save_folder training_agents --n_act 1
```


