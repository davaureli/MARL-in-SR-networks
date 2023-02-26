# MARL-in-SR-networks (SR-iLLC)

We propose a framework based on __Deep Reinforcement Learning__ to proactively and autonomously take under control links loads in __Segment Routing (SR) networks__.
The main idea is to monitor local link loads and, in case of anomalous situation, to execute local routing changes at milliseconds timescale.

The solution proposed is based on a __Multi Agent Reinforcement Learning__ (MARL) approach: a subset of nodes is equipped with a local agent, powered
by a __Deep Q-Network__ (DQN) algorithm, referred to as __SRv6__ rerouting for Local In-network Link Load Control (SR-LILLC).

