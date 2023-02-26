# MARL-in-SR-networks (SR-iLLC)

We propose a framework based on Deep Reinforcement Learning to proactively and autonomously
take under control links loads in Segment Routing (SR) networks. 

The main idea is to monitor local link loads and to execute local routing changes if the local utilization exceeds a specific threshold.

The solution proposed is based on a Multi Agent Reinforcement Learning (MARL) approach: a subset of nodes is equipped with a local agent, powered
by a Deep Q-Network (DQN) algorithm, referred to as SRv6 rerouting for Local In-network Link Load Control (SR-LILLC).

