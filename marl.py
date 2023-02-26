
import numpy as np
from scipy.io import loadmat
import numpy as np
import networkx as nx
import glob
import string
import operator
import copy
import sys
import pandas as pd
import pickle
import dill
import itertools
from pathlib import Path


from collections import deque

#Plot Traffic
import matplotlib.pyplot as plt

from itertools import islice
from scipy.io import loadmat
import numpy as np
import math
import copy
import networkx as nx

import tensorflow as tf
from tensorflow.keras.layers import Dense,Activation
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
import pickle
import sys

from itertools import chain
import random
import time

#altri import
#from *file* import *classe/classi*
from PrioritizedReplayBuffer import PrioritizedReplayBuffer
from Agents import DDQNAgent
from statics import *
from utils_marl import *
import utils_marl


class MARL(object):
    """This object represent a Multiple Agent Reinforcement Learning.
    It is created from a list of agents"""

    def __init__(self,
                    id_node_lst, # id node of the agents
                    alpha_lst, # alphas
                    linkRete,
                    network_number_nodes, # number of the network in which the agent will be put
                    gamma_lst, # gammas
                    Graph,
                    epsilon_lst, # epsilons
                    batch_size_lst,
                    epsilon_dec_lst,
                    epsilon_end_lst,
                    fcl1_lst,
                    fcl2_lst,
                    #single_fname_qeval,
                    #single_fname_qtarget,
                    replace_target_lst,
                    #mem_size=1000000,
                    mem_size=20000,
                    scores=[],
                    eps_history=[],
                    best_score=-1000000):

        super(MARL, self).__init__()


        #list of agents created by a list comprehension; list variables are assigned using a zip function
        # list comprehension of:
        #    DQNAgent createb by passing some common values (like linkRete or batch_size) and
        #    some variables values obtained by using the zip function at the end
        self.agents = [
                    DDQNAgent(id_node = id, # fom zip
                            alpha = a, # fom zip
                            linkRete = linkRete,
                            network_number_nodes = network_number_nodes,
                            gamma = g, # fom zip
                            Graph = Graph,
                            epsilon = eps, # fom zip
                            batch_size = btch_sz,
                            epsilon_dec = eps_dec, # fom zip
                            epsilon_end = eps_end,# fom zip
                            mem_size = mem_size,
                            fname_qeval = create_agent_name('.h5',
                                                                id = id,
                                                                bs = btch_sz,
                                                                g = g,
                                                                a = a,
                                                                ms = mem_size,
                                                                rt = rplc_trgt,
                                                                fcl1 = fcl1,
                                                                fcl2 = fcl2,
                                                                netType = 'eval',
                                                                RLType = 'PER_ddqn_model'),#str(id) + '_'+'eval_PER_ddqn_model.h5',
                            fname_qtarget = create_agent_name('.h5',
                                                                id = id,
                                                                bs = btch_sz,
                                                                g = g,
                                                                a = a,
                                                                ms = mem_size,
                                                                rt = rplc_trgt,
                                                                fcl1 = fcl1,
                                                                fcl2 = fcl2,
                                                                net_type = 'target',
                                                                RL_type = 'PER_ddqn_model'), #str(id) + '_'+'target_PER_ddqn_model.h5',
                            fcl1 = fcl1,
                            fcl2 = fcl2,
                            replace_target=rplc_trgt,
                            scores = scores,
                            eps_history = eps_history,
                            best_score = best_score)

                            for id, a, g, eps, eps_dec, eps_end ,rplc_trgt, btch_sz, fcl1, fcl2 in \
                                    zip(id_node_lst, alpha_lst, gamma_lst, epsilon_lst, epsilon_dec_lst, epsilon_end_lst, replace_target_lst, batch_size_lst, fcl1_lst, fcl2_lst)
        ]



    def score_initialization(self):
        # inizialization of the score for each agent with an empty list
        for agent in self.agents:
            agent.score = []

    def get_playing_agents(self):
        '''return the list of agent that can play (in a specific episode) i.e. the agents that have:
        1) at leat one link over th
        AND
        2) the agent did no have done static.MAX_NUMBER_ACTIONS action selection (fot the specific episode)
        AND
        3) the agent have not jet "done" i.e. the agent didn't a terminating action '''
        playing_agents = []
        dones = []
        for agent in self.agents:
            # if there are some link over th  AND the number of iteration of the aget are no the maximum allowed AND the agent is not already satisfied
            if agent.number_link_over_th()>0 and \
                                agent.iteration_done<MAX_NUMBER_ACTIONS and \
                                agent.done == False:
                playing_agents.append(agent)
                dones.append(False)
        return playing_agents, dones

    def marl_caricoLink(self, R_blu, R_orange,flows_blu, flows_orange, linkRete):
        # generalization of the caricoLink function to all agents in the MARL object
        for agent in self.agents:
            agent.caricoLink(R_blu, R_orange,flows_blu, flows_orange, linkRete)

    #da eliminare
    def new_marl_caricoLink(self, R_blu, R_orange,flows_blu, flows_orange, linkRete, old):
        # generalization of the caricoLink function to all agents in the MARL object
        for agent in self.agents:
            agent.new_caricoLink(R_blu, R_orange,flows_blu, flows_orange, linkRete, old)

    def marl_orange_caricoLink(self, R, flows, linkRete):
            for agent in self.agents:
                agent.orange_caricoLink(R,flows, linkRete)

    def computeState(self, YbExp_orange, PSID_orange):
        #generalization of the computeState function
        for agent in self.agents:
            agent.computeState(YbExp_orange, PSID_orange)

    # da eliminare
    def new_computeState(self, YbExp_orange, PSID_orange, old):
        #generalization of the computeState function
        for agent in self.agents:
            agent.new_computeState(YbExp_orange, PSID_orange, old)

    def marl_number_link_over_th(self):
        # return the number of links over the THRESHOLD among all agents
        link_over_th=0
        for agent in self.agents:
            link_over_th += agent.number_link_over_th()
        return link_over_th

    def marl_reward_function(self, playing_agents, with_cycle, cammini, actions):
        all_negative = False # not all rewards are negative
        for agent in playing_agents:
            #print("cycle: ", with_cycle)
            if with_cycle: # if the agents make cycles
                agent.score.append(MAX_NEGATIVE_REWARD) # add the MAX_NEGATIVE_REWARD # -1 per distinguerlo dal peggioramento globbale in scores
                #print(agent.id_node, ' reward ', MAX_NEGATIVE_REWARD)
                agent.number_of_cycles += 1
            else:
                agent.score.append(agent.reward_function(cammini, actions[agent.id_node])[-1]) # add the agent's reward
                #print(agent.id_node, ' reward ', agent.score[-1])

        #check all negative
        all_last_rewards = [agent.score[-1] for agent in playing_agents] # obtaining a list of all last rewards
        if MAX_NEGATIVE_REWARD in all_last_rewards: #if there is at least one MAX_NEGATIVE_REWARD
            if len(set(all_last_rewards))== 1: # compute the set and check the length is equal 1 i.e. all in the list was MAX_NEGATIVE_REWARD
                all_negative=True
            else:
                print_error('!!!!!!!!!!!!!!! qualcosa non va! alcune rewards sono negative e altre no!!!!!!!!!!!!!\n' + str(all_last_rewards))

        return all_last_rewards, all_negative

    def save_epsilons(self):
        for agent in self.agents:
            agent.eps_history.append(agent.epsilon)

    def save_scores(self):
        for agent in self.agents:
            agent.scores.append(agent.score)

    def save_best_score(self):
        # generalizazion of the updating of the best score fot all agents
        for agent in self.agents:
            agent.best_score = agent.avg_score

    def marl_compute_avg_score(self, i_game_score):
        for agent in self.agents:
            agent.computeAvgScore(i_game_score)

    def marl_remember(self, playing_agents, actions, rewards):
        for agent, action, reward in zip(playing_agents, actions.values(), rewards):
            agent.remember(action, reward)

    def marl_learn(self, playing_agents):
        for agent in playing_agents:
            agent.learn()

    def marl_update_iterations(self, playing_agents):
        for agent in playing_agents:
            agent.iteration_done+=1

    def reset_iterations(self):
        for agent in self.agents:
            agent.iteration_done = 0
            agent.done=False

    def save(self, folder):
        # saving of the agent object
        for agent in self.agents:
            agent.save(folder)

    def test_resuming(self, marl, save_folder):
        check, marl, i_game_score = resume(marl, save_folder + BEST_FOLDER_NAME) # tring to resuming

        if not check: # if resuming not possible
            print_error('Resuming for test not possible\nSee the forlder: ' + str(save_folder + BEST_FOLDER_NAME))
            exit()
        else: # it's possible to resume
            for agent in marl.agents: # for each agent ...
                agent.epsilon = 0 # set the epsilon to 0 (for test phase)

            return marl
