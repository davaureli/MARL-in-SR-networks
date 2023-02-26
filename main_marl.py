import numpy as np
from scipy.io import loadmat
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
import math
import copy

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #togliere, messo per evitare conflitti con l'utilizzo della GPU

import tensorflow as tf
from tensorflow.keras.layers import Dense,Activation
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt

from itertools import chain
import random
import time

#altri import
#from *file* import *classe/classi*
from PrioritizedReplayBuffer import PrioritizedReplayBuffer
from Agents import DDQNAgent
from statics import *
from utils_marl import *
from utils_heuristic import *
import utils_marl
import utils_heuristic
from marl import MARL
import marl
import argparse


if __name__ == "__main__":

    #input from terminal
    parser = argparse.ArgumentParser()
    #adding options to the parser
    parser.add_argument('--netw', help='network you want: [nobel, germany]') # nobel or germany
    parser.add_argument('--agents_file', help='csv containing the agents you want to introduce')
    parser.add_argument('--train_flag', help='is you want train; put false for testing')
    parser.add_argument('--save_folder', help='folder in which you eant to save your execution')
    parser.add_argument('--n_act', help = 'number of action for each agent')
    # parsing
    args = parser.parse_args()
    # assigning to variables
    NETWORK_NAME = args.netw
    agents_file = args.agents_file
    train_flag = 'train' if args.train_flag == 'train' else 'test'
    print('train flag: ', train_flag)
    # if there is some input we can use it as a folder in which we can save;
    # otherwise we set this as an empty string in order to obtain the default saving directory
    save_folder = args.save_folder + '/' if args.save_folder != None else ''
    
    #We can set the MAX NUMBER of Actions for each episode
    tmp_numberAction = int(args.n_act)
    MAX_NUMBER_ACTIONS = tmp_numberAction
    marl.MAX_NUMBER_ACTIONS = MAX_NUMBER_ACTIONS


    #initial setup
    #    network choosen
    network_number_nodes=-1
    net_folder=''

    if NETWORK_NAME=='nobel':
        network_number_nodes=NETWORK_NUMBER_NODES_NOBEL
        net_folder='nobel/'
    elif NETWORK_NAME=='germany':
        network_number_nodes=NETWORK_NUMBER_NODES_GERMANY
        net_folder='germany/'
    else:
        print('I can\'t recognize your network... ;) ' )
        exit()

    base = './'+net_folder + 'input/' # base from wich we can take the input files
    save_folder = './'+net_folder + 'output/' + save_folder # base folder for saving

    annots = loadmat(base + NETWORK_NAME + '_A.mat')
    A = annots["A"]
    Graph = nx.from_numpy_matrix(A)

    #   agents_file option
    agents_dict = get_agents_dict(agents_file) # obtain the dictionary of the agents
    
    print("Agents Dict: ", agents_dict)


    #print_warning('MONO AGENT! BUTTTTTT we set n actions to 1')

    # setup some static variables in the other libraries
    utils_marl.base = base
    marl.base = base
    utils_marl.network_number_nodes = network_number_nodes
    utils_heuristic.network_number_nodes = network_number_nodes
    utils_marl.NETWORK_NAME = NETWORK_NAME
    utils_marl.A = A
    utils_marl.Graph = Graph
    utils_heuristic.Graph = Graph

    linkRete = customize_bandwidth(agents_dict['id_node_lst']) # customize the bandwidth for the specific network

    #reading the traffic matrices for both granuarities
    diz_trafficMatrix_orange = pd.read_pickle(base + "matrici_arancioni.pkl")
    diz_trafficMatrix_blu = pd.read_pickle(base + "matrici_blu.pkl")

    if train_flag == 'train': #train

        #os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #togliere, messo per evitare conflitti con l'utilizzo della GPU

        index_train = pd.read_pickle(base + "index_trainNEW.pkl")
        print("WE ARE TRAINING: \n")
        print("Train: " + str(len(index_train)) )
        
        #Increase Number of matrices during the training
        index_train = index_train*1000

        print("Tot. Train Matrices: ",len(index_train))
        utils_marl.lenIndexTrain=len(index_train)

        #Re-shuffle train
        random.seed(42) # set the seed in order to have always the same shuffled dataset
        # and in oder to be able to restart the execution after a crash.
        random.shuffle(index_train)

        matrices_not_used = 0
        episode_saved_model = 0
        
        print("Create MARL Object")

        #create the marl object
        marl = MARL(id_node_lst = agents_dict['id_node_lst'],
                    alpha_lst = agents_dict['alpha_lst'],
                    linkRete = linkRete,
                    network_number_nodes = network_number_nodes,
                    gamma_lst = agents_dict['gamma_lst'],
                    Graph = Graph,
                    epsilon_lst = agents_dict['epsilon_lst'],
                    epsilon_dec_lst = agents_dict['epsilon_dec_lst'],
                    epsilon_end_lst = agents_dict['epsilon_end_lst'],
                    batch_size_lst = agents_dict ['batch_size_lst'],
                    replace_target_lst =  agents_dict['replace_target_lst'],
                    fcl1_lst = agents_dict['fcl1_lst'],
                    fcl2_lst = agents_dict['fcl2_lst'])
        
        print("Created MARL Object")
        
        check, marl_t, i_game_score_t = resume(marl, save_folder) # try to resume
        
        
        
        if check: # if it's possible to resume we can use the resumed files
          marl = marl_t
        i_game_score=i_game_score_t # in all cases i_game_score is returned
                                    # -1 if it's not possible to resume
                                    # the correct i_Game_score if is possible to resume


        #print("Davide check:")
        #for a in marl.agents:
        #    print(a.memory.buffer.maxlen,a.memory.mem_cntr)
        #    print()
        #input()


        print('Start iterating!')
        with open(save_folder + 'log.txt', 'a') as target:
            print('Start iterating!', file=target)
            
        startingTime=time.time()
        utils_marl.startingTime = startingTime


        for i_game_score in range(i_game_score+1, len(index_train)):
            i_game=index_train[i_game_score]

            TM_blu = diz_trafficMatrix_blu[i_game]
            TM_orange = diz_trafficMatrix_orange[i_game]

            used_matrix = False

            np.fill_diagonal(TM_blu, 0)
            np.fill_diagonal(TM_orange, 0)


            marl.score_initialization() # initialize the score of each agent to an empty string

            flows_blu, R_blu,SL_blu, YbExp_blu, PSID_blu, cammini_blu, L_blu,K_blu  = compute_allParameters(TM_blu,linkRete)
            flows_orange, R_orange,SL_orange, YbExp_orange, PSID_orange, cammini_orange,L_orange,K_orange = compute_allParameters(TM_orange,linkRete)

            #Commentare
            #for agent in marl.agents:
            #    print_warning(str(agent.id_node) + '  new_local_out_prc_usage_orange  ' + str(agent.new_starting_local_out_prc_usage_orange))
            #Commentare

            marl.marl_caricoLink(R_blu, R_orange,flows_blu, flows_orange, linkRete)

            #marl.new_marl_caricoLink(R_blu, R_orange,flows_blu, flows_orange, linkRete, True)


            marl.computeState(YbExp_orange, PSID_orange)
            #marl.new_computeState(YbExp_orange, PSID_orange, True)
            
            #Commentare
            #for agent in marl.agents:
            #    print_warning(str(agent.id_node) + '  new_local_out_prc_usage_orange  ' + str(agent.new_starting_local_out_prc_usage_orange))
            #Commentare

            there_agents_over_100 = check_agent_over_100(marl)
            #print('i_game_score: ', i_game_score)
            if there_agents_over_100:
                matrices_not_used +=1
                print('matrici not used: ', matrices_not_used)
                break # --> completely stop the simulation cause there is an utilization over 100%
                #continue

            marl.reset_iterations() # set the number of iteration done to 0 for each agent, and done flag to False

            playing_agents, dones = marl.get_playing_agents() # va cambiata link sopra th e numbero di iter <3 per ogni agente
            #dones = [agent.done for agent in playing_agents]

            if len(playing_agents)>0:
                used_matrix = True

            # abbiamo creato get_playing_agents per ottenere gli agenti che hanno almeno un link sopra th e con gli altri cosa facciamo?
            # dobbiamo considerarli?
            dict_encap_blu_dict = {}
            dict_encap_orange_dict = {}

            for agent in marl.agents:
                dict_encap_blu_dict[agent.id_node] = {}
                dict_encap_orange_dict[agent.id_node] = {}

            while len(playing_agents) > 0: # if there is at least one link over TH among all agents
                #while not dones and check_iteration<3:

                actions={agent.id_node : agent.choose_action() for agent in playing_agents}

                map_allPossible_Actions_dict = {agent.id_node : agent.map_allPossible_Actions[actions[agent.id_node]] for agent in playing_agents}
                
                #Commentare
                #print('actions:\n', actions)
                #print('map_allPossible_Actions_dict:\n', map_allPossible_Actions_dict)
                #print()
                #print("Dict Orange")
                #print(dict_encap_orange_dict)
                #print()
                #Commentare

                # BLU -> delta grande
                #dict_encap_blu_dict = multi_fill_Encap(dict_encap_blu_dict, SL_blu, cammini_blu, map_allPossible_Actions_dict)
                #print('dict_encap_blu_dict', dict_encap_blu_dict)
                #B,Yb_blu, PSID_blu, R_blu, with_cycle = multi_aggiornaPSID(SL_blu, cammini_blu, flows_blu, network_number_nodes, linkRete, dict_encap_blu_dict)
                
                
                
                # ORANGE
                dict_encap_orange_dict = multi_fill_Encap(dict_encap_orange_dict, SL_orange, cammini_orange, map_allPossible_Actions_dict)
                #print('dict_encap_orange_dict', dict_encap_orange_dict)
                B, Yb_orange, PSID_orange, R_orange, with_cycle = multi_aggiornaPSID(SL_orange, cammini_orange, flows_orange, network_number_nodes, linkRete, dict_encap_orange_dict)
                
                
                #Commentare
                #print("Flows, ", flows_orange[(flows_orange[:,1]==0) & (flows_orange[:,2]==map_allPossible_Actions_dict[0][-1])])
                #id_f = flows_orange[(flows_orange[:,1]==0) & (flows_orange[:,2]==map_allPossible_Actions_dict[0][-1])][0][0]
                #print("ID Flows, ", flows_orange[(flows_orange[:,1]==0) & (flows_orange[:,2]==map_allPossible_Actions_dict[0][-1])][0][0])
                #print("R_orange",R_orange[:,id_f])
                #print("Link Rete: ", linkRete[R_orange[:,id_f]>0,:])
                #print("shape", R_orange.shape)
                #print()
                #print("B",B)
                #print("SL orange",SL_orange)
                #print(len(cammini_orange))
                #Commentare
                
                #marl.marl_caricoLink(R_blu, R_orange,flows_blu, flows_orange, linkRete)
                marl.marl_orange_caricoLink(R_orange, flows_orange, linkRete)


                marl.computeState(Yb_orange, PSID_orange)
                #marl.new_computeState(YbExp_orange, PSID_orange, False)

                rewards, all_negative_bool = marl.marl_reward_function(playing_agents, with_cycle, cammini_orange, actions)
                
                #Commentare
                #print("REWARD: ", rewards)
                #print("Flag Bool: ", all_negative_bool)
                #print()
                #Commentare
                
                #aggiorniamo il numero di mosse prima non dopo il remember
                marl.marl_update_iterations(playing_agents)

                
                # va rivista: se un agente ha abbassato i link e gli altri no?
                if all_negative_bool:
                    for agent in playing_agents:
                        agent.done=True
                    #dones = [True]*len(playing_agents)
                else:
                    for agent in playing_agents:

                        if agent.number_link_over_th() == 0:
                            agent.done = True #
                            
                        if agent.iteration_done==MAX_NUMBER_ACTIONS:

                            agent.done = True
                        
                        #else:
                        #    False

                    #agent.number_link_over_th() == 0  under the ipothesys that we have MAX_NUMBER_ACTIONS = 1
                    # else False in case we have many actions...

                if all_negative_bool and i_game_score > 50000:
                    print_error('!!!!!!!!!!!!!!!!!!!! Global Error !!!!!!!!!!!!!!!!!!!!!!!!!!!')

                marl.marl_remember(playing_agents, actions, rewards)

                marl.marl_learn(playing_agents)

                #marl.marl_update_iterations(playing_agents)

                playing_agents, dones = marl.get_playing_agents()

                #print('all_last_rewards:\n', rewards)
                
                #Commentare
                #for agent in marl.agents:
                #    print_warning(str(agent.id_node) + '  new_local_out_prc_usage_orange  ' + str(agent.new_starting_local_out_prc_usage_orange))
                #Commentare

            
            marl.save_epsilons()
            marl.save_scores()


            marl.marl_compute_avg_score(i_game_score)

            if i_game_score>30000:

                flag_save = 0

                for agent in marl.agents:
                    if agent.avg_score > agent.best_score and agent.avg_score>0:

                        flag_save += 1

                if flag_save == len(marl.agents):

                    agent.best_score = agent.avg_score
                    saveAll(marl, i_game_score, startingTime, save_folder + BEST_FOLDER_NAME)
                    episode_saved_model = i_game_score

            if i_game_score % 10000 == 0:
                saveAll(marl, i_game_score, startingTime, save_folder)

                print("Last Saving Model Best Results in Episode ", episode_saved_model)
                for agent in marl.agents:
                    print("Best Results: " + str(agent.best_score))
                #print("Matrici non utilizzate: " + str(matrices_not_used))
                #print('i_game_score', i_game_score, '\n\n\n Fine episodio \n\n\n')

            if i_game_score % 500 == 0:
                print("Siamo arrivati a 500")

            if not used_matrix:
                matrices_not_used += 1

            i_game_score += 1

            #input("Fine Episodio")
            #print("\n\n\n")

            #if i_game_score%5000==0:
            #    print(i_game_score)

                #for agent in marl.agents:
                #    print("Agent: ", agent.id_node, "score: %.2f" % np.mean(agent.score),"best score: %.2f" % agent.best_score, "average_score %.2f" % agent.avg_score, "Epsilon: %.2f" %agent.epsilon)
                #    print("Last 20 scores: ", agent.scores[-20:])
                #    print('Time elapsed: ', str(dt.timedelta(seconds = time.time() - startingTime)))



            with open(save_folder + 'log.txt', 'a') as target:

                print("Last Saving Model Best Results in Episode ", episode_saved_model, file=target)
                for agent in marl.agents:
                    print("Best Results: " + str(agent.best_score), file=target)
                print("Matrici non utilizzate: " + str(matrices_not_used), file=target)
                print('i_game_score', i_game_score, '\n\n\n Fine episodio \n\n\n', file=target)


    elif train_flag == 'test': # test fase

        os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #togliere, messo per evitare conflitti con l'utilizzo della GPU

        index_test = pd.read_pickle(base + "index_testNEW.pkl")

        print("Test: " + str(len(index_test)) )


        print("Tot. Test Matrices: ",len(index_test))
        utils_marl.lenIndexTest=len(index_test)

        matrices_not_used = 0
        episode_saved_model = 0

        #create the marl object
        marl = MARL(id_node_lst = agents_dict['id_node_lst'],
                    alpha_lst = agents_dict['alpha_lst'],
                    linkRete = linkRete,
                    network_number_nodes = network_number_nodes,
                    gamma_lst = agents_dict['gamma_lst'],
                    Graph = Graph,
                    epsilon_lst = [0 for _ in range(len(agents_dict['epsilon_lst']))], # epsilon must be 0 in test
                    epsilon_dec_lst = agents_dict['epsilon_dec_lst'],
                    epsilon_end_lst = agents_dict['epsilon_end_lst'],
                    batch_size_lst = agents_dict ['batch_size_lst'],
                    replace_target_lst =  agents_dict['replace_target_lst'],
                    fcl1_lst = agents_dict['fcl1_lst'],
                    fcl2_lst = agents_dict['fcl2_lst'])

        # reload the best networks

        marl = marl.test_resuming(marl, save_folder)

        print('Start iterating!')
        startingTime=time.time()
        utils_marl.startingTime = startingTime

        result_test_dict = {i_game_score : {} for i_game_score in index_test}

        result_test_Antonio = {i_game_score:{} for i_game_score in index_test}

        for i_game_score in range(len(index_test)):
        #for i_game_score in range(1):
            #i_game_score = 257
            i_game=index_test[i_game_score]
            print('i game: ', i_game, ' i_game score: ', i_game_score)

            TM_blu = diz_trafficMatrix_blu[i_game]
            TM_orange = diz_trafficMatrix_orange[i_game]

            used_matrix = False

            np.fill_diagonal(TM_blu, 0)
            np.fill_diagonal(TM_orange, 0)


            marl.score_initialization() # iniialize the score of each agent to an ampty sting

            flows_blu, R_blu,SL_blu, YbExp_blu, PSID_blu, cammini_blu, L_blu,K_blu  = compute_allParameters(TM_blu,linkRete)
            flows_orange, R_orange,SL_orange, YbExp_orange, PSID_orange, cammini_orange,L_orange,K_orange = compute_allParameters(TM_orange,linkRete)

            #PER ANTONIO AGGIUNTO ORA 02/03
            for R, flows in zip([R_orange], [flows_orange]):
                # zip:-> triples of r, flow and color

                request = R.dot(flows[:,-1])
                percentage_usage = request/ linkRete[:,-1]
                print(percentage_usage.shape)
                result_test_Antonio[i_game] = [percentage_usage]
                print("Appeso inizio")
            #PER ANTONIO AGGIUNTO ORA 02/03
            #input()


            '''
            #Cancellare METRICHE:
            for agent in marl.agents:
                print_warning(str(agent.id_node) + '  new_local_out_prc_usage_orange  ' + str(agent.new_starting_local_out_prc_usage_orange))
                print("MAX ORANGE: ",max(agent.new_starting_global_percentage_usage_orange))
                print("Link Max Utilization: ",np.argmax(agent.new_starting_global_percentage_usage_orange),linkRete[np.argmax(agent.new_starting_global_percentage_usage_orange)])
                print()
            '''

            #print_warning('primo carico link')
            marl.marl_caricoLink(R_blu, R_orange,flows_blu, flows_orange, linkRete)

            marl.computeState(YbExp_orange, PSID_orange)

            check_agent_over_100(marl)

            marl.reset_iterations() # set the number of iteration done to 0 for each agent

            playing_agents, dones = marl.get_playing_agents() # va cambiata link sopra th e numbero di iter <3 per ogni agente
            #dones = [agent.done for agent in playing_agents]

            if len(playing_agents)>0:
                used_matrix = True

            #salvataggio valori iniziali
            for agent in marl.agents:
                if agent not in playing_agents: # non sta giocando: scrivi empty
                    result_test_dict[i_game][agent.id_node] = ['empty']
                else: # sta giocando: salva i valori iniziali:
                    result_test_dict[i_game][agent.id_node] = [agent.new_starting_local_out_prc_usage_orange, np.max(agent.new_starting_global_percentage_usage_orange)]


            # abbiamo creato get_playing_agents per ottenere gli agenti che hanno almeno un link sopra th e con gli altri cosa facciamo?
            # dobbiamo considerarli?
            dict_encap_blu_dict = {}
            dict_encap_orange_dict = {}

            for agent in marl.agents:
                dict_encap_blu_dict[agent.id_node] = {}
                dict_encap_orange_dict[agent.id_node] = {}

            '''
            #Cancellare METRICHE:
            for agent in marl.agents:
                print_warning(str(agent.id_node) + '  new_local_out_prc_usage_orange  ' + str(agent.new_starting_local_out_prc_usage_orange))
                print("MAX ORANGE: ",max(agent.new_starting_global_percentage_usage_orange))
                print("Link Max Utilization: ",np.argmax(agent.new_starting_global_percentage_usage_orange),linkRete[np.argmax(agent.new_starting_global_percentage_usage_orange)])
                print()
            '''
            while len(playing_agents) > 0: # if there is at least one link over TH among all agents
                #while not dones and check_iteration<3:
                #print('in while')
                actions={agent.id_node : agent.choose_action() for agent in playing_agents}



                map_allPossible_Actions_dict = {agent.id_node : agent.map_allPossible_Actions[actions[agent.id_node]] for agent in playing_agents}

                
                print('actions:\n', actions)
                print('map_allPossible_Actions_dict:\n', map_allPossible_Actions_dict)
                print('map all possible action:\n', agent.map_allPossible_Actions[actions[agent.id_node]])
                
                # BLU -> delta grande
                #dict_encap_blu_dict = multi_fill_Encap(dict_encap_blu_dict, SL_blu, cammini_blu, map_allPossible_Actions_dict)
                #print('dict_encap_blu_dict', dict_encap_blu_dict)
                #B,Yb_blu, PSID_blu, R_blu, with_cycle = multi_aggiornaPSID(SL_blu, cammini_blu, flows_blu, network_number_nodes, linkRete, dict_encap_blu_dict)

                # ORANGE
                dict_encap_orange_dict = multi_fill_Encap(dict_encap_orange_dict, SL_orange, cammini_orange, map_allPossible_Actions_dict)
                #print('dict_encap_orange_dict', dict_encap_orange_dict)
                B, Yb_orange, PSID_orange, R_orange, with_cycle = multi_aggiornaPSID(SL_orange, cammini_orange, flows_orange, network_number_nodes, linkRete, dict_encap_orange_dict)

                print_warning('secondo carico link')
                #marl.marl_caricoLink(R_blu, R_orange,flows_blu, flows_orange, linkRete)
                marl.marl_orange_caricoLink(R_orange, flows_orange, linkRete)


                marl.computeState(Yb_orange, PSID_orange)


                rewards, all_negative_bool = marl.marl_reward_function(playing_agents, with_cycle, cammini_orange, actions)

                '''
                print('rewards: \n', rewards, "CICLI: ", with_cycle)

                ###########################
                #Cancellare METRICHE:
                for agent in marl.agents:
                    print_warning(str(agent.id_node) + '  new_local_out_prc_usage_orange  ' + str(agent.new_starting_local_out_prc_usage_orange))
                    print("MAX ORANGE: ",max(agent.new_starting_global_percentage_usage_orange))
                    print("Link Max Utilization: ",np.argmax(agent.new_starting_global_percentage_usage_orange),linkRete[np.argmax(agent.new_starting_global_percentage_usage_orange)])

                    print()
                ############################
                '''


                if all_negative_bool:
                    for agent in playing_agents:
                        agent.done=True

                    for agent in marl.agents:
                        result_test_dict[i_game][agent.id_node] = ['globalError']
                    #dones = [True]*len(playing_agents)
                else:
                    for agent in playing_agents:
                        if agent.number_link_over_th() == 0:
                            agent.done = True #
                        #else:
                        #    False


                if all_negative_bool:
                    print_error('!!!!!!!!!!!!!!!!!!!! Global Error !!!!!!!!!!!!!!!!!!!!!!!!!!!')


                marl.marl_update_iterations(playing_agents)

                playing_agents, dones = marl.get_playing_agents()

                #print('all_last_rewards:\n', rewards)
            print('res prima dela fine:\n', result_test_dict[i_game])
            for agent in marl.agents:
                if not isinstance(result_test_dict[i_game][agent.id_node][0], str):
                    #if result_test_dict[i_game][agent.id_node][0] != 'empty' or result_test_dict[i_game][agent.id_node][0] != 'globalError':
                    result_test_dict[i_game][agent.id_node] = result_test_dict[i_game][agent.id_node] \
                                                                + [agent.new_starting_local_out_prc_usage_orange, np.max(agent.new_starting_global_percentage_usage_orange)]
            print('res in ', str(i_game), ' :', result_test_dict[i_game])
            marl.save_scores()

            '''
            print('out of while')
            for agent in marl.agents:
                print('score: ', agent.score)
                print('scores: ', agent.scores)
            '''

            marl.marl_compute_avg_score(i_game_score)

            if not used_matrix:
                matrices_not_used += 1



            #PER ANTONIO AGGIUNTO ORA 02/03
            for R, flows in zip([R_orange], [flows_orange]):
                # zip:-> triples of r, flow and color

                request = R.dot(flows[:,-1])
                percentage_usage = request/ linkRete[:,-1]
                print(percentage_usage.shape)
                result_test_Antonio[i_game].append(percentage_usage)
                print("Appeso fine")
            #PER ANTONIO AGGIUNTO ORA 02/03
            #input()

            print(result_test_Antonio[i_game])

            i_game_score += 1

            print()
            print("Matrici non utilizzate: " + str(matrices_not_used))
            print('i_game_score', i_game_score, '\n\n\n Fine episodio \n\n\n')

        #with open(save_folder + 'test_results.pkl', 'wb') as f:
        #    pickle.dump(result_test_dict, f)

        with open(save_folder + 'test_results_Antonio_8agents.pkl', 'wb') as f:
            pickle.dump(result_test_Antonio, f)


        
    else:
        print('train flag non corretto')
