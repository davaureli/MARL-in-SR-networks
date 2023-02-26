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

from utils_marl import *

from collections import deque

#Plot Traffic
import matplotlib.pyplot as plt

def extractLink_overTh(vettore,th,linkRete,id_agent):

    print("Starting vettore")
    print(vettore)
    print('id_agent: ', id_agent)

    link_agente = linkRete[linkRete[:,1] == id_agent]
    link_agente= np.hstack((link_agente,vettore.reshape((link_agente.shape[0],1))))

    link_agente = link_agente[link_agente[:,-1]>th]

    #Lista ordinata decrescente link sopra soglia
    link_agente = link_agente[link_agente[:,-1].argsort()[::-1]]

    return link_agente

def is_sub(sub, lst):
    ln = len(sub)
    for i in range(len(lst) - ln + 1):
        if all(sub[j] == lst[i+j] for j in range(ln)):
            return True
    return False

def FlowsExtraction(G,flows,id_link,linkRete,Yb,flag_order,list_remove_aggregate):

    #G --> psid * fl
    flows_overLink= flows[(G[id_link,:]==1) & (flows[:,1]==linkRete[id_link][1]) & (~np.isin(flows[:,2],list_remove_aggregate))]

    #print("Flows Over Link")
    #print(flows_overLink)
    #input()

    #psid_ = flows_overLink[:,1]*50 + flows_overLink[:,2] --> 50 possibili coppie nella network (50 germany , 17 nobel)
    psid_ = flows_overLink[:,1]*network_number_nodes + flows_overLink[:,2]

    #YbExp_orange[psid_.tolist()]
    vettore_contatori = Yb[psid_.tolist()]

    map_psid_contatore = np.hstack((psid_.reshape(psid_.shape[0],1),vettore_contatori))

    if flag_order == "a":

        map_psid_contatore = map_psid_contatore[map_psid_contatore[:,-1].argsort()]

    elif flag_order == "d":

        map_psid_contatore = map_psid_contatore[map_psid_contatore[:,-1].argsort()[::-1]]

    return map_psid_contatore

def checkAvailability(dizionario,linkRete,th,utilizzazione,aggregato,id_node):

    print("Link Alternativi: ", dizionario)

    key_del = []

    for key in dizionario:
        print('key, id_node', key, id_node)
        all_link_info = linkRete[(linkRete[:,1]==id_node) & (linkRete[:,2]==key)][0]
        var_util = aggregato[-1]/all_link_info[-1]

        print(utilizzazione[all_link_info[0]],var_util,utilizzazione[all_link_info[0]] + var_util,key)

        if utilizzazione[all_link_info[0]] + var_util > th:
            print("Link Alternativi non buoni: ", key, utilizzazione[all_link_info[0]] + var_util)
            key_del.append(key)
    for kk in key_del:
        del dizionario[kk]

    return dizionario

def checkAmmissibile(path, global_utilization_blu, global_utilization_orange,originalPath, linkRete,aggregato):


    count_error = 0

    max_check = max(global_utilization_blu)

    for i in range(1,len(path)-1):
        src_n = path[i]
        dst_n = path[i+1]

        if not is_sub([src_n,dst_n],originalPath):

            all_link_info = linkRete[(linkRete[:,1]==src_n) & (linkRete[:,2]==dst_n)][0]

            id_link = all_link_info[0]

            link_band = all_link_info[-1]

            util_var = aggregato[1]/link_band

            if util_var + global_utilization_blu[id_link] > max_check:

                return False

            #Check Se l'euristica va in errore al globale
            print("VALORI EURISTICA CHECK GLOBALE")
            print(util_var + global_utilization_orange[id_link], max(global_utilization_orange))

            if util_var + global_utilization_orange[id_link] > max(global_utilization_orange):
                print("ERRORE GLOBALE")
                print(util_var + global_utilization_orange[id_link],max(global_utilization_orange))
                print(path)
                count_error += 1

                val_globalUtilization_Orange.append((max(global_utilization_orange),util_var + global_utilization_orange[id_link]))

            else:
                val_globalUtilization_Orange.append((max(global_utilization_orange),max(global_utilization_orange)))

    if count_error>0:
        print("ERRORE EURISTICA")
        #input()
        return "ERRORE"


    return True


def Euristic(agent, id_node, starting_local_out_prc_usage_orange,global_percentage_usage_blu,global_percentage_usage_orange,linkRete,PSID_orange,YbExp_orange,tot_ERROR_global, dict_encap_orange_dict):
    number_of_cycles = 0
    #Number of paths for each output link from agent node
    k_parameter = 1

    #Immagazzinare Azioni fatte :

    azioni_fatte = []


    list_remove_aggregate= []

    flag_while = True

    numero_azioni_fatte = -1

    dict_encap_blu={}
    dict_encap_orange={}

    #Step 1 --> Ordinare i link per utilization (parametro th e agent_node)
    while flag_while:

        if len(azioni_fatte) == numero_azioni_fatte:

            return azioni_fatte, tot_ERROR_global,agent.new_starting_local_out_prc_usage_orange, number_of_cycles

        numero_azioni_fatte = len(azioni_fatte)


        print('azioni fatte: ', azioni_fatte)
        print('numero_azioni_fatte: ', numero_azioni_fatte)
        print("RESTART !!!")
        print()

        link_agente = extractLink_overTh(agent.new_starting_local_out_prc_usage_orange,0.5,linkRete,id_node)



        print()
        print("Situazione link output agente: ")
        print(link_agente)
        print()
        if link_agente.shape[0] == 0:

            return azioni_fatte, tot_ERROR_global, agent.new_starting_local_out_prc_usage_orange, number_of_cycles


        #Step 2: Selezionare il link (Highest utilization)
        for k_link in range(link_agente.shape[0]):
            print('k_link: ', k_link,'\nlinkAgente: ', link_agente)
            # print("FOR LINK")

            link_to_work = link_agente[k_link][0]

            #Recuperati gli aggragati per ogni active segment passando per il link "link to work"
            all_aggregates = FlowsExtraction(G,flows_orange,link_to_work,linkRete,YbExp_orange,"d",list_remove_aggregate)

            # print("Numero di aggregati: " , len(all_aggregates))
            # print()

            for i_agg in range(all_aggregates.shape[0]):
            #for i_agg in range(1):
                # print("FOR aGGREGATE")
                aggregate_selected = all_aggregates[i_agg]
                # print()
                # print(all_aggregates)
                # print()
                # print(aggregate_selected)

                #Scelto l'aggregato 0 ed estratto il PSID
                src_node = PSID_orange[aggregate_selected[0]][1]
                dst_node = PSID_orange[aggregate_selected[0]][2]

                print('src and dst: ', src_node, dst_node)

                # print()
                # print("Src Node: " , src_node, " Dst Node: ", dst_node)
                # print()
                alternative_path_considered = []

                number_path = {elem:0 for elem in linkRete[linkRete[:,1] == src_node][:,2].tolist() if elem not in link_agente[:,2].tolist()}

                #Check per i link non in link agente se possiamo spostare effettivamente il traffico su di loro senza portarli
                #sopra threshold
                number_path = checkAvailability(number_path,linkRete,0.5,agent.new_starting_global_percentage_usage_orange,aggregate_selected,id_node)

                for path in k_shortest_paths(Graph, src_node, dst_node, 100):
                    # messo lo stop a max 3 percorsi con relativo check dei vincoli (manca il vincolo se destinazione**)
                    if path[1] in number_path and number_path[path[1]] != k_parameter:
                        number_path[path[1]] += 1

                        alternative_path_considered.append(path)


                alternative_path_considered= sorted(alternative_path_considered, key=len)
                print("Numero percorsi alternativi: " , len(alternative_path_considered))

                #CHECK VINCOLI:
                    #1) Osservare utilizzazione locale per selzione link (fatto)
                    #2) Utilizzazione globale (fare)

                for j_path in range(len(alternative_path_considered)):
                #for j_path in range(1):
                    path_select = alternative_path_considered[j_path]

                    # print(path_select)


                    check_ammissibile = checkAmmissibile(path_select, agent.new_starting_global_percentage_usage_blu, agent.new_starting_global_percentage_usage_orange,
                                                         cammini_orange, linkRete,aggregate_selected)

                    print("Check ammissibile")
                    print(check_ammissibile)

                    if check_ammissibile and check_ammissibile != "ERRORE":
                        #dict_encap_blu={}
                        #dict_encap_orange={}


                        list_remove_aggregate.append(dst_node)

                        azioni_fatte.append(path_select)

                        #return azioni_fatte

                        #applicar el'ultima azione inserita:

                        # print("AZIONE FATTA")
                        # print(azioni_fatte[-1])

                        '''
                        dict_encap_blu = fill_Encap(dict_encap_blu,SL_blu,cammini_blu,azioni_fatte[-1])
                        R_blu = np.zeros((L_blu,K_blu))
                        B,YbExp_blu, PSID_blu, R_blu = aggiornaPSID(SL_blu, cammini_blu, flows_blu, N, linkRete, dict_encap_blu, agent_node, R_blu)
                        '''
                        print('azioni fatte: ', azioni_fatte)
                        dict_encap_orange_dict[id_node] = fill_Encap(dict_encap_orange_dict[id_node],SL_orange,cammini_orange,azioni_fatte[-1])
                        R_orange = np.zeros((L_orange,K_orange))
                        B,YbExp_orange, PSID_orange, R_orange, with_cycle = multi_aggiornaPSID(SL_orange, cammini_orange, flows_orange, network_number_nodes, linkRete, dict_encap_orange_dict)
                        agent.iteration_done += 1

                        if with_cycle:
                            number_of_cycles += 1

                        agent.orange_caricoLink(R_orange, flows_orange, linkRete)

                        print("Variation after action: ")
                        print()
                        print(agent.new_starting_local_out_prc_usage_orange)
                        print()
                        print(dict_encap_orange_dict)
                        print("GLOBAL")
                        print(max(agent.new_starting_global_percentage_usage_orange))
                        #input()

                        print('new: ', agent.new_starting_local_out_prc_usage_orange)
                        print('old: ', agent.old_starting_local_out_prc_usage_orange)
                        index_new = np.argwhere(agent.new_starting_local_out_prc_usage_orange<=THRESHOLD).reshape(1, -1)[0].tolist()
                        index_old = np.argwhere(agent.old_starting_local_out_prc_usage_orange<=THRESHOLD).reshape(1, -1)[0].tolist()

                        print('iteration done: ', agent.iteration_done)

                        if not set(index_old).issubset(index_new):
                            print_error('errore globale!')
                            tot_ERROR_global = 1
                            return "Nada",tot_ERROR_global,np.array(["errore"]), number_of_cycles

                        if agent.iteration_done == MAX_NUMBER_ACTIONS:
                            return azioni_fatte, tot_ERROR_global, agent.new_starting_local_out_prc_usage_orange, number_of_cycles

                        break


                    elif check_ammissibile == "ERRORE":
                        tot_ERROR_global = 1

                        return "Nada",tot_ERROR_global,np.array(["errore"]), number_of_cycles

                    else:

                        pass

                #list_remove_aggregate.append(dst_node)

                if len(azioni_fatte)> numero_azioni_fatte:
                    print("eccomi for aggregate")
                    break
            if len(azioni_fatte)> numero_azioni_fatte:
                print("eccomi for link")
                break

        if azioni_fatte == MAX_NUMBER_ACTIONS:
            flag_while = False
