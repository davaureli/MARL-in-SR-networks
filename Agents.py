from PrioritizedReplayBuffer import PrioritizedReplayBuffer

import tensorflow as tf
import numpy as np
import copy
from tensorflow.keras.layers import Dense,Activation
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from statics import *
from utils_marl import *

import itertools


class DDQNAgent(object):
    def __init__(self,
                    id_node,
                    alpha,
                    linkRete,
                    network_number_nodes,
                    gamma,
                    Graph,
                    epsilon,
                    batch_size,
                    fname_qeval,
                    fname_qtarget,
                    fcl1,
                    fcl2,
                    epsilon_dec=0.9995,
                    epsilon_end=0.01,
                    mem_size=1000000,
                    replace_target=1,
                    scores=None,
                    eps_history=[],
                    best_score=-1000000):

        self.id_node = id_node
        self.iteration_done = None
        self.done = False
        self.map_allPossible_Actions = self.get_action_space(linkRete, Graph)
        self.n_actions = len(self.map_allPossible_Actions)
        self.action_space = [i for i in range(self.n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.fname_qeval = fname_qeval
        self.fname_qtarget = fname_qtarget
        self.replace_target = replace_target
        #self.memory = Replay_Buffer(mem_size, input_dims, n_actions,
        #                           discrete=True)

        self.memory = PrioritizedReplayBuffer(mem_size, self.n_actions)
        self.input_dims = self.computeDims(linkRete, network_number_nodes)
        #2 Layers
        self.q_eval = self.build_dqn(alpha, self.n_actions, self.input_dims, fcl1, fcl2)
        self.q_target = self.build_dqn(alpha, self.n_actions, self.input_dims, fcl1, fcl2)
        #More than 2 layers
        #self.q_eval = build_dqn(alpha, n_actions, input_dims, 1024, 512, 256)
        #self.q_target = build_dqn(alpha, n_actions, input_dims, 1024, 512, 256)

        self.eps_history=eps_history
        self.score=[]
        self.scores=[]
        self.best_score=best_score
        self.avg_score=None

        self.old_starting_local_percentage_usage_blu=0
        self.old_starting_global_percentage_usage_blu=0
        self.old_starting_local_out_prc_usage_blu=0

        self.old_starting_local_percentage_usage_orange = 0
        self.old_starting_global_percentage_usage_orange = 0
        self.old_starting_local_out_prc_usage_orange = 0

        self.new_starting_local_percentage_usage_blu=0
        self.new_starting_global_percentage_usage_blu=0
        self.new_starting_local_out_prc_usage_blu=0

        self.new_starting_local_percentage_usage_orange = 0
        self.new_starting_global_percentage_usage_orange = 0
        self.new_starting_local_out_prc_usage_orange = 0

        self.old_observation=None
        self.new_observation=None

        self.number_of_cycles = 0

    def get_action_space(self, linkRete, Graph):
        map_allPossible_Actions = {}
        #del K_MAX_PATH = 1
        node_dst_from_agent = Graph.nodes()
        conto = 0
        for n in node_dst_from_agent:
            if n != self.id_node:
                number_path = {elem:0 for elem in linkRete[linkRete[:,1] == self.id_node][:,2].tolist()}
                for path in k_shortest_paths(Graph, self.id_node, n, 200):
                    # messo lo stop a max 3 percorsi con relativo check dei vincoli (manca il vincolo se destinazione**)
                    if path[1] in number_path and number_path[path[1]] != K_MAX_PATH:
                        number_path[path[1]] += 1
                        map_allPossible_Actions[conto] = path
                        #print(path)
                        #print()
                        conto += 1
                #print(number_path)

        return map_allPossible_Actions


    def build_dqn(self, lr, n_actions, input_dims, fc0_dims,fcl_dims):
        model = Sequential([
            Dense (fc0_dims, input_shape= (input_dims, ), activation='relu'),
            Dense(fcl_dims, activation = "relu"),

            Dense(n_actions)])

        model.compile(optimizer = Adam(learning_rate = lr), loss = "mse", sample_weight_mode="temporal")

        print(model.summary())

        #exit()

        return model

    def build_dqn_more_than_2_hidden_layers(lr, n_actions, input_dims, fc0_dims,fcl_dims,fc2_dims):
        model = Sequential([
            Dense (fc0_dims, input_shape= (input_dims, ), activation='relu'),
            #Activation("relu"),
            Dense(fc0_dims, activation='relu'),
            #Activation("relu"),
            #Added Layers
            Dense(fcl_dims, activation='relu'),
            #Activation("relu"),
            Dense(fcl_dims, activation='relu'),
            #Activation("relu"),
            Dense(fc2_dims, activation='relu'),
            #Activation("relu"),
            Dense(n_actions)])

        model.compile(optimizer = Adam(lr = lr), loss = "mse", sample_weight_mode="temporal")

        return model

    def build_dqn_3_hidden(lr, n_actions, input_dims, fc0_dims,fcl_dims, fc2_dims):
        model = Sequential([
            Dense (fc0_dims, input_shape= (input_dims, ), activation='relu'),
            #Activation("relu"),
            Dense(fcl_dims, activation='relu'),
            #Activation("relu"),
            Dense(fc2_dims, activation='relu'),
            #Activation("relu"),
            Dense(n_actions)])

        model.compile(optimizer = Adam(lr = lr), loss = "mse", sample_weight_mode="temporal")

        return model
    def number_link_over_th(self):
        return sum(self.new_starting_local_out_prc_usage_orange>THRESHOLD)

    def reset_carico_link(self):
        self.old_starting_local_percentage_usage_blu = 0
        self.old_starting_global_percentage_usage_blu = 0
        self.old_starting_local_out_prc_usage_blu = 0


        self.old_starting_local_percentage_usage_orange = 0
        self.old_starting_global_percentage_usage_orange = 0
        self.old_starting_local_out_prc_usage_orange = 0

        self.new_starting_local_percentage_usage_blu = 0
        self.new_starting_global_percentage_usage_blu = 0
        self.new_starting_local_out_prc_usage_blu = 0

        self.new_starting_local_percentage_usage_orange = 0
        self.new_starting_global_percentage_usage_orange = 0
        self.new_starting_local_out_prc_usage_orange = 0

    def caricoLink(self, R_blu, R_orange, flows_blu, flows_orange, linkRete):

        # saving the ex new (the oldest) into the old variables to feel free to update the new
        self.old_starting_local_percentage_usage_blu = copy.deepcopy(self.new_starting_local_percentage_usage_blu)
        self.old_starting_global_percentage_usage_blu = copy.deepcopy(self.new_starting_global_percentage_usage_blu)
        self.old_starting_local_out_prc_usage_blu = copy.deepcopy(self.new_starting_local_out_prc_usage_blu)


        self.old_starting_local_percentage_usage_orange = copy.deepcopy(self.new_starting_local_percentage_usage_orange)
        self.old_starting_global_percentage_usage_orange = copy.deepcopy(self.new_starting_global_percentage_usage_orange)
        self.old_starting_local_out_prc_usage_orange = copy.deepcopy(self.new_starting_local_out_prc_usage_orange)

        #computing newest
        for R, flows, color in zip([R_blu, R_orange], [flows_blu, flows_orange], ['blu', 'orange']):
            # zip:-> triples of r, flow and color

            request = R.dot(flows[:,-1])

            #print(request)

            percentage_usage = request/linkRete[:,-1]

            id_link = linkRete[(linkRete[:,1]==self.id_node) | (linkRete[:,2]==self.id_node)][:,0]

            #bndwdth = linkRete[(linkRete[:,1]==1) | (linkRete[:,2]==1)][:,-1]

            #local_info

            local_perc_usage = percentage_usage[list(id_link)]

            id_link_out = linkRete[(linkRete[:,1]==self.id_node)][:,0]

            linkout_perc_usage = percentage_usage[list(id_link_out)]

            # updating the newest
            #print('agent: ', self.id_node)
            if color=='blu':
                self.new_starting_local_percentage_usage_blu = local_perc_usage
                self.new_starting_global_percentage_usage_blu = percentage_usage
                self.new_starting_local_out_prc_usage_blu = linkout_perc_usage
            else:
                self.new_starting_local_percentage_usage_orange = local_perc_usage
                self.new_starting_global_percentage_usage_orange = percentage_usage
                self.new_starting_local_out_prc_usage_orange = linkout_perc_usage
        '''
        print('Agente: ', self.id_node)
        print('orange old')
        print(self.old_starting_local_out_prc_usage_orange)
        print('orange new')
        print(self.new_starting_local_out_prc_usage_orange)
        input('go')
        '''


                #print(self.id_node, '   ', np.max(self.old_starting_global_percentage_usage_orange), '   ',np.max(self.new_starting_global_percentage_usage_orange) )

    # da eliminare
    def new_caricoLink(self, R_blu, R_orange, flows_blu, flows_orange, linkRete, old):
        #computing newest
        for R, flows, color in zip([R_blu, R_orange], [flows_blu, flows_orange], ['blu', 'orange']):
            # zip:-> triples of r, flow and color

            request = R.dot(flows[:,-1])

            #print(request)

            percentage_usage = request/linkRete[:,-1]

            id_link = linkRete[(linkRete[:,1]==self.id_node) | (linkRete[:,2]==self.id_node)][:,0]

            #bndwdth = linkRete[(linkRete[:,1]==1) | (linkRete[:,2]==1)][:,-1]

            #local_info

            local_perc_usage = percentage_usage[list(id_link)]

            id_link_out = linkRete[(linkRete[:,1]==self.id_node)][:,0]

            linkout_perc_usage = percentage_usage[list(id_link_out)]

            # updating the newest
            if color=='blu':
                if old:
                    self.old_starting_local_percentage_usage_blu = local_perc_usage
                    self.old_starting_global_percentage_usage_blu = percentage_usage
                    self.old_starting_local_out_prc_usage_blu = linkout_perc_usage
                else:
                    self.new_starting_local_percentage_usage_blu = local_perc_usage
                    self.new_starting_global_percentage_usage_blu = percentage_usage
                    self.new_starting_local_out_prc_usage_blu = linkout_perc_usage
            else:
                if old:
                    self.old_starting_local_percentage_usage_orange = local_perc_usage
                    self.old_starting_global_percentage_usage_orange = percentage_usage
                    self.old_starting_local_out_prc_usage_orange = linkout_perc_usage
                else:
                    self.new_starting_local_percentage_usage_orange = local_perc_usage
                    self.new_starting_global_percentage_usage_orange = percentage_usage
                    self.new_starting_local_out_prc_usage_orange = linkout_perc_usage

    def orange_caricoLink(self, R, flows, linkRete):

        # saving the ex new (the oldest) into the old variables to feel free to update the new
        self.old_starting_local_percentage_usage_orange = self.new_starting_local_percentage_usage_orange
        self.old_starting_global_percentage_usage_orange = self.new_starting_global_percentage_usage_orange
        self.old_starting_local_out_prc_usage_orange = self.new_starting_local_out_prc_usage_orange

        #computing newest
        request = R.dot(flows[:,-1])

        #print(request)

        percentage_usage = request/linkRete[:,-1]

        id_link = linkRete[(linkRete[:,1]==self.id_node) | (linkRete[:,2]==self.id_node)][:,0]

        #bndwdth = linkRete[(linkRete[:,1]==1) | (linkRete[:,2]==1)][:,-1]

        #local_info

        local_perc_usage = percentage_usage[list(id_link)]

        id_link_out = linkRete[(linkRete[:,1]==self.id_node)][:,0]

        linkout_perc_usage = percentage_usage[list(id_link_out)]


        self.new_starting_local_percentage_usage_orange = local_perc_usage
        self.new_starting_global_percentage_usage_orange = percentage_usage
        self.new_starting_local_out_prc_usage_orange = linkout_perc_usage

        '''
        print('orange old')
        print(self.old_starting_local_out_prc_usage_orange)
        print('orange new')
        print(self.new_starting_local_out_prc_usage_orange)
        '''

            #print(self.id_node, '   ', np.max(self.old_starting_global_percentage_usage_orange), '   ',np.max(self.new_starting_global_percentage_usage_orange)

    def computeDims(self, linkRete, network_number_nodes):
        local_dim = linkRete[(linkRete[:,1]==self.id_node) | (linkRete[:,2]==self.id_node)].shape[0]
        global_dim = linkRete.shape[0]
        return local_dim + global_dim + network_number_nodes

    def computeState(self, YbExp_orange, PSID_orange):
        #new in hold
        self.old_observation=copy.deepcopy(self.new_observation)

        #print_warning('----->' + str(self.new_starting_local_percentage_usage_orange) + '    ' + str(self.new_starting_global_percentage_usage_blu))
        #computa new
        self.new_observation=np.concatenate((self.new_starting_local_percentage_usage_orange, self.new_starting_global_percentage_usage_blu))

        #**Create a complete input considering both information** Links Utilization and Counter Vector
        Yobs = YbExp_orange[PSID_orange[:,1]==self.id_node]
        Yobs = Yobs/sum(Yobs)
        self.new_observation = np.concatenate((self.new_observation, np.reshape(Yobs,(-1,))))

    # da eliminare
    def new_computeState(self, YbExp_orange, PSID_orange, old):
        #new in hold
        self.old_observation=copy.deepcopy(self.new_observation)

        if old:
            #print_warning('----->' + str(self.new_starting_local_percentage_usage_orange) + '    ' + str(self.new_starting_global_percentage_usage_blu))
            #computa new
            self.new_observation=np.concatenate((self.old_starting_local_percentage_usage_orange, self.old_starting_global_percentage_usage_blu))
        else:
            self.new_observation=np.concatenate((self.new_starting_local_percentage_usage_orange, self.new_starting_global_percentage_usage_blu))

        #**Create a complete input considering both information**
        Yobs = YbExp_orange[PSID_orange[:,1]==self.id_node]
        Yobs = Yobs/sum(Yobs)
        self.new_observation = np.concatenate((self.new_observation, np.reshape(Yobs,(-1,))))

    def computeAvgScore(self, i_game_score):
        avg_list = self.scores[max(0,i_game_score - 200):(i_game_score+1)]
        self.avg_score = np.mean(list(itertools.chain(*avg_list)))

    def remember(self, action, reward):
        #self.memory.store_transition(state, action, reward, new_state, done)
        #print("Remember: done = ",self.done)
        self.memory.add((self.old_observation, action ,reward, self.new_observation, 1 -int(self.done)))

    def choose_action(self):
        state = self.new_observation[np.newaxis, :]
        state = np.array(state, dtype=np.float64)

        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            #begin_time = time.time()
            actions = self.q_eval.predict(state)
            #print(time.time() - begin_time)
            action = np.argmax(actions)

        return action

    def learn(self,a = 0.75):
        if self.memory.mem_cntr > self.batch_size:
            #state, action, reward, new_state, done = \
            #                              self.memory.sample_buffer(self.batch_size)

            (state, action, reward, new_state, done), importance, indices = self.memory.sample(self.batch_size, priority_scale=a)


            action = np.array(action, dtype = np.int)
            state = np.array(state, dtype = np.float)
            new_state = np.array(new_state, dtype = np.float)
            reward = np.array(reward, dtype = np.float)
            done = np.array(done, dtype = np.float)

            action_values = np.array(self.action_space, dtype=np.int64)
            action_indices = np.dot(action, action_values)

            q_next = self.q_target.predict(new_state)
            q_eval = self.q_eval.predict(new_state)
            q_pred = self.q_eval.predict(state)



            max_actions = np.argmax(q_eval, axis=1)

            q_target = copy.deepcopy(q_pred)

            batch_index = np.arange(self.batch_size, dtype=np.int32)


            q_target[batch_index, action_indices] = reward + self.gamma*q_next[batch_index, max_actions.astype(int)]*done


            #Compute the delta error which is used to update the prob to select an item
            error = np.abs(q_pred[batch_index, action_indices] - q_target[batch_index, action_indices])

            self.memory.set_priorities(indices, error)

            _ = self.q_eval.fit(state, q_target, verbose=0,sample_weight=(np.array(importance**(1-self.epsilon))))

            self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon >  self.epsilon_min else self.epsilon_min

            if self.memory.mem_cntr % self.replace_target == 0:
                self.update_network_parameters()


    def update_network_parameters(self):
        self.q_target.set_weights(self.q_eval.get_weights())

    def reward_function(self, cammini, action):
        # compute old path
        old_path = cammini[self.id_node][self.map_allPossible_Actions[action][-1]]

        action_selected = self.map_allPossible_Actions[action]

        val_1 = round(np.max(self.old_starting_global_percentage_usage_orange) /np.max(self.new_starting_global_percentage_usage_orange),3)
        #val_1 = round(round(np.max(self.old_starting_global_percentage_usage_orange),2) /round(np.max(self.new_starting_global_percentage_usage_orange),2),3)
        '''
        print("Calcolo componente Val 1",np.max(self.old_starting_global_percentage_usage_orange),np.max(self.new_starting_global_percentage_usage_orange))
        print(np.max(self.old_starting_global_percentage_usage_orange) /np.max(self.new_starting_global_percentage_usage_orange))
        print(val_1)
        '''
        #print(np.max(self.old_starting_global_percentage_usage_orange), '   ',np.max(self.new_starting_global_percentage_usage_orange) )

        #Global Reward
        #print_warning('val1: '+ str(val_1))
        if val_1 >= 1:
            #rew_1 = math.e**(2*(val_1-1))
            rew_1 = 0
        elif val_1 < 1:
            #rew_1 = -math.e**(2*(1/val_1-1))
            #rew_1 = -10
            rew_1 = MAX_NEGATIVE_REWARD
            #print_warning('peggioramento globale!')
        else:
            rew_1 = 0

        #Calcolo val_2
        num = round(compute_distance(self.old_starting_local_out_prc_usage_orange),4)
        den = round(compute_distance(self.new_starting_local_out_prc_usage_orange),4)

        #print('start = %s and end = %s'%(num, den))
        if num == 0 and den == 0:
            val_2 = 1
        elif num != 0 and den == 0:
            val_2 = MAX_POSITIVE_REWARD
        else:
            val_2 = round(num/den,3)

        #print_warning('val_2: ' + str(val_2))
        #Local Reward
        if val_2 == MAX_POSITIVE_REWARD:# or val_2 == 0:
            rew_2 = MAX_POSITIVE_REWARD
        elif val_2 > 1:
            try:
                rew_2 = min(BOUNDED_REWARD,math.e**(val_2))
                #rew_2 = min(10,val_2)
                #rew_2 = val_2
            except:
                rew_2 = BOUNDED_REWARD
        elif val_2 < 1:
            try :
                rew_2 = max(-BOUNDED_REWARD,-math.e**((1/val_2)))
                #rew_2 = max(-10,- 1/val_2)
                #rew_2 = - 1/val_2
            except:
                rew_2 = -BOUNDED_REWARD
        else:
            rew_2 = 0
            add = compute_newPath_length(old_path, action_selected)
            # print("Valutata variazione lunghezza path ",add)
            rew_2 += add

        #fixed value
        rew_3 = 0

        #reward_ = round(rew_1 + rew_2+val_3 ,4)
        if rew_1 == MAX_NEGATIVE_REWARD:
            #print("Globale NEGATIVA")
            reward_ = MAX_NEGATIVE_REWARD
        else:
            reward_ = round(rew_1 + rew_2 ,3)
            #print("Total Reward: ", reward_)
        # print("Reward: ", reward_)
        # print()
        # print()

        return rew_1, rew_2, rew_3, reward_

    def save_model(self, folder):
        self.q_eval.save(folder+self.fname_qeval)
        self.q_target.save(folder+self.fname_qtarget)

    def load_model(self, folder):
      self.q_eval = load_model(folder + self.fname_qeval)
      self.q_target = load_model(folder + self.fname_qtarget)
        #if self.memory.mem_cntr % self.replace_target == 0:
          #self.update_network_parameters()

    #alla fine era inutile
    def save(self, folder):
        self.save_model(folder) # saving NNs before save the object since NNs cannot be saved with the object

        with open(folder + 'agent_' + str(self.id_node), 'wb') as f: #creation of the name of the agent's file
            #writing object file
            dill.dump(self, f)


    def __getstate__(self): # obtain the description of the object
        state = self.__dict__.copy()
        # Don't pickle q_eval and q_target
        del state["q_eval"]
        del state["q_target"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state) # reloding of the object
        # Add q_eval since it doesn't exist in the pickle
        #self.load_model() # lasciato solo nel resume poichè qui non abbiamo la cartella
