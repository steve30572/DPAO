
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
from copy import deepcopy
import random
from importlib import import_module
import pandas as pd
import time

random.seed(100)
torch.manual_seed(100)
loss_save=[]



to_memory = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])


class EstimatorNetwork(nn.Module):
    '''
    Real DQN part, all made with MLP(Linear) layers
    '''

    def __init__(self, action_num=4, input_dimension=None, mlp_layers=None):
        '''
        Initializing Q network
        action_num for last layer
        state_shape and mlp_layers for stacking the layers
        '''
        super(EstimatorNetwork, self).__init__()
        self.action_num = action_num
        self.input_dimension = input_dimension
        self.mlp_layers = mlp_layers
        layers = [input_dimension] + self.mlp_layers
        fc = [nn.Flatten()]
        for i in range(len(layers) - 1):
            fc.append(nn.Linear(layers[i], layers[i + 1], bias=True))
            nn.init.xavier_uniform_(fc[-1].weight)
            fc.append(nn.BatchNorm1d(layers[i+1]))
            fc.append(nn.Dropout(p=0.5))
            fc.append(nn.ReLU())

        fc.append(nn.Linear(layers[-1], self.action_num, bias=True))
        nn.init.xavier_uniform_(fc[-1].weight)
        self.DQN = nn.Sequential(*fc)

    def forward(self, s):
        '''
        Predicting action values
        '''
        return self.DQN(s)


# In[4]:


class Estimator(object):
    '''
    Q network for target and source +
    '''

    def __init__(self, action_num=4, learning_rate=0.001, input_dimension=None, mlp_layers=None, device=None):
        self.action_num = action_num
        self.learning_rate = learning_rate
        self.input_dimension = input_dimension
        self.mlp_layers = mlp_layers
        self.device = device

        qnet = EstimatorNetwork(action_num, input_dimension, mlp_layers)
        qnet = qnet.to(self.device)
        self.qnet = qnet
        self.qnet.eval()
        for p in self.qnet.parameters():
            if len(p.data.shape) > 1:
                nn.init.xavier_uniform_(p.data)

        self.mse = nn.MSELoss(reduction='mean')
        self.optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.learning_rate)

    def predict_nograd(self, s):
        '''
        predict with no gradient flow.
        used when to predict the target value(max Q(s',a'))
        '''
        with torch.no_grad():
            s = s.to(self.device)
            # print("shape is ")
            # print(s.shape)

            # print(s.shape)
            q_as = self.qnet(s).to('cpu').numpy()
        return q_as

    def update(self, s, a, y):
        '''
        update with state and value and action
        '''
        self.optimizer.zero_grad()
        self.qnet.train()
        s = s
        q_as = self.qnet(s)
        Q = torch.zeros(y.shape, dtype=torch.float)
        for i in range(len(Q)):
            Q[i] = q_as[i][a[i]]
        Q = Q.to(self.device)
        y=y.to(self.device)
        batch_loss = self.mse(Q, y)
        l2_lambda = 0.0002
        l2_reg = torch.tensor(0.)
        l2_reg = l2_reg.to(self.device)
        for param in self.qnet.parameters():
            l2_reg += torch.norm(param)
        batch_loss += l2_lambda * l2_reg

        batch_loss.backward()
        self.optimizer.step()
        batch_loss = batch_loss.item()
        self.qnet.eval()
        return batch_loss


# In[5]:


class Normalizer(object):
    ''' Normalizer class that tracks the running statistics for normlization
    '''

    def __init__(self):
        ''' Initialize a Normalizer instance.
        '''
        self.mean = None
        self.std = None
        self.state_memory = []
        self.max_size = 1000
        self.length = 0

    def normalize(self, s):
        ''' Normalize the state with the running mean and std.
        Args:
            s (numpy.array): the input state
        Returns:
            a (int):  normalized state
        '''
        if self.length == 0:
            return s
        return (s - self.mean) / (self.std + 1e-8)

    def append(self, s):
        ''' Append a new state and update the running statistics
        Args:
            s (numpy.array): the input state
        '''
        if len(self.state_memory) > self.max_size:
            self.state_memory.pop(0)
        self.state_memory.append(s)
        self.mean = np.mean(self.state_memory, axis=0)
        self.std = np.mean(self.state_memory, axis=0)
        self.length = len(self.state_memory)


class Memory(object):
    ''' Memory for saving transitions
    '''

    def __init__(self, memory_size, batch_size):
        ''' Initialize
        Args:
            memory_size (int): the size of the memroy buffer
        '''
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []

    def save(self, state, action, reward, next_state, done):
        ''' Save transition into memory
        Args:
            state (numpy.array): the current state
            action (int): the performed action ID
            reward (float): the reward received
            next_state (numpy.array): the next state after performing the action
            done (boolean): whether the episode is finished
        '''
        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        transition = to_memory(state, action, reward, next_state, done)
        self.memory.append(transition)

    def append(self, to_memory):
        self.memory.append(to_memory)

    def sample(self):
        ''' Sample a minibatch from the replay memory
        Returns:
            state_batch (list): a batch of states
            action_batch (list): a batch of actions
            reward_batch (list): a batch of rewards
            next_state_batch (list): a batch of states
            done_batch (list): a batch of dones
        '''
        samples = random.sample(self.memory, self.batch_size)
        return map(torch.tensor, zip(*samples))




class Double_DQN(object):
    def __init__(self, KGAT, optimizer, action_num, learning_rate, state_shape, mlp_layers, device, discount_factor,
                 replay_size, batch_size, epsilon_decay, alpha,max_timestamp,count_train,large_K, base_line_number):

        self.imported=import_module('time')
        self.KGAT = KGAT
        self.count_train=count_train
        self.K=large_K
        self.q_estimator_user = Estimator(action_num, learning_rate, state_shape, mlp_layers, device)
        self.target_estimator_user = Estimator(action_num, learning_rate, state_shape, mlp_layers, device)
        self.q_estimator_item = Estimator(action_num, learning_rate, state_shape, mlp_layers, device)
        self.target_estimator_item = Estimator(action_num, learning_rate, state_shape, mlp_layers, device)
        self.target_estimator_user = deepcopy(self.q_estimator_user)
        self.target_estimator_item = deepcopy(self.q_estimator_item)
        self.discount_factor = discount_factor
        self.normalizer_user = Normalizer()
        self.normalizer_item = Normalizer()
        self.device = device
        self.batch_size=batch_size
        self.memory_user = Memory(replay_size, batch_size)
        self.memory_item = Memory(replay_size, batch_size)
        self.epsilon_decay = epsilon_decay
        self.total_t = 0
        self.train_t = 0
        self.epsilons = np.linspace(1.0, 0.1, self.epsilon_decay)
        self.action2 = []
        self.action3 = []
        self.action4 = []
        self.action5 = []
        self.action_list = [self.action2, self.action3, self.action4, self.action5]
        self.action = []
        self.user_index = []
        self.user_count = 0
        self.alpha = alpha
        self.state_user = []
        self.state_item = []
        self.action_user = []
        self.action_item = []
        self.reward_user = []
        self.reward_item = []
        self.reward = []
        self.dictionary = {'user': [], 'item': [], 'u_action': [], 'i_action': [], 'neg_item': [], 'neg_action': []}
        self.dictionary_neg = {'user': [], 'item': [], 'u_action': [], 'i_action': [], 'neg_user': [], 'neg_action': []}
        self.pair_user = []
        self.pair_pos = []
        self.pair_neg = []
        self.just_for_print = []
        self.max_timestamp=max_timestamp
        self.deepcopy_count=0
        self.base_line_reward=[0]
        self.base_line_number=base_line_number
        self.action_num = action_num
        self.optimizer = optimizer
        self.KGAT_train=[]
        self.KGAT_train_neg = []
        self.kgat_user = np.arange(self.KGAT.n_users)
        self.kgat_user = self.kgat_user + self.KGAT.n_entities
        self.interacted_user = np.array([])
        self.kgat_item = np.arange(self.KGAT.n_entities)
        self.interacted_item = np.array([])
        self.index_dict_add = 0
        self.set_for_dup = set()
        self.set_for_dup_item = set()
        self.numbers = dict()
        self.count = 0
        self.sparsity = 100000
        self.base_line_reward_user = []
        self.base_line_reward_item = []



    
    def update_target(self):
        print("update target estimators")
        self.target_estimator_user = deepcopy(self.q_estimator_user)
        self.target_estimator_item = deepcopy(self.q_estimator_item)
    
    def get_reward_item_new2(self, item, action, hop_result, a):
        user_list = np.array(self.state_user)
        pos_user_candidate = self.KGAT.adj_list[item].toarray()
        pos_user_candidate = pos_user_candidate[0]
        pos_user_candidate = np.argwhere(pos_user_candidate>=1)
        real_pos = np.intersect1d(user_list, pos_user_candidate)

        neg_user_candidate = np.setdiff1d(user_list, pos_user_candidate)


        if len(real_pos) != 0:
            item_embed = hop_result[action, item]
            reward = 0
            for i in real_pos:
                temp_reward = 0
                index_ = np.where(user_list == i)[0][0]
                neg_user = np.random.choice(neg_user_candidate, 1)[0]
                neg_index = np.where(user_list == neg_user)[0][0]
                self.dictionary_neg['item'] += [item]
                self.dictionary_neg['user'] += [i]
                self.dictionary_neg['neg_user'] += [neg_user]
                self.dictionary_neg['i_action'] += [action]
                self.dictionary_neg['u_action'] += [self.action_user[index_]]
                self.dictionary_neg['neg_action'] += [self.action_user[neg_index]]
                pos_embed = hop_result[self.action_user[index_], i]
                neg_embed = hop_result[self.action_user[neg_index], neg_user]
                temp_reward += torch.matmul(item_embed, pos_embed.T).item() - torch.matmul(item_embed, neg_embed.T).item()
                total_score = 0
                for j in range(4):
                    item_embed_action = hop_result[j, item]
                    total_score += torch.matmul(item_embed_action, pos_embed.T).item() - torch.matmul(item_embed_action, neg_embed.T).item()
                temp_reward /= total_score
                reward += temp_reward
            reward = reward / len(real_pos)
            return reward
        else:
            return -10

    def get_reward_user_new2(self, user, action, hop_result, a):
        item_list = np.array(self.state_item)
        pos_item_candidate = self.KGAT.adj_list[user].toarray()
        pos_item_candidate = pos_item_candidate[0]
        pos_item_candidate = np.argwhere(pos_item_candidate >= 1)
        real_pos = np.intersect1d(item_list, pos_item_candidate)
        neg_item_candidate = np.setdiff1d(item_list, pos_item_candidate)
        if len(real_pos) != 0:
            user_embed = hop_result[action, user]
            reward = 0
            for i in real_pos:
                temp_reward = 0
                index_ = np.where(item_list == i)[0][0]
                neg_item = np.random.choice(neg_item_candidate, 1)[0]
                neg_index = np.where(item_list == neg_item)[0][0]
                self.dictionary['user'] += [user]
                self.dictionary['item'] += [i]
                self.dictionary['neg_item'] += [neg_item]
                self.dictionary['u_action'] += [action]
                self.dictionary['i_action'] += [self.action_item[index_]]
                self.dictionary['neg_action'] += [self.action_item[neg_index]]
                pos_embed = hop_result[self.action_item[index_], i]
                neg_embed = hop_result[self.action_item[neg_index], neg_item]
                temp_reward += torch.matmul(user_embed, pos_embed.T).item() - torch.matmul(user_embed, neg_embed.T).item()
                total_score = 0
                for j in range(4):
                    user_embed_action = hop_result[j, user]
                    total_score += torch.matmul(user_embed_action, pos_embed.T).item() - torch.matmul(user_embed_action, neg_embed.T).item()
                temp_reward /= total_score
                reward += temp_reward
            reward = reward / len(real_pos)
            return reward
        else:
            return -10

    def learn(self, g, train_dict,edge_based_dict, KGAT, baseline_train):
        with torch.no_grad():
            a, hop_result = self.KGAT('make_hop', 'gcn',  g)
        b = time.time()
        for i in range(self.epsilon_decay):
            first_time = time.time()
            self.base_line_reward_user = []
            self.base_line_reward_item = []
            self.set_user = []

            count = 0
            self.random_pick(g, train_dict, [], KGAT)
            user = self.user
            item = self.item
            done = True

            for j in range(self.max_timestamp):

                self.state_user.append(user)

                self.state_item.append(item)
                user_embed = self.KGAT.entity_user_embed(torch.tensor(user, device=self.device))
                item_embed = self.KGAT.entity_user_embed(torch.tensor(item, device=self.device))
                action_4_user = self.action_user_func(user_embed)
                action_4_item = self.action_item_func(item_embed)
                self.action_user.append(action_4_user)
                self.action_item.append(action_4_item)
                next_state_4_user, bool_user = self.next_state_user(user, action_4_user, item)

                next_state_4_item, bool_item = self.next_state_item(item, action_4_item, user)
                if j >= baseline_train:
                    reward_user_cand = self.get_reward_user_new2(user, action_4_user, hop_result, a)
                    reward_item_cand = self.get_reward_item_new2(item, action_4_item, hop_result, a)
                    if count == self.max_timestamp:
                        if reward_user_cand != 0:
                            transition_user = to_memory(user, action_4_user, reward_user_cand, 0, 0)
                            self.memory_user.append(transition_user)
                        if reward_item_cand != 0:
                            transition_item = to_memory(item, action_4_item, reward_item_cand, 0, 0)
                            self.memory_item.append(transition_item)
                    else:
                        if reward_user_cand != 0:
                            transition_user = to_memory(user, action_4_user, reward_user_cand, next_state_4_user, 1)
                            self.memory_user.append(transition_user)
                        if reward_item_cand != 0:
                            transition_item = to_memory(item, action_4_item, reward_item_cand, next_state_4_item, 1)
                            self.memory_item.append(transition_item)
                user = next_state_4_user
                item = next_state_4_item


                count += 1
                if count >= self.max_timestamp:
                    done = False
            loss_user = self.train_user()
            loss_item = self.train_item()
            self.state_user = []
            self.state_item = []
            print(len(self.dictionary['user']))
            if len(self.dictionary['user']) >= self.count_train:
                user = torch.tensor(self.dictionary['user'], dtype=int).detach()
                item = torch.tensor(self.dictionary['item'], dtype=int).detach()
                neg_item = torch.tensor(self.dictionary['neg_item'], dtype=int).detach()
                u_action = torch.tensor(self.dictionary['u_action'], dtype=int).detach()
                i_action = torch.tensor(self.dictionary['i_action'], dtype=int).detach()
                neg_action = torch.tensor(self.dictionary['neg_action'], dtype=int).detach()
                temp_result = [user, item, neg_item, u_action, i_action, neg_action]
                self.KGAT_train.append(temp_result)
                self.dictionary = {'user': [], 'item': [], 'u_action': [], 'i_action': [], 'neg_item': [],
                                   'neg_action': []}


            if len(self.dictionary_neg['user']) >= self.count_train:
                user = torch.tensor(self.dictionary_neg['user'], dtype=int).detach()
                item = torch.tensor(self.dictionary_neg['item'], dtype=int).detach()
                neg_item = torch.tensor(self.dictionary_neg['neg_user'], dtype=int).detach()
                u_action = torch.tensor(self.dictionary_neg['u_action'], dtype=int).detach()
                i_action = torch.tensor(self.dictionary_neg['i_action'], dtype=int).detach()
                neg_action = torch.tensor(self.dictionary_neg['neg_action'], dtype=int).detach()
                temp_result = [user, item, neg_item, u_action, i_action, neg_action]
                self.KGAT_train_neg.append(temp_result)
                self.dictionary_neg = {'user': [], 'item': [], 'u_action': [], 'i_action': [], 'neg_user': [],
                                       'neg_action': []}
            
            self.total_t += 1
            self.deepcopy_count += 1
        self.total_t=0
        self.train_KGAT(g)
        self.reset_for_dict()



    def train_KGAT(self,g):
        self.KGAT.train()
        print("length of the KGAT_train is: ",len(self.KGAT_train), len(self.KGAT_train_neg))
        result=[]
        count = 0
        if len(self.KGAT_train) == 0 or len(self.KGAT_train_neg) == 0:
            return 0
        for j in range(1):
            for i in self.KGAT_train:


                self.optimizer.zero_grad()
                loss = self.KGAT('calc_cf_loss', g, i[0], i[1], i[2], i[3], i[4], i[5])


                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                del loss
        count = 0
        for j in range(1):
            for i in self.KGAT_train_neg:

                self.optimizer.zero_grad()
                loss = self.KGAT('opposite_calc_cf_loss', g, i[0], i[1], i[2], i[3], i[4], i[5])

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                del loss
        self.KGAT_train = []
        self.KGAT_train_neg = []
        print("finished")


    def train_user(self):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory_user.sample()
        next_state_batch=next_state_batch.to(self.device)
        next_state_batch = self.KGAT.entity_user_embed(next_state_batch)
        next_state_batch = next_state_batch.reshape(-1, self.KGAT.args.entity_dim)
        q_values_next = self.q_estimator_user.predict_nograd(next_state_batch)
        best_actions = np.argmax(q_values_next, axis=-1)
        q_values_next_target = self.target_estimator_user.predict_nograd(next_state_batch)
        target_batch = reward_batch+ done_batch * self.discount_factor * \
                       q_values_next_target[np.arange(self.batch_size), best_actions]
        target_batch=target_batch.to(torch.float32)
        state_batch = state_batch.to(self.device)
        state_batch = self.KGAT.entity_user_embed(state_batch)
        loss = self.q_estimator_user.update(state_batch, action_batch, target_batch)
        if self.total_t==self.epsilon_decay-2:
            print("saving the loss")
            loss_save.append(loss)
        if self.deepcopy_count % 5000 == 0:
            print("it is going to be changed")
            self.target_estimator_user = deepcopy(self.q_estimator_user)
            self.target_estimator_item = deepcopy(self.q_estimator_item)
        return loss

    def train_item(self):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory_item.sample()
        next_state_batch = next_state_batch.to(self.device)
        next_state_batch = self.KGAT.entity_user_embed(next_state_batch)
        next_state_batch = next_state_batch.reshape(-1, self.KGAT.args.entity_dim)
        q_values_next = self.q_estimator_item.predict_nograd(next_state_batch)
        best_actions = np.argmax(q_values_next, axis=-1)
        q_values_next_target = self.target_estimator_item.predict_nograd(next_state_batch)
        target_batch = reward_batch + done_batch * self.discount_factor * \
                       q_values_next_target[np.arange(self.batch_size), best_actions]
        target_batch=target_batch.to(torch.float32)
        state_batch = state_batch.to(self.device)
        state_batch = self.KGAT.entity_user_embed(state_batch)
        loss = self.q_estimator_item.update(state_batch, action_batch, target_batch)

        return loss

    def reset_for_dict(self):
        self.pair_user = []
        self.pair_pos = []
        self.pair_neg = []
        self.state_user = []
        self.state_item = []
        self.action_user = []
        self.action_item = []
        self.reward_user = []
        self.reward_item = []
        self.base_line_reward_user = []
        self.base_line_reward_item = []
        self.user_count = 0
        self.state_user = []
        self.state_item = []
        self.action_user = []
        self.action_item = []
        self.reward = []

        self.interacted_user = np.array([])
        self.interacted_item = np.array([])
        self.index_dict_add = 0
        self.set_for_dup = set()
        self.set_for_dup_item = set()


    def reset_dict(self):
       self.numbers = dict()
       self.count = 0


    def next_state_user(self, idx, action, item_index):
        action = action + 1
        for_check = np.zeros(self.KGAT.n_entities+self.KGAT.n_users)
        if action % 2 == 1:
            action += 1
        temp = self.KGAT.adj_list[idx] @ self.KGAT.adj_list
        for i in range(action-1):
            if i % 2 == 1:
                for_check = np.copy(temp.toarray())
            temp2 = temp
            temp = temp @ self.KGAT.adj_list
        index_aggregate = temp
        index_aggregate_np = index_aggregate.toarray()
        neighbor = np.argwhere(index_aggregate_np >= 1)
        neighbor = neighbor[:, 1]
        neighbor_user = neighbor[neighbor >= self.KGAT.n_entities]
        neighbor_for_check = np.where(for_check>=1)[0]
        neighbor_user = np.setdiff1d(neighbor_user, neighbor_for_check)
        next_state_bool = True
        if len(neighbor_user) == 0:
            print("why nobody.. something is wrong")
            next_state_bool = False
            return 0, next_state_bool
        else:
            next_state = np.random.choice(neighbor_user)

            return next_state, next_state_bool

    def next_state_item(self, idx, action, user_idx):
        action = action + 1
        if action % 2 == 1:
            action += 1
        a = time.time()
        temp = self.KGAT.adj_list[idx] @ self.KGAT.adj_list
        for_check = np.zeros(self.KGAT.n_entities + self.KGAT.n_users)
        for i in range(action-1):
            if i % 2 == 1:
                for_check = np.copy(temp.toarray())
            temp2 = temp
            temp = temp @ self.KGAT.adj_list
        index_aggregate = temp
        index_aggregate_np = index_aggregate
        neighbor = np.argwhere(index_aggregate_np >= 1)
        neighbor = neighbor[:, 1]
        neighbor_item = neighbor[neighbor < self.KGAT.n_items]#self.KGAT.n_entities]
        neighbor_for_check = np.where(for_check>=1)[0]
        neighbor_item = np.setdiff1d(neighbor_item, neighbor_for_check)
        next_state_bool = True
        if len(neighbor_item) == 0:
            print("why nobody in item...something wrong")
            next_state_bool = False
            return 0, next_state_bool
        else:
            next_state = np.random.choice(neighbor_item)
            return next_state, next_state_bool

    def action_user_func(self, state):
        '''
        getting action(number of hops) from user from the result of Q network with some epsilon values
        '''
        epsilon = self.epsilons[min(self.total_t, self.epsilon_decay - 1)]
        a = np.ones(self.action_num, dtype=float) * epsilon / self.action_num
        state=torch.unsqueeze(state,0)
        q_values = self.q_estimator_user.predict_nograd(state)[0]
        best_action = np.argmax(q_values)
        a[best_action] += (1.0 - epsilon)
        action = np.random.choice(np.arange(len(a)), p=a)
        self.user_index.append(self.user_count)
        self.user_count += 1
        return action

    def action_item_func(self, state):
        '''
        getting action(number of hops) from item from the result of Q network with some epsilon values
        '''
        epsilon = self.epsilons[min(self.total_t, self.epsilon_decay - 1)]
        a = np.ones(self.action_num, dtype=float) * epsilon / self.action_num
        state=torch.unsqueeze(state,0)
        q_values = self.q_estimator_user.predict_nograd(state)[0]
        best_action = np.argmax(q_values)
        a[best_action] += (1.0 - epsilon)
        action = np.random.choice(np.arange(len(a)), p=a)
        self.user_count += 1
        return action

    def random_pick(self, g, train_dict, edge_based_dict, KGAT):
        number_of_user = self.KGAT.n_users
        number_of_entity = self.KGAT.n_entities
        number_of_item = self.KGAT.n_items
        user = np.random.choice(number_of_user, 1)
        item_candidate = np.array(train_dict[user[0]+number_of_entity])
        item = np.random.choice(item_candidate, 1)
        self.item = item[0]
        self.user = user[0]+number_of_entity

    def reset(self):
        self.action2 = []
        self.action3 = []
        self.action4 = []
        self.action5 = []
        self.action_list = [self.action2, self.action3, self.action4, self.action5]
        self.action = []
        self.user_index = []
        self.user_count = 0
        self.state_user = []
        self.state_item = []
        self.action_user = []
        self.action_item = []
        self.reward = []













