
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple
from copy import deepcopy
import random
from importlib import import_module
import time
import itertools



random.seed(0)
torch.manual_seed(0)
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
            fc.append(nn.ReLU())

        fc.append(nn.Linear(layers[-1], self.action_num, bias=True))
        nn.init.xavier_uniform_(fc[-1].weight)
        # fc.append(nn.Sigmoid())
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
        self.optimizer = torch.optim.SGD(self.qnet.parameters(), lr=self.learning_rate)

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
        #a = a.astype(int)
        #a = torch.from_numpy(a).int().to(self.device)
        # print(y)
        # y=torch.from_numpy(y).float().to(self.device)
        #y = np.array(y, dtype=np.float32)
        #y = torch.tensor(y, device=self.device, dtype=torch.float)
        q_as = self.qnet(s)
        # print(a, a.unsqueeze(-1),y.shape)
        #Q=q_as.clone()
        Q = torch.zeros(y.shape, dtype=torch.float)
        for i in range(len(Q)):
            Q[i] = q_as[i][a[i]]
        #print("Q is: ",Q[0],Q.shape)
        # print(q_as)
        Q = Q.to(self.device)
        y=y.to(self.device)
        # Q=torch.gather(q_as,dim=-1,index=a.unsqueeze(-1)[0]).squeeze(-1)
        batch_loss = self.mse(Q, y)
        l2_lambda = 0.002
        l2_reg = torch.tensor(0.)
        l2_reg = l2_reg.to(self.device)
        for param in self.qnet.parameters():
            l2_reg += torch.norm(param)
        #print("loss of the DQN is : ", batch_loss,' ',  l2_lambda*l2_reg)
        batch_loss += l2_lambda * l2_reg

        batch_loss.backward()
        #print(batch_loss.grad)
        self.optimizer.step()
        batch_loss = batch_loss.item()
        self.qnet.eval()
        loss_save.append(batch_loss)
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
        samples = random.sample(self.memory, self.batch_size)  # self.batch_size) 5 should be changed -HS
        return map(torch.tensor, zip(*samples))




class Double_DQN(object):
    def __init__(self, NGCF, optimizer, action_num, learning_rate, state_shape, mlp_layers, device, discount_factor,
                 replay_size, batch_size, epsilon_decay, alpha,max_timestamp,count_train,large_K, base_line_number):  # 추후에 추가하기 alpha can be erased

        self.imported=import_module('time')
        self.NGCF = NGCF
        self.count_train=count_train
        self.K=large_K
        self.q_estimator_user = Estimator(action_num, learning_rate, state_shape, mlp_layers, device)
        self.target_estimator_user = Estimator(action_num, learning_rate, state_shape, mlp_layers, device)
        self.q_estimator_item = Estimator(action_num, learning_rate, state_shape, mlp_layers, device)
        self.target_estimator_item = Estimator(action_num, learning_rate, state_shape, mlp_layers, device)
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
        self.epsilons = np.linspace(1.0, 0.1, self.epsilon_decay)  # hyperparameter 조정하기
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
        self.base_line_reward=[]
        self.base_line_number=base_line_number
        self.action_num = action_num  # need to be changed(hyperparameter)--HS
        self.optimizer = optimizer#torch.optim.Adam(self.KGAT.parameters(),lr=1)#optimizer   #--0.01 need to be changed -HS
        self.NGCF_train=[]
        self.NGCF_train_neg = []
        self.kgat_user = np.arange(self.NGCF.n_user)
        self.kgat_user = self.kgat_user # + self.KGAT.n_items
        self.interacted_user = np.array([])
        self.kgat_item = np.arange(self.NGCF.n_item)
        self.interacted_item = np.array([])
        self.index_dict_add = 0
        self.base_line_reward_user = []
        self.base_line_reward_item = []
        self.reset_count = 0
        self.total_reward_user = []
        self.total_reward_item = []

    def eval_step(self, embed_vec,index):
        if index==0:
            q_values=self.q_estimator_user.predict_nograd(embed_vec)
            #print(q_values.shape)
            actions=np.argmax(q_values,axis=1)
            return actions
        if index==1:
            q_values=self.q_estimator_item.predict_nograd(embed_vec)
            actions=np.argmax(q_values,axis=-1)
            return actions

    def get_action_final_eval(self,user,pos_item):
        #embedded = self.get_embed(g, model.entity_user_embed)
        user_embed = self.NGCF.embedding_dict['user_emb'][user]#torch.tensor(user, device=self.device))
        pos_item_embed = self.NGCF.embedding_dict['item_emb'][pos_item]#torch.tensor(pos_item, device=self.device))
        action_user = self.eval_step(user_embed, 0)
        action_pos = self.eval_step(pos_item_embed, 1)
        return action_user, action_pos
    def get_action_final(self, user,pos_item,neg_item):
        #embedded=self.get_embed(g,self.KGAT.entity_user_embed)
        user_embed=self.NGCF.embedding_dict['user_emb'][user]#torch.tensor(user,device=self.device))
        pos_item_embed=self.NGCF.embedding_dict['item_emb'][pos_item]#torch.tensor(pos_item,device=self.device))
        neg_item_embed=self.NGCF.embedding_dict['item_emb'][neg_item]#torch.tensor(neg_item, device=self.device))
        action_user=self.eval_step(user_embed,0)
        action_pos=self.eval_step(pos_item_embed,1)
        action_neg=self.eval_step(neg_item_embed,1)
        return action_user, action_pos, action_neg

    def get_item_list(self, user_embed, user_index, train_dict):
        return_item = []
        return_action_item = []
        a = time.time()
        first_item_candidate = self.NGCF.plain_adj[user_index].toarray()#train_dict[user_index]
        # first_item_candidate = first_item_candidate[first_item_candidate>=self.NGCF.n_user]
        first_item_candidate = first_item_candidate[0]
        first_item_candidate = np.where(first_item_candidate>=1)
        first_item_candidate = first_item_candidate[0]
        # print(first_item_candidate, self.NGCF.n_user, user_index)
        first_item_candidate = first_item_candidate[first_item_candidate>=self.NGCF.n_user]
        # print(first_item_candidate, "yes")
        first_item = np.random.choice(first_item_candidate)
        return_item.append(first_item-self.NGCF.n_user)
        a = time.time()

        for i in range(10):
            with torch.no_grad():
                item_embed = self.NGCF.embedding_dict['item_emb'][torch.tensor(return_item[i], device=self.device)]
                action_item = self.action_item_func(item_embed)
                return_action_item.append(action_item)
                next_state_item, _ = self.next_state_item(return_item[i], action_item, user_index)
                return_item.append(next_state_item)
        return torch.tensor(return_item[:10]) + self.NGCF.n_user, torch.tensor(return_action_item)
    
    def get_user_list(self, item_embed, item_index, train_dict):
        return_user = []
        return_action_user = []
        first_user_candidate = self.NGCF.plain_adj[item_index].toarray()
        # first_user_candidate = first_user_candidate[first_user_candidate<self.NGCF.n_user]
        #print(first_user_candidate.shape)
        first_user_candidate = first_user_candidate[0]
        first_user_candidate = np.where(first_user_candidate>=1)
        first_user_candidate = first_user_candidate[0]
        first_user_candidate = first_user_candidate[first_user_candidate<self.NGCF.n_user]
        #first_user_candidate = first_user_candidate[first_user_candidate_index]
        #first_user_candidate = first_user_candidate[0]
        first_user = np.random.choice(first_user_candidate)
        return_user.append(first_user)
        for i in range(10):
            with torch.no_grad():
                user_embed = self.NGCF.embedding_dict['user_emb'][torch.tensor(return_user[i], device=self.device)]
                action_user = self.action_user_func(user_embed)
                return_action_user.append(action_user)
                #print(return_user[i], i, "yes")
                next_state_user, _ = self.next_state_user(return_user[i], action_user, item_index)
                return_user.append(next_state_user)
        return torch.tensor(return_user[:10]), torch.tensor(return_action_user)

    def get_reward_user_new(self, user_idx, action_idx, item_list_for_reward, action_list_for_reward, hop_result, train_dict):
        user_embed = hop_result[action_idx, user_idx]
        pos_item_candidate = self.NGCF.plain_adj[user_idx].toarray()#np.array(train_dict[user_idx])
        pos_item_candidate = pos_item_candidate[0]
        pos_item_candidate = np.argwhere(pos_item_candidate>=1)
        pos_item_candidate = pos_item_candidate[:, 0]
        pos_item_candidate = pos_item_candidate[pos_item_candidate>=self.NGCF.n_user]
        item_list_for_reward = item_list_for_reward.numpy()
        pos_item_candidate = pos_item_candidate
        pos_item = np.intersect1d(pos_item_candidate, item_list_for_reward)
        neg_item_candidate = np.setdiff1d(item_list_for_reward, pos_item_candidate)
        action_list = []
        neg_item_list = []
        neg_action_list = []
        if len(neg_item_candidate) == 0:  #ml-1m
            print("why 0? too dense?")
            print(item_list_for_reward, action_list_for_reward)
            neg_item_candidate = np.setdiff1d(np.arange(self.NGCF.n_user+self.NGCF.n_item), pos_item_candidate)
            for i in pos_item:
                action_list.append(action_list_for_reward[np.where(item_list_for_reward == i)[0][0]])
                neg_list_candidate = np.random.choice(neg_item_candidate)
                neg_item_list.append(neg_list_candidate - self.NGCF.n_user)
                neg_action_list.append(np.random.choice(np.arange(4)))

        else:
            for i in pos_item:
                action_list.append(action_list_for_reward[np.where(item_list_for_reward==i)[0][0]])
                neg_list_candidate = np.random.choice(neg_item_candidate)
                neg_item_list.append(neg_list_candidate-self.NGCF.n_user)
                neg_action_list.append(action_list_for_reward[np.where(item_list_for_reward==neg_list_candidate)[0][0]])
        # print(len(self.dictionary['item']), len(pos_item), "first")
        # print(pos_item_candidate)
        # print(item_list_for_reward)
        self.dictionary['user'] += [user_idx] * len(pos_item)
        self.dictionary['item'] += (pos_item-self.NGCF.n_user).tolist()
        self.dictionary['neg_item'] += neg_item_list
        self.dictionary['u_action'] += [action_idx] * len(pos_item)
        self.dictionary['i_action'] += action_list
        self.dictionary['neg_action'] += neg_action_list
        # print(len(self.dictionary['item']), len(pos_item), "second", len(self.dictionary['item']))
        action_list = np.array(action_list)
        neg_action_list = np.array(neg_action_list)
        neg_item_list = np.array(neg_item_list)
        whole_item_hop_result = hop_result[action_list, pos_item]
        neg_item_hop_result = hop_result[neg_action_list, neg_item_list]

        ##### added 0604
        #whole_item_hop_result = torch.nn.functional.normalize(whole_item_hop_result, p=2, dim=1)
        #neg_item_hop_result = torch.nn.functional.normalize(neg_item_hop_result, p=2, dim=1)

        cos_simil_pos = torch.matmul(user_embed, whole_item_hop_result.T)
        cos_simil_neg = torch.matmul(user_embed, neg_item_hop_result.T)
        return torch.mean(cos_simil_pos).item() - torch.mean(cos_simil_neg).item()

        reward = self.get_baseline_user(torch.mean(cos_simil_pos).item() - torch.mean(cos_simil_neg).item())
        return reward

    def get_reward_item_new(self, item_idx, action_idx, user_list_for_reward, action_list_for_reward, hop_result):
        item_embed = hop_result[action_idx, item_idx]
        pos_user_candidate = self.NGCF.plain_adj[item_idx].toarray()
        pos_user_candidate = pos_user_candidate[0]
        pos_user_candidate = pos_user_candidate
        pos_user_candidate = np.argwhere(pos_user_candidate>=1)
        # print(pos_user_candidate.shape)
        pos_user_candidate = pos_user_candidate[:, 0]

        pos_user_candidate = pos_user_candidate[pos_user_candidate < self.NGCF.n_user]
        # print(pos_user_candidate)
        user_list_for_reward = user_list_for_reward.numpy()
        pos_user = np.intersect1d(pos_user_candidate, user_list_for_reward)
        neg_user_candidate = np.setdiff1d(user_list_for_reward, pos_user_candidate)
        action_list = []
        neg_user_list = []
        neg_action_list = []
        if len(neg_user_candidate) == 0:  #ml-1m
            print("why 0? too dense? user")
            print(user_list_for_reward, action_list_for_reward)
            neg_user_candidate = np.setdiff1d(np.arange(self.NGCF.n_user+self.NGCF.n_item), pos_user_candidate)
            for i in pos_user:
                action_list.append(action_list_for_reward[np.where(user_list_for_reward == i)[0][0]])
                neg_list_candidate = np.random.choice(neg_user_candidate)
                neg_user_list.append(neg_list_candidate)
                neg_action_list.append(np.random.choice(np.arange(4)))
        else:
            for i in pos_user:
                action_list.append(action_list_for_reward[np.where(user_list_for_reward==i)[0][0]])
                neg_list_candidate = np.random.choice(neg_user_candidate)
                neg_user_list.append(neg_list_candidate)
                neg_action_list.append(action_list_for_reward[np.where(user_list_for_reward==neg_list_candidate)[0][0]])
        # print(len(self.dictionary_neg['item']), "first", pos_user, len(self.dictionary['item']))
        # print(pos_user_candidate)
        # print(user_list_for_reward)
        self.dictionary_neg['item'] += [item_idx] * len(pos_user)
        self.dictionary_neg['user'] += pos_user.tolist()
        self.dictionary_neg['neg_user'] += neg_user_list
        self.dictionary_neg['i_action'] += [action_idx] * len(pos_user)
        self.dictionary_neg['u_action'] += action_list
        self.dictionary_neg['neg_action'] += neg_action_list
        # print(len(self.dictionary_neg['user']), "second", len(pos_user))
        action_list = np.array(action_list)
        neg_action_list = np.array(neg_action_list)
        neg_user_list = np.array(neg_user_list)
        whole_user_hop_result = hop_result[action_list, pos_user]
        neg_user_hop_result = hop_result[neg_action_list, neg_user_list]
        #####

        #whole_user_hop_result = torch.nn.functional.normalize(whole_user_hop_result, p=2, dim=1)
        #neg_usr_hop_result = torch.nn.functional.normalize(neg_user_hop_result, p=2, dim=1)

        cos_simil_pos = torch.matmul(item_embed, whole_user_hop_result.T)
        cos_simil_neg = torch.matmul(item_embed, neg_user_hop_result.T)
        #return torch.mean(cos_simil_pos).item() - torch.mean(cos_simil_neg).item()

        reward = self.get_baseline_item(torch.mean(cos_simil_pos).item() - torch.mean(cos_simil_neg).item())
        return reward


    def update_target(self):
        print("update target estimators")
        self.target_estimator_user = deepcopy(self.q_estimator_user)
        self.target_estimator_item = deepcopy(self.q_estimator_item)

    def get_reward_item_new2(self, item, action, hop_result):
        user_list = np.array(self.state_user)
        pos_user_candidate = self.NGCF.plain_adj[item+self.NGCF.n_user].toarray()
        # pos_user_candidate[0, item+self.NGCF.n_user]=0
        pos_user_candidate = pos_user_candidate[0]
        pos_user_candidate = np.argwhere(pos_user_candidate>=1)
        # print(pos_user_candidate)
        real_pos = np.intersect1d(user_list, pos_user_candidate)
        neg_user_candidate = np.setdiff1d(user_list, pos_user_candidate)
        # print(len(pos_user_candidate), len(user_list), len(real_pos))



        #####
        if len(real_pos) != 0:
            item_embed = hop_result[action, item+self.NGCF.n_user]
            reward = 0
            # print(len(real_pos), 'is the length of real_pos')
            for i in real_pos:
                temp_reward = 0
                index_ = np.where(user_list == i)[0][0]
                neg_user = np.random.choice(neg_user_candidate, 1)[0]
                neg_index = np.where(user_list == neg_user)[0][0]
                ###
                #sample_positive = np.random.choice(pos_user_candidate[0], 1)[0]
                #sample_negative = np.random.choice(np.arange(self.NGCF.n_user, self.NGCF.n_user + self.NGCF.n_item), 1)[0]
                #self.dictionary_neg['item'] += [item]
                #self.dictionary_neg['user'] += [sample_positive]
                #self.dictionary_neg['neg_user'] += [sample_negative]
                #self.dictionary_neg['i_action'] += [action]
                #self.dictionary_neg['u_action'] += [2]
                #self.dictionary_neg['neg_action'] += [2]
                #pos_embed = hop_result[2, sample_positive]
                #neg_embed = hop_result[2, sample_negative]
                ####
                self.dictionary_neg['item'] += [item]
                self.dictionary_neg['user'] += [i]
                self.dictionary_neg['neg_user'] += [neg_user]
                self.dictionary_neg['i_action'] += [action]
                self.dictionary_neg['u_action'] += [self.action_user[index_]]
                self.dictionary_neg['neg_action'] += [self.action_user[neg_index]]
                pos_embed = hop_result[self.action_user[index_], i]
                neg_embed = hop_result[self.action_user[neg_index], neg_user]
                temp_reward += torch.matmul(item_embed, pos_embed.T) - torch.matmul(item_embed, neg_embed.T)
                total_score = 0
                for j in range(4):
                    item_embed_action = hop_result[j, item]
                    total_score += torch.matmul(item_embed_action, pos_embed.T).item() - torch.matmul(item_embed_action, neg_embed.T).item()
                temp_reward /= total_score
                reward += temp_reward
            reward = reward / len(real_pos)
            return reward
        else:
            return -20



    def get_reward_user_new2(self, user,action, hop_result):
        item_list = np.array(self.state_item)
        pos_item_candidate = self.NGCF.plain_adj[user].toarray()
        # pos_item_candidate[0, user] = 0
        pos_item_candidate = pos_item_candidate[0]
        # pos_user_candidate = pos_user_candidate
        pos_item_candidate = np.argwhere(pos_item_candidate >= 1)
        real_pos = np.intersect1d(item_list, pos_item_candidate)
        #print(len(pos_item_candidate), "is the lenght of positive interaction")
        neg_item_candidate = np.setdiff1d(item_list, pos_item_candidate)
        if len(real_pos) != 0:
            # print(real_pos, "is 0...")
            user_embed = hop_result[action, user]
            reward = 0
            for i in real_pos:
                temp_reward = 0
                index_ = np.where(item_list == i)[0][0]
                neg_item = np.random.choice(neg_item_candidate, 1)[0]
                neg_index = np.where(item_list == neg_item)[0][0]

                ### random sampling
                #print(pos_item_candidate[0], "is the candidate first")

                #pos_item = np.random.choice(pos_item_candidate[0], 1)[0]
                #neg_item = np.random.choice(np.arange(self.NGCF.n_item), 1)[0]
                #self.dictionary['user'] += [user]
                #self.dictionary['item'] += [pos_item]
                #self.dictionary['neg_item'] += [neg_item]
                #self.dictionary['u_action'] += [action]
                #self.dictionary['i_action'] += [2]
                #self.dictionary['neg_action'] += [2]
                #pos_embed = hop_result[2, pos_item]
                #neg_embed = hop_result[2, neg_item]



                ###
                self.dictionary['user'] += [user]
                self.dictionary['item'] += [i]
                self.dictionary['neg_item'] += [neg_item]
                self.dictionary['u_action'] += [action]
                self.dictionary['i_action'] += [self.action_item[index_]]
                self.dictionary['neg_action'] += [self.action_item[neg_index]]
                pos_embed = hop_result[self.action_item[index_], i+self.NGCF.n_user]
                neg_embed = hop_result[self.action_item[neg_index], neg_item+self.NGCF.n_user]
                temp_reward += torch.matmul(user_embed, pos_embed.T) - torch.matmul(user_embed, neg_embed.T)
                total_score = 0
                for j in range(4):
                    user_embed_action = hop_result[j, user]
                    total_score += torch.matmul(user_embed_action, pos_embed.T).item() - torch.matmul(user_embed_action, neg_embed.T).item()
                temp_reward /= total_score
                reward += temp_reward
            reward = reward / len(real_pos)
            return reward
        else:
            return -20



    def learn(self,train_dict, baseline_train):
        reward_change_user = []
        reward_change_item = []
        print("start")
        first_time = time.time()
        user_count = 0
        item_count = 0
        mean_reward_user = 0
        mean_reward_item = 0
        for i in range(self.epsilon_decay):
            # print(time.time()-first_time, "time of one epsilon decay")
            reward_user=[]
            traverse_time = time.time()
            count = 0
            item, user = self.item, self.user#self.random_pick(train_dict)  # need parameter check --HS
            done = True
            with torch.no_grad():
                hop_result = self.NGCF.make_hop()
            for j in range(self.max_timestamp):    # 하나당 0.4초
                self.state_user.append(user)

                self.state_item.append(item)
                #print("user and item is: ", user, item, i, self.NGCF.n_user)
                user_embed = self.NGCF.embedding_dict['user_emb'][torch.tensor(user, device=self.device)]
                item_embed = self.NGCF.embedding_dict['item_emb'][torch.tensor(item, device=self.device)]

                action_4_user = self.action_user_func(user_embed)
                action_4_item = self.action_item_func(item_embed)
                self.action_user.append(action_4_user)
                self.action_item.append(action_4_item)
                next_state_4_user, bool_user = self.next_state_user(user, action_4_user, item)
                next_state_4_item, bool_item = self.next_state_item(item, action_4_item, user)
                if j >= baseline_train:
                    if j == baseline_train:
                        user = self.user
                        item = self.item
                        self.state_user.append(user)
                        self.state_item.append(item)
                        action_4_user = self.action_user_func(user_embed)
                        action_4_item = self.action_item_func(item_embed)
                        self.action_user.append(action_4_user)
                        self.action_item.append(action_4_item)
                        next_state_4_user, bool_user = self.next_state_user(user, action_4_user, item)
                        next_state_4_item, bool_item = self.next_state_item(item, action_4_item, user)
                    reward_user_cand = self.get_reward_user_new2(user, action_4_user, hop_result)
                    reward_item_cand = self.get_reward_item_new2(item, action_4_item, hop_result)
                    if i == self.epsilon_decay-1:
                        if reward_user_cand != -20:
                            mean_reward_user += reward_user_cand
                            #print(reward_user_cand)
                            user_count += 1
                        if reward_item_cand != -20:
                            mean_reward_item += reward_item_cand
                            item_count += 1
                    if count == self.max_timestamp:
                        if reward_user_cand != -20:
                            transition_user = to_memory(user, action_4_user, reward_user_cand, 0, 0)
                            self.memory_user.append(transition_user)
                        if reward_item_cand != -20:
                            transition_item = to_memory(item, action_4_item, reward_item_cand, 0, 0)
                            self.memory_item.append(transition_item)
                    else:
                        if reward_user_cand != -20:
                            transition_user = to_memory(user, action_4_user, reward_user_cand, next_state_4_user, 1)
                            self.memory_user.append(transition_user)
                        if reward_item_cand != -20:
                            transition_item = to_memory(item, action_4_item, reward_item_cand, next_state_4_item, 1)
                            self.memory_item.append(transition_item)
                user = next_state_4_user
                item = next_state_4_item

            # self.user_item_toDQN()  # later make it to batch
            #bool_ = self.make_reward(train_dict)
            # if i>=1:
            loss_user = self.train_user()
            loss_item = self.train_item()

            if loss_user != 0 and loss_item != 0:
                self.memory_item.memory = []
                self.memory_user.memory = []
                self.state_user = []
                self.state_item = []


            print(len(self.dictionary['user']), "length of dictionary", len(self.dictionary_neg['user']))
            if len(self.dictionary['user']) >= self.count_train:
                user = torch.tensor(self.dictionary['user'], dtype=int)  # , dtype=int)
                item = torch.tensor(self.dictionary['item'], dtype=int)  # , dtype=int)
                neg_item = torch.tensor(self.dictionary['neg_item'], dtype=int)  # , dtype=int)
                u_action = torch.tensor(self.dictionary['u_action'], dtype=int)  # , dtype=int)
                i_action = torch.tensor(self.dictionary['i_action'], dtype=int)  # , dtype=int)
                neg_action = torch.tensor(self.dictionary['neg_action'], dtype=int)  # , dtype=int)
                temp_result = [user, item, neg_item, u_action, i_action, neg_action]
                self.NGCF_train.append(temp_result)

            if len(self.dictionary_neg['user']) >= self.count_train:
                item = torch.tensor(self.dictionary_neg['user'], dtype=int)  # , dtype=int)
                user = torch.tensor(self.dictionary_neg['item'], dtype=int)  # , dtype=int)
                neg_item = torch.tensor(self.dictionary_neg['neg_user'], dtype=int)  # , dtype=int)
                u_action = torch.tensor(self.dictionary_neg['u_action'], dtype=int)  # , dtype=int)
                i_action = torch.tensor(self.dictionary_neg['i_action'], dtype=int)  # , dtype=int)
                neg_action = torch.tensor(self.dictionary_neg['neg_action'], dtype=int)  # , dtype=int)
                temp_result = [user, item, neg_item, u_action, i_action, neg_action]
                self.NGCF_train_neg.append(temp_result)
            #if bool_:
            # self.train_KGAT()
            # self.reset_for_dict()
            self.total_t += 1
            self.reset_count += 1





        #self.reset_for_dict()
        self.total_t=0
        #temp_result = np.array(temp_result)
        # self.reward.append(np.mean(temp_result).item())
        self.train_KGAT()
        self.reset_for_dict()
        if mean_reward_user != 0:
            self.total_reward_user.append(mean_reward_user.item() / user_count)
        if mean_reward_item != 0:
            self.total_reward_item.append(mean_reward_item.item() / item_count)
        # self.reset()


    def train_KGAT(self):
        self.NGCF.train()
        time1=self.imported.time()
        print("length of the KGAT_train is: ",len(self.NGCF_train))
        result=[]

        for j in range(10):  #100 나중에 hyperparameter로 바꾸기6
            for i in self.NGCF_train:
                if j==0:
                    result.append(len(i[0]))
                    #print(len(i[0]), len(np.unique(i[0])))
                self.optimizer.zero_grad()
                u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = self.NGCF(i[0],i[1],i[2],i[3],i[4],i[5], 0)
                # print(u_g_embeddings.shape, pos_i_g_embeddings.shape, neg_i_g_embeddings.shape)
                batch_loss, batch_mf_loss, batch_emb_loss = self.NGCF.create_bpr_loss(u_g_embeddings,
                                                                                  pos_i_g_embeddings,
                                                                                  neg_i_g_embeddings)
                batch_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            for i in self.NGCF_train_neg:
                self.optimizer.zero_grad()
                u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = self.NGCF(i[0],i[1],i[2],i[3],i[4],i[5], 1)
                batch_loss, batch_mf_loss, batch_emb_loss = self.NGCF.create_bpr_loss(u_g_embeddings,
                                                                                  pos_i_g_embeddings,
                                                                                  neg_i_g_embeddings)
                batch_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
        # print("loss of the KGAT is : ", j, "   ", batch_loss)
        time2=self.imported.time()
        #print(result)
        print("the time of train_KGAT is: ",time2- time1, len(self.NGCF_train_neg))
        self.NGCF_train=[]
        self.NGCF_train_neg = []


    def user_item_toDQN(self):
        hop_result = self.NGCF.make_hop()
        DQN_network = self.q_estimator_user

        user_embed = self.NGCF.embedding_dict['user_emb']
        q_value_user = DQN_network.predict_nograd(user_embed)
        best_actions = torch.argmax(torch.from_numpy(q_value_user), dim=1)
        best_actions = best_actions.numpy()
        reward_user = hop_result[best_actions, np.arange(self.NGCF.n_user)]

        DQN_network = self.q_estimator_item

        item_embed = self.NGCF.embedding_dict['item_emb']
        q_value_item = DQN_network.predict_nograd(item_embed)
        best_actions = torch.argmax(torch.from_numpy(q_value_item), dim=1)
        best_actions = best_actions.numpy()
        reward_item = hop_result[best_actions, np.arange(self.NGCF.n_user, self.NGCF.n_user+self.NGCF.n_item)]
        self.reward_user = reward_user
        self.reward_item = reward_item

        # all_embed = self.KGAT.entity_user_embed(g.ndata['id'])
        # #self.KGAT=self.KGAT.to('cpu')
        # hop_result = self.KGAT('make_hop',g).detach()
        # #self.KGAT=self.KGAT.to(self.device)
        # reward_for_all = g.ndata['reward']
        # DQN_network=self.q_estimator_item
        # state=all_embed[:self.KGAT.n_entities].reshape(self.KGAT.n_entities,-1)
        # q_values=DQN_network.predict_nograd(state)
        # best_actions=torch.argmax(torch.from_numpy(q_values),dim=1)
        # best_actions=best_actions.numpy()
        # reward_for_all[:self.KGAT.n_entities]=hop_result[best_actions,np.arange(self.KGAT.n_entities)]
        # DQN_network=self.q_estimator_user
        # state=all_embed[self.KGAT.n_entities:].reshape((len(all_embed)-self.KGAT.n_entities),-1)
        # q_values = DQN_network.predict_nograd(state)
        # best_actions = torch.argmax(torch.from_numpy(q_values),dim=1)
        # best_actions = best_actions.numpy()
        # reward_for_all[self.KGAT.n_entities:] = hop_result[best_actions, np.arange(self.KGAT.n_entities,len(all_embed))]
        # g.ndata['reward'] = reward_for_all

    def train_user(self):
        if len(self.memory_user.memory) < self.batch_size:
        #    print("length of state_user is: ", len(self.state_user))
            if self.deepcopy_count % 900 != 0:
                self.deepcopy_count += 1
            return 0
        #print("is it printed?", self.deepcopy_count)
        #print("memory sample size", len(self.memory_user.memory))
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory_user.sample()
        next_state_batch=next_state_batch.to(self.device)     ############
        next_state_batch = self.NGCF.embedding_dict['user_emb'][next_state_batch]
        next_state_batch = next_state_batch.reshape(-1, 64)
        q_values_next = self.q_estimator_user.predict_nograd(next_state_batch)  ###next_state_batch from sampling
        best_actions = np.argmax(q_values_next, axis=-1)
        q_values_next_target = self.target_estimator_user.predict_nograd(next_state_batch)
        target_batch = reward_batch+ done_batch * self.discount_factor * \
                       q_values_next_target[np.arange(self.batch_size), best_actions]#[np.arange(5), best_actions]
        target_batch=target_batch.to(torch.float32)
        state_batch = state_batch.to(self.device)  ###########
        state_batch = self.NGCF.embedding_dict['user_emb'][state_batch]
        loss = self.q_estimator_user.update(state_batch, action_batch, target_batch)
        #print("loss of the q function is : ", loss)
        if self.total_t==self.epsilon_decay-1:
            # print("saving the loss")
            loss_save.append(loss)
        if self.deepcopy_count % 900 == 0:#if self.train_t % 100 == 0:  #2000 should be hyperparamter---HS
            print("it is going to be changed")
            self.target_estimator_user = deepcopy(self.q_estimator_user)
            self.target_estimator_item = deepcopy(self.q_estimator_item)
        self.deepcopy_count+=1
        return loss

    def train_item(self):
        if len(self.memory_item.memory) < self.batch_size:
            return 0
        #print("item memory sample size", len(self.memory_item.memory))
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory_item.sample()
        next_state_batch = next_state_batch.to(self.device)
        next_state_batch = self.NGCF.embedding_dict['item_emb'][next_state_batch]
        next_state_batch = next_state_batch.reshape(-1, 64)
        q_values_next = self.q_estimator_item.predict_nograd(next_state_batch)  ###next_state_batch from sampling
        best_actions = np.argmax(q_values_next, axis=-1)
        q_values_next_target = self.target_estimator_item.predict_nograd(next_state_batch)
        target_batch = reward_batch + done_batch * self.discount_factor * \
                       q_values_next_target[np.arange(self.batch_size), best_actions]
        target_batch=target_batch.to(torch.float32)
        state_batch = state_batch.to(self.device)
        state_batch = self.NGCF.embedding_dict['item_emb'][state_batch]#torch.tensor(state_batch, device=self.device))
        loss = self.q_estimator_item.update(state_batch, action_batch, target_batch)
        return loss

    def reset_for_dict(self):
        # print(len(np.unique(self.interacted_user)), len(self.state_user), self.index_dict_add)
        self.dictionary = {'user': [], 'item': [], 'u_action': [], 'i_action': [], 'neg_item': [], 'neg_action': []}
        self.dictionary_neg = {'user': [], 'item': [], 'u_action': [], 'i_action': [], 'neg_user': [], 'neg_action': []}

        self.pair_user = []
        self.pair_pos = []
        self.pair_neg = []
        # self.state_user = []
        # self.state_item = []
        # self.action_user = []
        # self.action_item = []

        ####
        #self.action = []
        #self.user_index = []
        self.user_count = 0
        self.state_user = []
        self.state_item = []
        self.action_user = []
        self.action_item = []
        self.reward_user = []
        self.reward_item = []
        self.reward = []
        self.base_line_reward_user= []
        self.base_line_reward_item = []

        # self.interacted_user = np.array([])
        # self.interacted_item = np.array([])
        self.index_dict_add = 0
        self.memory_user.memory = []
        self.memory_item.memory = []


    def dictionary_add(self, train_dict):  # 시간 줄여보기--HS
        #print("in dictionary_add : ", self.index_dict_add, len(self.state_user))
        count = 0
        for i in self.state_user:
            np_usr = np.array(train_dict[i]) #train.txt와 관련되어있는 아이템
            np_itm = np.array(self.state_item)
            pos_itm = np.intersect1d(np_usr, np_itm)
            neg_itm = np.setdiff1d(np_itm, np_usr)
            #print("item list ", np_itm, pos_itm)
            if len(neg_itm) != 0:
                for j in pos_itm:
                    j_index = np.where(self.state_item == j)
                    neg_itm_candidate = np.random.choice(neg_itm, 1)
                    neg_index = np.where(self.state_item == neg_itm_candidate)
                    self.dictionary['user'].append(i)
                    self.dictionary['item'].append(j)
                    self.dictionary['u_action'].append(self.action_user[count])
                    self.dictionary['i_action'].append(self.action_item[j_index[0][0]])
                    self.dictionary['neg_item'].append(neg_itm_candidate[0])
                    self.dictionary['neg_action'].append(self.action_item[neg_index[0][0]])
        if len(self.dictionary['user']) >= self.count_train:
            user = torch.tensor(self.dictionary['user'], dtype=int)  # , dtype=int)
            item = torch.tensor(self.dictionary['item'],dtype=int)#, dtype=int)
            neg_item = torch.tensor(self.dictionary['neg_item'],dtype=int)#, dtype=int)
            u_action = torch.tensor(self.dictionary['u_action'],dtype=int)#, dtype=int)
            i_action = torch.tensor(self.dictionary['i_action'],dtype=int)#, dtype=int)
            neg_action = torch.tensor(self.dictionary['neg_action'],dtype=int)#, dtype=int)
            temp_result=[user,item,neg_item,u_action,i_action,neg_action]
            self.NGCF_train.append(temp_result)
            return True
        return False



    def make_reward(self, train_dict):  # 우선은 데이터 하나당 돌려서 g.ndata['reward'] 업데이트 하는 방향
        '''
        self.action_user(item) --> get action and get state from self.state_user(item)
        state information to KGAT.reward
        나중에 DQN에서 KGAT 업데이트 하는 코드 더 필요함-HS
        '''
        user_reward = torch.tensor(self.reward_user, device=self.device)
        item_reward = torch.tensor(self.reward_item, device=self.device)
        if len(user_reward) == 0:
            print("0 value of user")

        item_reward = torch.transpose(item_reward, 0, 1)
        reward_score = torch.matmul(user_reward, item_reward)
        bool_ = self.dictionary_add(train_dict)
        for i in range(len(self.state_user)):
            user_index = self.state_user[i]
            user_score = reward_score[user_index]
            reward_score_sorted, index_cosine =torch.sort(user_score, descending=True)
            reward = self.get_reward(i, user_index, user_score, train_dict, index_cosine)
            if i == len(self.state_user) - 1:
                done = False
                transition_user = to_memory(self.state_user[i], self.action_user[i], reward, 0, 0)
                transition_item = to_memory(self.state_item[i], self.action_item[i], reward, 0, 0)
            else :
                done = True
                transition_user = to_memory(self.state_user[i+self.index_dict_add], self.action_user[i+self.index_dict_add], reward, self.state_user[i+self.index_dict_add + 1],
                                            1)
                transition_item = to_memory(self.state_item[i+self.index_dict_add], self.action_item[i+self.index_dict_add], reward, self.state_item[i+self.index_dict_add + 1],
                                            1)
            self.memory_user.append(transition_user)
            self.memory_item.append(transition_item)
        #if bool_:
        #    self.reset_for_dict()
        return bool_





    def get_reward(self,indexing, user, cosine_simil,train_dict, index_cosine):
        pos_item_candidate = np.array(train_dict[user])
        neg_item_candidate = np.setdiff1d(np.array(np.arange(self.NGCF.n_item)), pos_item_candidate)
        pos_item = np.random.choice(pos_item_candidate, 3)
        neg_item = np.random.choice(neg_item_candidate, 3)
        reward = torch.sum(cosine_simil[pos_item]) - torch.sum(cosine_simil[neg_item])
        average_reward = self.get_baseline(reward)
        return reward - average_reward


        ##############


        items=train_dict[user]
        ideal=0
        index=0
        if len(items)>=self.K:
            index=self.K
        else:
            index=len(items)
        for i in range(index):
            dcg_score=1/(np.log2(i+2))
            ideal +=dcg_score
        real=0
        for i in range(self.K):
            dcg_score=(np.exp2(cosine_simil[i].item())-1)/(np.log2(i+2))
            real+=dcg_score

        average_reward = self.get_baseline(real/ideal)
        reward = (real/ideal-average_reward)
        # if rank<=self.K and self.state_item[indexing] in items:
        #     reward += 30-rank
        #    print("wow it is here")
        return reward#torch.tensor(reward)

        rank = rank.cpu().detach().numpy()
        if rank > self.K:  # 20 should be hyperparameter --HS
               reward=real/ideal
        else:
            relevance_score = cosine_simil[indexing,rank]  # 10 should be hyperparameter --HS
            relevance_score=relevance_score.cpu().detach().numpy()
            g_score = np.exp2(relevance_score)-1
            reward = g_score / (np.log2(rank+2))
            reward = reward+ real/ideal
        temp_ndcg[self.state_user[indexing] - self.KGAT.n_entities] = torch.tensor(reward)
        g.ndata['ndcg']=temp_ndcg
        return torch.tensor(real/ideal)



    def get_baseline_user(self,number):
        self.base_line_reward_user.append(number)
        if len(self.base_line_reward_user)<= self.base_line_number:
            return sum(self.base_line_reward_user)/len(self.base_line_reward_user)
        else:
            self.base_line_reward_user.pop(0)
            return sum(self.base_line_reward_user)/len(self.base_line_reward_user)

    def get_baseline_item(self,number):
        self.base_line_reward_item.append(number)
        if len(self.base_line_reward_item)<= self.base_line_number:
            return sum(self.base_line_reward_item)/len(self.base_line_reward_item)
        else:
            self.base_line_reward_item.pop(0)
            return sum(self.base_line_reward_item)/len(self.base_line_reward_item)


    def next_state_user(self, idx, action, item_idx):
        # aggregate_list = self.KGAT.adj_list[action - 2]  # -2 need to be changed(hyperparameter)--HS
        # index_aggregate = aggregate_list[idx]


        # if action == 0:
        #     next_candidate = self.NGCF.plain_adj[item_idx]
        #     next_candidate = next_candidate.toarray()
        #     # next_candidate = next_candidate[:self.NGCF.n_user]
        #     next_candidate = np.where(next_candidate>=1)[1]
        #     next_candidate = next_candidate[next_candidate<self.NGCF.n_user]
        #     # print(next_candidate)
        #     next_state = np.random.choice(next_candidate)
        #     return next_state, True

        action = action+1
        for_check = np.zeros(self.NGCF.n_user + self.NGCF.n_item)
        if action % 2 == 1:
            action += 1
        # action = 1
        temp = self.NGCF.plain_adj[idx] @ self.NGCF.plain_adj
        for i in range(action-1):
            if i % 2 == 1:
                for_check = np.copy(temp.toarray())
            temp = temp @ self.NGCF.plain_adj
        index_aggregate = temp
        index_aggregat_np = index_aggregate.toarray()

        neighbor = np.argwhere(index_aggregat_np >= 1)
        neighbor_for_check = np.where(for_check >=1)[0]
        neighbor_for_check = neighbor_for_check
        neighbor = neighbor[:, 1]
        neighbor_for_check_u = neighbor_for_check[neighbor_for_check < self.NGCF.n_user]
        neighbor_user = neighbor[neighbor < self.NGCF.n_user]
        neighbor_user = np.setdiff1d(neighbor_user, neighbor_for_check_u)
        #print("next_state user length: ", len(neighbor_user))
        if len(neighbor_user) == 0:
            print("why nobody")
            return 0, False

        next_state = np.random.choice(neighbor_user)
        return next_state, True



        # temp = self.KGAT.adj_list[idx] @ self.KGAT.adj_list
        # for i in range(action-3):
        #     temp = temp @ self.KGAT.adj_list
        # index_aggregate = temp
        # index_aggregate_np = index_aggregate.toarray()
        # neighbor = np.argwhere(index_aggregate_np >= 1)
        # neighbor = neighbor[:, 1]
        # neighbor_user = neighbor[neighbor >= self.KGAT.n_entities]
        # neighbor_user = np.setdiff1d(neighbor_user, self.interacted_user)
        # next_state_bool = True
        # if len(neighbor_user) == 0:
        #     next_state_bool = False
        #     return 0, next_state_bool
        # else:
        #     next_state = np.random.choice(neighbor_user)
        #     self.interacted_user = np.append(self.interacted_user, next_state)
        #
        #     return next_state, next_state_bool

    def next_state_item(self, idx, action, user_idx):
        # aggregate_list = self.KGAT.adj_list[action - 2]  # -2 need to be changed(hyperparameter)--HS
        # index_aggregate = aggregate_list[idx]
        # if action == 0:
        #     next_candidate = self.NGCF.plain_adj[user_idx]
        #     next_candidate = next_candidate.toarray()
        #     # next_candidate = next_candidate[self.NGCF.n_user:]
        #     next_candidate = np.where(next_candidate>=1)[1]
        #     # print(next_candidate, "next_candidate of action 0")
        #     next_candidate = next_candidate[next_candidate>=self.NGCF.n_user]
        #     next_state = np.random.choice(next_candidate)
        #     return next_state-self.NGCF.n_user, True

        action = action + 1
        if action % 2 == 1:
            action += 1
        action = 1
        temp = self.NGCF.plain_adj[idx+self.NGCF.n_user] @ self.NGCF.plain_adj
        for_check = np.zeros(self.NGCF.n_user + self.NGCF.n_item)
        for i in range(action-1):
            if i % 2 == 1:
                for_check = np.copy(temp.toarray())
            temp = temp @ self.NGCF.plain_adj
        index_aggregate = temp
        index_aggregat_np = index_aggregate.toarray()
        neighbor = np.argwhere(index_aggregat_np >= 1)
        neighbor = neighbor[:, 1]
        neighbor_for_check = np.where(for_check >= 1)[0]
        # neighbor_for_check = neighbor_for_check[:, 0]
        #print("neighbor item: ",neighbor)
        neighbor_for_check_i = neighbor_for_check[neighbor_for_check >= self.NGCF.n_user]
        neighbor_item = neighbor[neighbor >= self.NGCF.n_user]
        neighbor_item = np.setdiff1d(neighbor_item, neighbor_for_check_i)
        #print("next_state item length: ", len(neighbor_item), idx)
        if len(neighbor_item) == 0:
            print(neighbor_for_check_i, neighbor_item)
            print('why nobody item')
            return 0, False
        #print("min value of item is : ", min(neighbor_item), self.NGCF.n_user)
        next_state = np.random.choice(neighbor_item)

        return next_state-self.NGCF.n_user, True

    def get_embed(self, g, embed_vec):
        '''
        getting embedding vector of all the nodes(user,item) in the graph
        '''
        return embed_vec(g.ndata['id'])

    def action_user_func(self, state):  ##need to normalize the state-HS
        '''
        getting action(number of hops) from user from the result of Q network with some epsilon values
        '''
        epsilon = self.epsilons[min(self.total_t, self.epsilon_decay - 1)]
        a = np.ones(self.action_num, dtype=float) * epsilon / self.action_num
        state=torch.unsqueeze(state,0)
        q_values = self.q_estimator_user.predict_nograd(state)[0]  # why zero? -HS
        best_action = np.argmax(q_values)
        #print(q_values.shape, a.shape)
        a[best_action] += (1.0 - epsilon)
        action = np.random.choice(np.arange(len(a)), p=a)
        self.user_index.append(self.user_count)
        self.user_count += 1
        return action

    def action_item_func(self, state):  # need to normalize the state-HS
        '''
        getting action(number of hops) from item from the result of Q network with some epsilon values
        '''
        epsilon = self.epsilons[min(self.total_t, self.epsilon_decay - 1)]
        a = np.ones(self.action_num, dtype=float) * epsilon / self.action_num
        state=torch.unsqueeze(state,0)
        q_values = self.q_estimator_item.predict_nograd(state)[0]  # why zero? -HS
        best_action = np.argmax(q_values)
        a[best_action] += (1.0 - epsilon)
        action = np.random.choice(np.arange(len(a)), p=a)
        self.user_count += 1
        return action 

    def random_pick(self, train_dict):
        # number_of_user = self.KGAT.n_users
        # number_of_entity = self.KGAT.n_entities
        # number_of_item = self.KGAT.n_items
        # item = np.random.choice(number_of_item, 1)
        # user = np.random.choice(number_of_user, 1)
        #user_candidate = np.setdiff1d(self.kgat_user, self.interacted_user)
        user = np.random.choice(self.kgat_user, 1)
        #self.interacted_user = np.append(self.interacted_user, user[0])
        #print(item, train_dict.keys())
        item_candidate = np.array(train_dict[user[0]])
        item = np.random.choice(item_candidate, 1)
        #self.interacted_item = np.append(self.interacted_item, item[0])]
        self.user = user[0]
        self.item = item[0]
        #return item[0], user[0]

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
        self.reward_user = []
        self.reward_item = []
        self.reward = []

    def get_reward_from_KGAT(self, g):
        '''
        getting all the aggregated_embedding with the
        '''
        result = self.KGAT.reward(self, 'reward', g, self.action_list)
        self.reward_list = result
        self.concat_result = []
        for i in range(len(self.reward_list)):
            self.concat_result += self.reward_list[i]
        return result

    def make_final(self):
        '''
        from the self.reward_list--> make it to final_item, final_user, final_reward
        '''
        self.user_index = np.array(self.user_index)
        self.action = np.array(self.action)
        self.concat_result = np.array(self.concat_result)
        self.final_user = self.concat_result[self.user_index]
        self.final_item = np.setdiff1d(self.concat_result, self.final_user)
        self.final_user = self.final_user.reshape(1, -1)
        self.final_item = self.final_item.reshape(1, -1)
        self.final_ui_pair = np.matmul(self.final_user.transpose(), self.final_item)
    def get_saved_loss(self):
        return np.array(loss_save)













