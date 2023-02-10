'''
Created on March 24, 2020

@author: Tinglin Huang (huangtinglin@outlook.com)
'''

import torch
import torch.optim as optim

from NGCF import NGCF
from utility.helper import *
from utility.batch_test import *

import numpy as np
import Double_DQN
import warnings
warnings.filterwarnings('ignore')
from time import time
from scipy import sparse


if __name__ == '__main__':

    args.device = torch.device('cuda:' + str(args.gpu_id))
    a = time()
    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()
    #norm_adj = norm_adj.toarray()
    #print(np.sum(norm_adj, axis=1), norm_adj[0][:20])
    #exit(0)
    print(plain_adj.shape)
    plain_adj = plain_adj + sparse.eye(len(plain_adj.toarray()))
    #plain_adj = sparse.csr_matrix(plain_adj)
    print(time()-a, "making adj matrix")
    #temp_adj = plain_adj[0].toarray()
    #print(np.argwhere(temp_adj >=1)[:, 1])
    #exit(0)

    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)
    a = time()
    model = NGCF(data_generator.n_users,
                 data_generator.n_items,
                 norm_adj,
                 args, plain_adj).to(args.device)

    model.load_state_dict(torch.load('./model/ml-1m99.pkl'))
    print(time()-a, "NGCF preparing time")
    t0 = time()
    """
    *********************************************************
    Train.
    """
    cur_best_pre_0, stopping_step = 0, 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    a = time()
    DDQN = Double_DQN.Double_DQN(model, optimizer, 4, 0.001, 64, [32, 64, 32, 16], args.device, 0.8, 1000,
                      5,  # TODO changed lr 0.00001 to 0.001
                      10, 100, 10, 10, 20, 10)  # need to c
    print(time()-a, "DDQN preparing time")
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    ### 강화학습 부분
    for i in range(0):
        if i % 100 == 0:
            DDQN.update_target()
        if i % 100 == 0:
            DDQN.random_pick(data_generator.train_items)
        a = time()
        DDQN.learn(data_generator.train_items)

        print(time()-a, "time of one learn")


   ###action 확정짓는거 리스트화 해서 코드로 만들기
    #reward_list = DDQN.reward()
    #np.save('./reward.pt', np.array(reward_list))
    #action_for_user = DDQN.q_estimator_user.predict_nograd(
    #    model.embedding_dict['user_emb'][
    #        torch.tensor(np.arange(0, data_generator.n_users), device=args.device).reshape(
    #            data_generator.n_users, -1)])
    #action_for_user = torch.argmax(torch.tensor(action_for_user), dim=1)
    #action_for_item = DDQN.q_estimator_item.predict_nograd(
    #    model.embedding_dict['item_emb'][torch.tensor(np.arange(data_generator.n_items), device=args.device).reshape(data_generator.n_items, -1)])
    #action_for_item = torch.argmax(torch.tensor(action_for_item), dim=1)
    

    #action_for_user[action_for_user !=2] = 2
    #action_for_item[action_for_item !=2] = 2
    #action_for_user = torch.load(str(args.batch_size)+args.dataset+'user_index.pt')
    #action_for_item = torch.load(str(args.batch_size)+args.dataset+'item_index.pt')
    #print(torch.bincount(action_for_user))
    #print(torch.bincount(action_for_item))
    #torch.save(action_for_user, str(args.batch_size)+args.dataset+'user_index3.pt')
    #torch.save(action_for_item, str(args.batch_size)+args.dataset+"item_index3.pt")
    ####
    #model = NGCF(data_generator.n_users,
    #             data_generator.n_items,
    #             norm_adj,
    #             args, plain_adj).to(args.device)

    t0 = time()
    """
    *********************************************************
    Train.
    """
    cur_best_pre_0, stopping_step = 0, 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    a = model.make_hop()
    initial_user = a[:, 100]
    print(initial_user.shape)
    #cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    #cos2 = nn.CosineSimilarity(dim=-1, eps=1e-6)
    b = torch.pow(initial_user, 2)
    c = torch.sum(b, dim=1).reshape(5, 1)
    c = torch.sqrt(c)
    #initial_user = initial_user / c
    print(initial_user.shape)
    random_item = a[:, data_generator.train_items[100]]
    random_item = a[:, 11]
    b = torch.pow(random_item, 2)
    b = torch.sum(b, dim=1).reshape(5, 1)
    b = torch.sqrt(b)
    #random_item = random_item / b
    print(torch.matmul(initial_user, random_item.T))



    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)
    print(torch.bincount(action_for_item), torch.bincount(action_for_user))
