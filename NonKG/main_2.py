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


if __name__ == '__main__':

    args.device = torch.device('cuda:' + str(args.gpu_id))

    plain_adj, norm_adj, mean_adj = data_generator.get_adj_mat()

    args.node_dropout = eval(args.node_dropout)
    args.mess_dropout = eval(args.mess_dropout)
    a = time()
    model = NGCF(data_generator.n_users,
                 data_generator.n_items,
                 norm_adj,
                 args, plain_adj).to(args.device)
    print(time()-a, "NGCF preparing time")
    t0 = time()
    """
    *********************************************************
    Train.
    """

    cur_best_pre_0, stopping_step = 0, 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    a =time()
    DDQN = Double_DQN.Double_DQN(model, optimizer, 4, 0.001, 64, [64, 128, 64, 32, 16], args.device, 0.98, 300,
                      30,  # TODO changed lr 0.00001 to 0.001
                      30, 100, 10, 32, 20, 50)  # need to c
    print(time()-a, "DDQN preparing time")
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    ### 강화학습 부분
   #  for i in range(10):
   #      if i % 1 == 0:
   #          DDQN.random_pick(data_generator.train_items)
   #      DDQN.learn(data_generator.train_items)
   #
   #
   # ###action 확정짓는거 리스트화 해서 코드로 만들기
   #  action_for_user = DDQN.q_estimator_user.predict_nograd(
   #      model.embedding_dict['user_emb'][
   #          torch.tensor(np.arange(0, data_generator.n_users), device=args.device).reshape(
   #              data_generator.n_users, -1)])
   #  action_for_user = torch.argmax(torch.tensor(action_for_user), dim=1)
   #  action_for_item = DDQN.q_estimator_item.predict_nograd(
   #      model.embedding_dict['item_emb'][torch.tensor(np.arange(data_generator.n_items), device=args.device).reshape(data_generator.n_items, -1)])
   #  action_for_item = torch.argmax(torch.tensor(action_for_item), dim=1)

    action_for_user = torch.load(str(args.batch_size)+args.dataset+'user_index.pt')
    action_for_item = torch.load(str(args.batch_size) + args.dataset + 'item_index.pt')
    ###
    model = NGCF(data_generator.n_users,
                 data_generator.n_items,
                 norm_adj,
                 args, plain_adj).to(args.device)
    model.load_state_dict(torch.load('./model/gowalla49.pkl'))
    t0 = time()
    """
    *********************************************************
    Train.
    """
    cur_best_pre_0, stopping_step = 0, 0
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epoch):
        t1 = time()
        loss, mf_loss, emb_loss = 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        for idx in range(n_batch):
            users, pos_items, neg_items = data_generator.sample()
            u_a = action_for_user[users]
            p_a = action_for_item[pos_items]
            n_a = action_for_item[neg_items]
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings = model(users,
                                                                           pos_items,
                                                                           neg_items,
                                                                           u_a, p_a, n_a,
                                                                           drop_flag=args.node_dropout_flag)

            batch_loss, batch_mf_loss, batch_emb_loss = model.create_bpr_loss(u_g_embeddings,
                                                                              pos_i_g_embeddings,
                                                                              neg_i_g_embeddings)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            mf_loss += batch_mf_loss
            emb_loss += batch_emb_loss

        if (epoch + 1) % 5 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss)
                print(perf_str)
            continue

        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret = test(model, users_to_test,action_for_user, action_for_item, drop_flag=False)

        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f], recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, mf_loss, emb_loss, ret['recall'][0], ret['recall'][-1],
                        ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1])
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=5)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
            torch.save(model.state_dict(), args.weights_path + str(epoch) + '.pkl')
            print('save the weights in path: ', args.weights_path + str(epoch) + '.pkl')

    users_to_test = list(data_generator.test_set.keys())
    ret = test(model, users_to_test, action_for_user, action_for_item, drop_flag=False)

    t3 = time()

    loss_loger.append(0)
    rec_loger.append(ret['recall'])
    pre_loger.append(ret['precision'])
    ndcg_loger.append(ret['ndcg'])
    hit_loger.append(ret['hit_ratio'])
    
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
