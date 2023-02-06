import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import random
import logging
import argparse
from time import time
from model.Double_DQN import Double_DQN


import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist


from model.KGAT import KGAT
from utility.parser_DPAO import *
from utility.log_helper import *
from utility.metrics import *
from utility.model_helper import *
from utility.loader_DPAO import DataLoaderKGAT
import pickle


def evaluate(model, train_graph, train_user_dict, test_user_dict, user_ids_batches, item_ids, K, action_for_user, action_for_item, data):
    model.eval()

    with torch.no_grad():
        att = model.compute_attention(train_graph)
    train_graph.edata['att'] = att

    n_users = len(test_user_dict.keys())
    item_ids_batch = item_ids.cpu().numpy()

    cf_scores = []
    precision = []
    recall = []
    ndcg = []
    ndcg_truncate = []

    with torch.no_grad():
        for user_ids_batch in user_ids_batches:
            user_action = action_for_user[user_ids_batch-data.n_entities]
            item_action = action_for_item[item_ids]
            cf_scores_batch = model('predict', train_graph, user_ids_batch, item_ids, user_action, item_action)       # (n_batch_users, n_eval_items)

            cf_scores_batch = cf_scores_batch.cpu()
            user_ids_batch = user_ids_batch.cpu().numpy()
            precision_batch, recall_batch, ndcg_batch, ndcg_truncate_batch = calc_metrics_at_k(cf_scores_batch, train_user_dict, test_user_dict, user_ids_batch, item_ids_batch, K)

            cf_scores.append(cf_scores_batch.numpy())
            precision.append(precision_batch)
            recall.append(recall_batch)
            ndcg.append(ndcg_batch)
            ndcg_truncate.append(ndcg_truncate_batch)

    cf_scores = np.concatenate(cf_scores, axis=0)
    precision_k = sum(np.concatenate(precision)) / n_users
    recall_k = sum(np.concatenate(recall)) / n_users
    ndcg_k = sum(np.concatenate(ndcg)) / n_users
    print(sum(np.concatenate(ndcg_truncate)) / n_users)
    return cf_scores, precision_k, recall_k, ndcg_k


def train(args):
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    # GPU / CPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # load data
    # with open('0909_yelp_new_data.pkl', 'wb') as outp:
    #    data = DataLoaderKGAT(args, logging)
    #    pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)
    # exit(0)

    with open('0909_yelp_new_data.pkl', 'rb') as inp:
       data = pickle.load(inp)

    #data = DataLoaderKGAT(args, logging)

    if args.use_pretrain == 1:
        user_pre_embed = torch.tensor(data.user_pre_embed)
        item_pre_embed = torch.tensor(data.item_pre_embed)
    else:
        user_pre_embed, item_pre_embed = None, None

    user_ids = list(data.test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + args.test_batch_size] for i in range(0, len(user_ids), args.test_batch_size)]
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]
    if use_cuda:
        user_ids_batches = [d.to(device) for d in user_ids_batches]

    item_ids = torch.arange(data.n_items, dtype=torch.long)
    if use_cuda:
        item_ids = item_ids.to(device)

    # construct model & optimizer

    model = KGAT(args, data.n_users, data.n_items, data.n_entities, data.n_relations, user_pre_embed, item_pre_embed)
    model.to(device)
    # if n_gpu > 1:
    #     model = nn.parallel.DistributedDataParallel(model)
    logging.info(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    DDQN = Double_DQN(model, optimizer, 4, 0.0001, 64, [32, 64, 128, 32, 16], device, 0.9, 1000,
                      10,  # TODO changed lr 0.00001 to 0.001
                      10, 100, 600, 20, 20, 5)  # need to c
    # move graph data to GPU
    train_graph = data.train_graph
    train_nodes = torch.LongTensor(train_graph.ndata['id'])
    train_edges = torch.LongTensor(train_graph.edata['type'])
    if use_cuda:
        train_nodes = train_nodes.to(device)
        train_edges = train_edges.to(device)
        train_graph = train_graph.to(device)
    train_graph.ndata['id'] = train_nodes
    train_graph.edata['type'] = train_edges

    test_graph = data.test_graph
    test_nodes = torch.LongTensor(test_graph.ndata['id'])
    test_edges = torch.LongTensor(test_graph.edata['type'])
    if use_cuda:
        test_nodes = test_nodes.to(device)
        test_edges = test_edges.to(device)
        test_graph = test_graph.to(device)
    test_graph.ndata['id'] = test_nodes
    test_graph.edata['type'] = test_edges
    model.hop(train_graph)
    for i in range(200): #need to be parameter changed --HS
        with torch.no_grad():
            att = model('calc_att', train_graph)
        train_graph.edata['att'] = att
        model.train()
        print("timestamp: ", i)

        #model.train()
        #
        if i % 20 == 0:
            DDQN.update_target()

        DDQN.learn(train_graph, data.train_user_dict, [], model, 520)

        with torch.no_grad():
            if i % 10  == 0:
                action_for_user = DDQN.q_estimator_user.predict_nograd(
                    model.entity_user_embed(
                        torch.tensor(np.arange(data.n_entities, data.n_entities + data.n_users), device=device).reshape(
                            data.n_users, -1)))
                action_for_user = torch.argmax(torch.tensor(action_for_user), dim=1)
                action_for_item = DDQN.q_estimator_item.predict_nograd(
                    model.entity_user_embed(torch.tensor(np.arange(data.n_items), device=device).reshape(data.n_items, -1)))
                action_for_item = torch.argmax(torch.tensor(action_for_item), dim=1)
                _, precision, recall, ndcg = evaluate(model, train_graph, data.train_user_dict, data.test_user_dict,
                                                      user_ids_batches, item_ids, args.K, action_for_user, action_for_item, data)
                logging.info('test_result: Precision {:.4f} Recall {:.4f} NDCG {:.4f}'.format(precision, recall, ndcg))
                print(torch.bincount(action_for_user), torch.bincount(action_for_item), i)


    # initialize metrics
    best_epoch = -1
    epoch_list = []
    precision_list = []
    recall_list = []
    ndcg_list = []

    # # action_for_user = torch.load('./99' + args.data_name + 'user_index.pt')
    # # action_for_item = torch.load('./99' + args.data_name + 'item_index.pt')
    # # action_for_user = torch.randint(0, 4, action_for_user.shape)
    # # action_for_item = torch.randint(0, 4, action_for_item.shape)
    # torch.save(action_for_user, './' + str(i) + args.data_name + 'user_index.pt')
    # torch.save(action_for_item, './' + str(i) + args.data_name + 'item_index.pt')
    # print(torch.bincount(action_for_user))
    # print(torch.bincount(action_for_item))
    # train model
    for epoch in range(args.n_epoch + 1):
        time0 = time()
        model.train()

        # update attention scores
        with torch.no_grad():
            att = model('calc_att', train_graph)
        train_graph.edata['att'] = att
#        logging.info('Update attention scores: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time0))

        # train cf
        time1 = time()
        cf_total_loss = 0
        n_cf_batch = data.n_cf_train // data.cf_batch_size + 1

        for iter in range(1, n_cf_batch + 1):
            time2 = time()
            cf_batch_user, cf_batch_pos_item, cf_batch_neg_item = data.generate_cf_batch(data.train_user_dict)
            if use_cuda:
                cf_batch_user = cf_batch_user.to(device)
                cf_batch_pos_item = cf_batch_pos_item.to(device)
                cf_batch_neg_item = cf_batch_neg_item.to(device)
            action_user = action_for_user[cf_batch_user-data.n_entities]
            action_pos = action_for_item[cf_batch_pos_item]
            action_neg = action_for_item[cf_batch_neg_item]
            action_user = action_user.to(device)
            action_pos = action_pos.to(device)
            action_neg = action_neg.to(device)
            cf_batch_loss = model('calc_cf_loss', train_graph, cf_batch_user, cf_batch_pos_item, cf_batch_neg_item, action_user, action_pos, action_neg)

            cf_batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            cf_total_loss += cf_batch_loss.item()

#            if (iter % args.cf_print_every) == 0:
#                logging.info('CF Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(epoch, iter, n_cf_batch, time() - time2, cf_batch_loss.item(), cf_total_loss / iter))
#        logging.info('CF Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(epoch, n_cf_batch, time() - time1, cf_total_loss / n_cf_batch))

        # train kg
        time1 = time()
        kg_total_loss = 0
        n_kg_batch = data.n_kg_train // data.kg_batch_size + 1


        logging.info('CF + KG Training: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time0))

        # evaluate cf
        if (epoch % args.evaluate_every) == 0:
            time1 = time()
            _, precision, recall, ndcg = evaluate(model, train_graph, data.train_user_dict, data.test_user_dict, user_ids_batches, item_ids, args.K, action_for_user, action_for_item, data)
            logging.info('CF Evaluation: Epoch {:04d} | Total Time {:.1f}s | Precision {:.4f} Recall {:.4f} NDCG {:.4f}'.format(epoch, time() - time1, precision, recall, ndcg))

            epoch_list.append(epoch)
            precision_list.append(precision)
            recall_list.append(recall)
            ndcg_list.append(ndcg)
            best_recall, should_stop = early_stopping(recall_list, args.stopping_steps)

            if should_stop:
                break

            if recall_list.index(best_recall) == len(recall_list) - 1:
                save_model(model, args.save_dir, epoch, best_epoch)
                logging.info('Save model on epoch {:04d}!'.format(epoch))
                best_epoch = epoch

    # save model
    save_model(model, args.save_dir, epoch)

    # save metrics
    _, precision, recall, ndcg = evaluate(model, train_graph, data.train_user_dict, data.test_user_dict, user_ids_batches, item_ids, args.K, action_for_user, action_for_item, data)
    logging.info('Final CF Evaluation: Precision {:.4f} Recall {:.4f} NDCG {:.4f}'.format(precision, recall, ndcg))

    epoch_list.append(epoch)
    precision_list.append(precision)
    recall_list.append(recall)
    ndcg_list.append(ndcg)

    metrics = pd.DataFrame([epoch_list, precision_list, recall_list, ndcg_list]).transpose()
    metrics.columns = ['epoch_idx', 'precision@{}'.format(args.K), 'recall@{}'.format(args.K), 'ndcg@{}'.format(args.K)]
    metrics.to_csv(args.save_dir + '/metrics.tsv', sep='\t', index=False)


def predict(args):
    # GPU / CPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # load data
    data = DataLoaderKGAT(args, logging)

    user_ids = list(data.test_user_dict.keys())
    user_ids_batches = [user_ids[i: i + args.test_batch_size] for i in range(0, len(user_ids), args.test_batch_size)]
    user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]
    if use_cuda:
        user_ids_batches = [d.to(device) for d in user_ids_batches]

    item_ids = torch.arange(data.n_items, dtype=torch.long)
    if use_cuda:
        item_ids = item_ids.to(device)

    # load model
    model = KGAT(args, data.n_users, data.n_entities, data.n_relations)
    model = load_model(model, args.pretrain_model_path)
    model.to(device)
    # if n_gpu > 1:
    #     model = nn.parallel.DistributedDataParallel(model)

    # move graph data to GPU
    train_graph = data.train_graph
    train_nodes = torch.LongTensor(train_graph.ndata['id'])
    train_edges = torch.LongTensor(train_graph.edata['type'])
    if use_cuda:
        train_nodes = train_nodes.to(device)
        train_edges = train_edges.to(device)
    train_graph.ndata['id'] = train_nodes
    train_graph.edata['type'] = train_edges

    test_graph = data.test_graph
    test_nodes = torch.LongTensor(test_graph.ndata['id'])
    test_edges = torch.LongTensor(test_graph.edata['type'])
    if use_cuda:
        test_nodes = test_nodes.to(device)
        test_edges = test_edges.to(device)
    test_graph.ndata['id'] = test_nodes
    test_graph.edata['type'] = test_edges

    # predict
    cf_scores, precision, recall, ndcg = evaluate(model, train_graph, data.train_user_dict, data.test_user_dict, user_ids_batches, item_ids, args.K)
    np.save(args.save_dir + 'cf_scores.npy', cf_scores)
    print('CF Evaluation: Precision {:.4f} Recall {:.4f} NDCG {:.4f}'.format(precision, recall, ndcg))



if __name__ == '__main__':
    args = parse_kgat_args()
    train(args)
    # predict(args)






