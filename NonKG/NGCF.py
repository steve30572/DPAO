'''
Created on March 24, 2020

@author: Tinglin Huang (huangtinglin@outlook.com)
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NGCF(nn.Module):
    def __init__(self, n_user, n_item, norm_adj, args, plain_adj):
        super(NGCF, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.device = args.device
        self.emb_size = args.embed_size
        self.batch_size = args.batch_size
        self.node_dropout = args.node_dropout[0]
        self.mess_dropout = args.mess_dropout
        self.batch_size = args.batch_size

        self.norm_adj = norm_adj
        self.plain_adj = plain_adj
        self.layers = eval(args.layer_size)
        self.decay = eval(args.regs)[0]

        """
        *********************************************************
        Init the weight of user-item.
        """
        self.embedding_dict, self.weight_dict = self.init_weight()

        """
        *********************************************************
        Get sparse adj.
        """
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.norm_adj).to(self.device)

    def init_weight(self):
        # xavier init
        initializer = nn.init.xavier_uniform_

        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_user,
                                                 self.emb_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_item,
                                                 self.emb_size)))
        })

        #nn.init.normal_(embedding_dict['user_emb'], std=0.1)
        #nn.init.normal_(embedding_dict['item_emb'], std=0.1)
        initializer(embedding_dict['user_emb'])
        initializer(embedding_dict['item_emb'])

        weight_dict = nn.ParameterDict()
        layers = [self.emb_size] + self.layers
        for k in range(len(self.layers)):
            weight_dict.update({'W_gc_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_gc_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

            weight_dict.update({'W_bi_%d'%k: nn.Parameter(initializer(torch.empty(layers[k],
                                                                      layers[k+1])))})
            weight_dict.update({'b_bi_%d'%k: nn.Parameter(initializer(torch.empty(1, layers[k+1])))})

        return embedding_dict, weight_dict

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def sparse_dropout(self, x, rate, noise_shape):
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def create_bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        maxi = nn.LogSigmoid()(pos_scores - neg_scores)

        mf_loss = -1 * torch.mean(maxi)

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / self.batch_size
        emb_loss = 0

        return mf_loss + emb_loss, mf_loss, emb_loss

    def rating(self, u_g_embeddings, pos_i_g_embeddings):
        return torch.matmul(u_g_embeddings, pos_i_g_embeddings.t())

    def make_hop(self):
        A_hat = self.sparse_dropout(self.sparse_norm_adj,
                                    self.node_dropout,
                                    self.sparse_norm_adj._nnz()) if True else self.sparse_norm_adj
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'],
                                    self.embedding_dict['item_emb']], 0)

        #norm_embed = F.normalize(ego_embeddings, p=2, dim=1)
        #all_embeddings = [norm_embed]

        all_embeddings = [ego_embeddings]

        for k in range(len(self.layers)):
            ego_embeddings = torch.sparse.mm(A_hat, ego_embeddings)
            #norm_embed = F.normalize(ego_embeddings, p=2, dim=1)
            #all_embeddings += [norm_embed]
            all_embeddings += [ego_embeddings]
        result = []
        for i in range(5):
            # print(i, len(all_embeddings))
            if i == 0:
                result.append(all_embeddings[0])
            else:
                temp_embeddings = all_embeddings[:i+1]
                embs = torch.stack(temp_embeddings, dim=1)
                temp_all_embeddings = torch.mean(embs, dim=1)
                # print(i, temp_all_embeddings.shape)
                result.append(temp_all_embeddings)
        # print(torch.stack(result).shape)
        return torch.stack(result)

    def forward(self, users, pos_items, neg_items, u_a, p_a, n_a, index, drop_flag=True):
        embed = self.make_hop()
        if index == 0:
            u_embeddings = embed[0, users]
            p_embeddings = embed[0, np.array(pos_items)+self.n_user]
            n_embeddings = embed[0, np.array(neg_items)+self.n_user]
            u_g_embeddings = embed[u_a, users]
            pos_i_g_embeddings = embed[p_a, np.array(pos_items)+self.n_user]
            neg_i_g_embeddings = embed[n_a, np.array(neg_items)+self.n_user]
            #u_g_embeddings = torch.cat((u_embeddings, u_g_embeddings), dim=1)
            #pos_i_g_embeddings = torch.cat((p_embeddings, pos_i_g_embeddings), dim=1)
            #neg_i_g_embeddings = torch.cat((n_embeddings, neg_i_g_embeddings), dim=1)

            return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
        else:
            # for i in range(len(pos_items)):
            #     temp_embeddings = embed[0, pos_items[i]]
            #     print(i, pos_items[i])
            # print(pos_items.shape, embed.shape, users.shape)
            u_embeddings = embed[0, np.array(users) + self.n_user]
            p_embeddings = embed[0, pos_items]
            n_embeddings = embed[0, neg_items]
            #print(u_a.shape, users.shape, min(users), max(users), self.n_user, embed.shape)
            u_g_embeddings = embed[u_a, np.array(users) + self.n_user]
            pos_i_g_embeddings = embed[p_a, pos_items]
            neg_i_g_embeddings = embed[n_a, neg_items]
            #u_g_embeddings = torch.cat((u_embeddings, u_g_embeddings), dim=1)
            #pos_i_g_embeddings = torch.cat((p_embeddings, pos_i_g_embeddings), dim=1)
            #neg_i_g_embeddings = torch.cat((n_embeddings, neg_i_g_embeddings), dim=1)
            return u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
