"""
GCN model for relation extraction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math

from model.tree import Tree, head_to_tree, tree_to_adj
from utils import constant, torch_utils

from .ON_LSTM import ONLSTMStack

class GCNClassifier(nn.Module):
    """ A wrapper classifier for GCNRelationModel. """
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.gcn_model = GCNRelationModel(opt, emb_matrix=emb_matrix)
        in_dim = opt['hidden_dim']
        self.classifier = nn.Linear(in_dim, opt['num_class'])
        self.opt = opt

    def conv_l2(self):
        return self.gcn_model.gcn.conv_l2()

    def clipEmbedding(self):
        self.gcn_model.clipEmbedding()

    def forward(self, inputs):
        outputs, pooling_output, att, dist, positive, negative = self.gcn_model(inputs)
        logits = self.classifier(outputs)
        return logits, pooling_output, att, dist, positive, negative

class GCNRelationModel(nn.Module):
    def __init__(self, opt, emb_matrix=None):
        super().__init__()
        self.opt = opt
        self.emb_matrix = emb_matrix

        # create embedding layers
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=constant.PAD_ID)
        self.pos_emb = nn.Embedding(len(constant.POS_TO_ID), opt['pos_dim']) if opt['pos_dim'] > 0 else None
        self.ner_emb = nn.Embedding(len(constant.NER_TO_ID), opt['ner_dim']) if opt['ner_dim'] > 0 else None
        self.chunk_emb = nn.Embedding(len(constant.CHUNK_TO_ID), opt['ner_dim']) if opt['ner_dim'] > 0 else None
        self.position_emb = nn.Embedding(2 * 400, opt['ner_dim']) if opt['ner_dim'] > 0 else None
        embeddings = (self.emb, self.pos_emb, self.ner_emb, self.chunk_emb, self.position_emb)
        self.init_embeddings()

        # gcn layer
        self.gcn = GCN(opt, embeddings, opt['hidden_dim'], opt['num_layers'])

        # output mlp layers
        in_dim = opt['hidden_dim']*3*2 + 2*2*opt['rnn_hidden']
        layers = [nn.Linear(in_dim, opt['hidden_dim']), nn.ReLU()]
        for _ in range(self.opt['mlp_layers']-1):
            layers += [nn.Linear(opt['hidden_dim'], opt['hidden_dim']), nn.ReLU()]
        self.out_mlp = nn.Sequential(*layers)

        self.disc = nn.Sequential(nn.Linear(opt['hidden_dim']*4*2*2, opt['hidden_dim']*4*2*2), nn.Tanh(),
                                  nn.Linear(opt['hidden_dim']*4*2*2, opt['hidden_dim']*4*2*2), nn.Tanh(),
                                  nn.Linear(opt['hidden_dim']*4*2*2,1), nn.Sigmoid())

    def init_embeddings(self):
        if self.emb_matrix is None:
            self.emb.weight.data[1:,:].uniform_(-1.0, 1.0)
        else:
            self.emb_matrix = torch.from_numpy(self.emb_matrix)
            self.emb.weight.data.copy_(self.emb_matrix)
        # decide finetuning
        if self.opt['topn'] <= 0:
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif self.opt['topn'] < self.opt['vocab_size']:
            print("Finetune top {} word embeddings.".format(self.opt['topn']))
            self.emb.weight.register_hook(lambda x: \
                    torch_utils.keep_partial_grad(x, self.opt['topn']))
        else:
            print("Finetune all embeddings.")

    def clipEmbedding(self):
        self.pos_emb.weight.data = clipTwoDimentions(self.pos_emb.weight.data)
        self.ner_emb.weight.data = clipTwoDimentions(self.ner_emb.weight.data)
        self.chunk_emb.weight.data = clipTwoDimentions(self.chunk_emb.weight.data)
        self.position_emb.weight.data = clipTwoDimentions(self.position_emb.weight.data)

    def forward(self, inputs):
        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type, chunks, on_path, dep_feat = inputs # unpack
        l = (masks.data.cpu().numpy() == 0).astype(np.int64).sum(1)
        maxlen = max(l)

        def inputs_to_tree_reps(head, words, l, prune, subj_pos, obj_pos):
            trees = [head_to_tree(head[i], words[i], l[i], prune, subj_pos[i], obj_pos[i]) for i in range(len(l))]
            adj = [tree_to_adj(maxlen, tree, directed=False, self_loop=False).reshape(1, maxlen, maxlen) for tree in trees]
            adj = np.concatenate(adj, axis=0)
            adj = torch.from_numpy(adj)
            return Variable(adj.cuda()) if self.opt['cuda'] else Variable(adj)

        # adj = inputs_to_tree_reps(head.data, words.data, l, self.opt['prune_k'], subj_pos.data, obj_pos.data)
        h, pool_mask, h2, h0, dist, att = self.gcn(None, inputs)
        # h, pool_mask = self.gcn(None, inputs)

        # pooling
        subj_mask, obj_mask = subj_pos.eq(0).eq(0).unsqueeze(2), obj_pos.eq(0).eq(0).unsqueeze(2) # invert mask
        pool_type = self.opt['pooling']
        h_out = pool(h, pool_mask, type=pool_type)
        h_out2 = pool(h2, pool_mask, type=pool_type)
        subj_out = pool(h, subj_mask, type=pool_type)
        subj_out2 = pool(h2, subj_mask, type=pool_type)
        subj_out_h0 = pool(h0, subj_mask, type=pool_type)
        obj_out = pool(h, obj_mask, type=pool_type)
        obj_out2 = pool(h2, obj_mask, type=pool_type)
        obj_out_h0 = pool(h0, obj_mask, type=pool_type)
        outputs = torch.cat([h_out, subj_out, obj_out, subj_out_h0, obj_out_h0], dim=1)
        outputs2 = torch.cat([h_out2, subj_out2, obj_out2], dim=1)
        positive = self.disc(torch.cat([outputs, outputs2], dim=1))
        negative = self.disc(torch.cat([outputs, outputs2[torch.randperm(outputs2.shape[0])]], dim=1))
        outputs = self.out_mlp(outputs)
        return outputs, h_out, att, dist, positive, negative

class GCN(nn.Module):
    """ A GCN/Contextualized GCN module operated on dependency graphs. """
    def __init__(self, opt, embeddings, mem_dim, num_layers):
        super(GCN, self).__init__()
        self.opt = opt
        self.layers = num_layers
        self.use_cuda = opt['cuda']
        self.mem_dim = mem_dim
        self.in_dim = opt['emb_dim'] + opt['pos_dim'] + opt['ner_dim'] + 3 * opt['ner_dim'] + 1 + 44

        self.emb, self.pos_emb, self.ner_emb, self.chunk_emb, self.position_emb = embeddings

        # rnn layer
        if self.opt.get('rnn', False):
            input_size = self.in_dim
            self.rnn = nn.LSTM(input_size, opt['rnn_hidden'], opt['rnn_layers'], batch_first=True, \
                    dropout=opt['rnn_dropout'], bidirectional=True)
            self.rnn2 = ONLSTMStack([self.in_dim, opt['rnn_hidden'] * 2, opt['rnn_hidden'] * 2], 5)
            self.in_dim = opt['rnn_hidden'] * 2
            self.rnn_drop = nn.Dropout(opt['rnn_dropout']) # use on last layer output

        self.in_drop = nn.Dropout(opt['input_dropout'])
        self.gcn_drop = nn.Dropout(opt['gcn_dropout'])

        # gcn layer
        self.W = nn.ModuleList()

        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))

        self.K = nn.Linear(opt['rnn_hidden'] * 2, 2 * opt['rnn_hidden'])
        self.Q = nn.Linear(opt['rnn_hidden'] * 2, 2 * opt['rnn_hidden'])
        self.V = nn.Linear(opt['rnn_hidden'] * 2, 2 * opt['rnn_hidden'])

        # self.ff1 = nn.Linear(4 * opt['rnn_hidden'], 2 * opt['rnn_hidden'])
        # self.ff2 = nn.Linear(2 * opt['rnn_hidden'], 2 * opt['rnn_hidden'])
        # self.ffl = nn.Linear(2 * opt['rnn_hidden'], 1)

        self.W_q = nn.Linear(2 * self.in_dim, self.in_dim)
        self.W_c = nn.Linear(2 * self.in_dim, self.in_dim)
        self.W_k = nn.Linear(self.in_dim, 1)
        self.W_m = nn.Linear(3 * self.in_dim, self.in_dim)

    def conv_l2(self):
        conv_weights = []
        for w in self.W:
            conv_weights += [w.weight, w.bias]
        return sum([x.pow(2).sum() for x in conv_weights])

    def encode_with_rnn(self, rnn_inputs, masks, batch_size):
        seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        h0, c0 = rnn_zero_state(batch_size, self.opt['rnn_hidden'], self.opt['rnn_layers'])
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, adj, inputs):
        words, masks, pos, ner, deprel, head, subj_pos, obj_pos, subj_type, obj_type, chunks, on_path, dep_feat = inputs # unpack
        word_embs = self.emb(words)
        embs = [word_embs]
        if self.opt['pos_dim'] > 0:
            embs += [self.pos_emb(pos)]
        if self.opt['ner_dim'] > 0:
            embs += [self.ner_emb(ner)]
        if self.opt['ner_dim'] > 0:
            embs += [self.chunk_emb(chunks)]
            embs += [self.position_emb(subj_pos)]
            embs += [self.position_emb(obj_pos)]
        embs = torch.cat(embs+[on_path.unsqueeze(2).float(), dep_feat.view(words.shape[0], words.shape[1], -1).float()], dim=2)
        # embs = self.in_drop(embs)

        # rnn layer
        if self.opt.get('rnn', False):
            gcn_inputs = self.rnn_drop(self.encode_with_rnn(embs, masks, words.size()[0]))
            rnn_outputs, _, _, _, dist = self.rnn2(embs.transpose(0,1), self.rnn2.init_hidden(embs.shape[0]))
            dist = dist[0][-1]
            dist = dist.transpose(0,1)
            # rnn_outputs = rnn_outputs[0]
            rnn_outputs = rnn_outputs.transpose(0,1)
            #
            # print((-torch.log(dist) * dist).sum(1).mean())
            #
            # dist = dist.masked_fill(masks, 2)
            # created_adj = create_adj(dist.data.cpu().numpy())
            # adj = created_adj.float()
            onlstm_output = rnn_outputs.float()
        else:
            gcn_inputs = embs
        
        # # gcn layer
        # denom = adj.sum(2).unsqueeze(2) + 1
        # mask = (adj.sum(2) + adj.sum(1)).eq(0).unsqueeze(2)
        # # zero out adj for ablation
        # if self.opt.get('no_adj', False):
        #     adj = torch.zeros_like(adj)
        #
        # for l in range(self.layers):
        #     Ax = adj.bmm(gcn_inputs)
        #     AxW = self.W[l](Ax)
        #     AxW = AxW + self.W[l](gcn_inputs) # self loop
        #     AxW = AxW / denom
        #
        #     gAxW = F.relu(AxW)
        #     gcn_inputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW
        #

        subj_mask, obj_mask = subj_pos.eq(0).eq(0).unsqueeze(2), obj_pos.eq(0).eq(0).unsqueeze(2)  # invert mask
        subj_out = pool(gcn_inputs, subj_mask, type='max')
        obj_out = pool(gcn_inputs, subj_mask, type='max')
        subj_obj = torch.cat([subj_out, obj_out], dim=1)

        c = Variable(torch.zeros(words.shape[0], self.in_dim), requires_grad=True).cuda()

        q = nn.ReLU()(self.W_q(subj_obj))
        t = nn.ReLU()(self.W_c(torch.cat([q, c], dim=1)))
        t = t.repeat(1, words.shape[1]).view(words.shape[0], words.shape[1], -1)
        sf1 = nn.Softmax(1)
        k = sf1(self.W_k(t * gcn_inputs))
        c = (k * gcn_inputs).sum(1)
        m = nn.ReLU()(self.W_m(torch.cat([c, subj_obj], dim=1)))
        m = m.repeat(1, words.shape[1]).view(words.shape[0], words.shape[1], -1)

        key = self.K(gcn_inputs)
        query = self.Q(gcn_inputs)
        value = self.V(gcn_inputs)

        sf2 = nn.Softmax(2)
        att = sf2(key.bmm(query.transpose(1,2))) / math.sqrt(self.opt['rnn_hidden'] * 2)
        att = att.masked_fill(torch.eye(att.shape[1],att.shape[2]).byte().cuda(), 0).sum(2)
        output = sf2(key.bmm(query.transpose(1,2)) / math.sqrt(self.opt['rnn_hidden'] * 2)).bmm(value)
        output = m * output

        # h1 = output.repeat(1, output.shape[1], 1)
        # h2 = output.repeat(1, 1, output.shape[1]).view(output.shape[0], output.shape[1]*output.shape[1], output.shape[2])
        # h = torch.cat([h1,h2], dim=2)
        # h = torch.sigmoid(self.ffl(self.ff2(self.ff1(h)))).squeeze()


        return output, masks.unsqueeze(2), onlstm_output, gcn_inputs, dist, att

def pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)

def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True, use_cuda=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    if use_cuda:
        return h0.cuda(), c0.cuda()
    else:
        return h0, c0

def clipTwoDimentions(mat, norm=3.0, device='cuda'):
   col_norms = ((mat ** 2).sum(0, keepdim=True)) ** 0.5
   desired_norms = col_norms.clamp(0.0, norm)
   scale = desired_norms / (1e-7 + col_norms)
   res = mat * scale
   res = res.to(device)
   return res

