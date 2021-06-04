import json
import math
import os
import pickle
import random
from collections import defaultdict
from multiprocessing import Pool
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import *
from models.bert import Bert
from utils import get_cuda
import time


IGNORE_INDEX = -100

def create_graph(ems_info):

    d = defaultdict(list)
    sent_table = np.where(ems_info[:,0]==3)[0]
    sent_num = sent_table.shape[0]
    mention_table = np.where(ems_info[:,0]==2)[0]
    mention_num = mention_table.shape[0]
    entity_table = np.where(ems_info[:,0]==1)[0]
    entity_num = entity_table.shape[0]

    N_nodes = sent_num+mention_num
    nodes_adj = np.zeros((N_nodes,N_nodes), dtype=np.int32)
    edges_cnt = 1
    # 1 Sentence-Sentence edges
    for i in range(sent_num):
        for j in range(i+1,sent_num):
            nodes_adj[i,j] = edges_cnt
            nodes_adj[j,i] = edges_cnt
    
    # 2 sentence mention
    sent2men = defaultdict(list)
    entity2men = defaultdict(list)
    edges_cnt += 1
    for i in range(mention_num):
        sent_id = ems_info[mention_table[i],-1]
        mention_id = mention_table[i] - entity_num + sent_num
        entity_id = ems_info[mention_table[i],-3]
        nodes_adj[sent_id,mention_id] = edges_cnt
        nodes_adj[mention_id,sent_id] = edges_cnt

        sent2men[sent_id].append(mention_id)
        entity2men[entity_id].append(mention_id)

    # 3 co reference
    edges_cnt += 1
    for m_set in entity2men.values():
        for i in range(len(m_set)):
            for j in range(i+1,len(m_set)):
                nodes_adj[m_set[i], m_set[j]] = edges_cnt
                nodes_adj[m_set[j], m_set[i]] = edges_cnt

    # 4 Mention mention 
    edges_cnt += 1
    for m_set in sent2men.values():
        for i in range(len(m_set)):
            for j in range(i+1,len(m_set)):
                nodes_adj[m_set[i], m_set[j]] = edges_cnt
                nodes_adj[m_set[j], m_set[i]] = edges_cnt


    nodes_info = np.zeros((N_nodes,6), dtype=np.int32)
    for iii in range(sent_num):
        sent_idx = sent_table[iii]
        nodes_info[iii] = np.array([ems_info[sent_idx,2],ems_info[sent_idx,3],-1,1,iii,iii])
    for iii in range(mention_num):
        mention_idx = mention_table[iii]
        nodes_info[iii+sent_num] = np.array([ems_info[mention_idx,2],ems_info[mention_idx,3],ems_info[mention_idx,-3],2,iii+sent_num,ems_info[mention_idx,-1]])

    meta_path = meta_path_finder(nodes_adj, nodes_info, sent_num, N_nodes)
    tree_path = tree_path_finder(nodes_adj, nodes_info, sent_num, N_nodes)

    assert np.sum(nodes_info[:,3]>0) == sent_num+mention_num

    return nodes_adj,nodes_info,meta_path,tree_path

def meta_path_finder(nodes_adj,nodes_info,id_start,id_end):

    path = dict()
    for i in range(id_start,id_end):
        for j in range(i, id_end):
            if nodes_info[i,2] == nodes_info[j,2]:
                continue
            record = []
            a = set(np.where(nodes_adj[i]==4)[0])
            b = set(np.where(nodes_adj[j]==4)[0])
            p3 = []
            p3_rev = []
            for tmp1 in list(a-b):
                for tmp2 in list(b-a):
                    if nodes_adj[tmp1,tmp2]==3:
                        p3.append([tmp1,tmp2])
                        p3_rev.append([tmp2,tmp1])
                        record.append([nodes_info[tmp1,5],nodes_info[tmp2,5]])

            a = set(np.where(nodes_adj[i]==2)[0])
            b = set(np.where(nodes_adj[j]==2)[0])
            p1 = [[val] for val in list(a & b)]
            p2 = []
            p2_rev = []
            for tmp1 in list(a-b):
                for tmp2 in list(b-a):
                    if [tmp1,tmp2] not in record:
                        p2.append([tmp1,tmp2])
                        p2_rev.append([tmp2,tmp1])
            
            path[(i, j)] = [p1]+[p2]+[p3]
            path[(j, i)] = [p1]+[p2_rev]+[p3_rev]

    return path

def tree_path_finder(nodes_adj,nodes_info,id_start,id_end):

    path = dict()
    for i in range(id_start,id_end):
        for j in range(id_start, id_end):
            if nodes_info[i,2] == nodes_info[j,2] or i == j:
                continue
            search_space = [[i]]
            reach_node = [i]
            get_path = []
            while search_space != []:
                new_search_space = []
                for sp in search_space:
                    if nodes_adj[sp[-1],j] > 0:
                        get_path.append(sp)
                        new_search_space = []
                        break
                    else:
                        adj = list(np.where(nodes_adj[sp[-1]]>0)[0])
                        for tmp in adj:
                            if tmp in reach_node or (nodes_info[tmp,2]==nodes_info[i,2]) or (nodes_info[tmp,2]==nodes_info[j,2]):
                                continue
                            new_search_space.append(sp[:]+[tmp])
                            reach_node.append(tmp)
                search_space = new_search_space
            path[(i, j)] = get_path 

    return path

class DGLREDataset(IterableDataset):

    def __init__(self, src_file, save_file, word2id, ner2id, rel2id,
                 dataset_type='train', instance_in_train=None, opt=None,length_limit=None):

        super(DGLREDataset, self).__init__()

        start_time = time.time()

        # record training set mention triples
        self.instance_in_train = set([]) if instance_in_train is None else instance_in_train
        self.data = None
        self.document_max_length = 512
        self.INTRA_EDGE = 0
        self.INTER_EDGE = 1
        self.LOOP_EDGE = 2
        self.count = 0

        print('Reading data from {}.'.format(src_file))
        if os.path.exists(save_file):
            with open(file=save_file, mode='rb') as fr:
                info = pickle.load(fr)
                self.data = info['data']
                self.instance_in_train = info['intrain_set']
            print('load preprocessed data from {}.'.format(save_file))

        else:
            with open(file=src_file, mode='r', encoding='utf-8') as fr:
                ori_data = json.load(fr)
            if length_limit is not None:
                ori_data = ori_data[:length_limit]
            print('loading..')
            self.data = []
            unk_word_cnt = 0
            total_word_cnt = 0

            for i, doc in enumerate(ori_data):

                title, entity_list, labels, sentences = \
                    doc['title'], doc['vertexSet'], doc.get('labels', []), doc['sents']

                Ls = [0]
                L = 0
                for x in sentences:
                    L += len(x)
                    Ls.append(L)
                for j in range(len(entity_list)):
                    for k in range(len(entity_list[j])):
                        sent_id = int(entity_list[j][k]['sent_id'])
                        entity_list[j][k]['sent_id'] = sent_id

                        dl = Ls[sent_id]
                        pos0, pos1 = entity_list[j][k]['pos']
                        entity_list[j][k]['global_pos'] = (pos0 + dl, pos1 + dl)

                # generate positive examples
                train_triple = []
                new_labels = []
                for label in labels:
                    head, tail, relation, evidence = label['h'], label['t'], label['r'], label['evidence']
                    assert (relation in rel2id), 'no such relation {} in rel2id'.format(relation)
                    label['r'] = rel2id[relation]

                    train_triple.append((head, tail))

                    label['in_train'] = False

                    # record training set mention triples and mark it for dev and test set
                    for n1 in entity_list[head]:
                        for n2 in entity_list[tail]:
                            mention_triple = (n1['name'], n2['name'], relation)
                            if dataset_type == 'train':
                                self.instance_in_train.add(mention_triple)
                            else:
                                if mention_triple in self.instance_in_train:
                                    label['in_train'] = True
                                    break

                    new_labels.append(label)

                # generate negative examples
                na_triple = []
                for j in range(len(entity_list)):
                    for k in range(len(entity_list)):
                        if j != k and (j, k) not in train_triple:
                            na_triple.append((j, k))

                # generate document ids
                words = []
                for sentence in sentences:
                    for word in sentence:
                        words.append(word)
                if len(words) > self.document_max_length:
                    words = words[:self.document_max_length]

                word_id = np.zeros((self.document_max_length,), dtype=np.int32)
                ner_id = np.zeros((self.document_max_length,), dtype=np.int32)
                pos_id = np.zeros((self.document_max_length,), dtype=np.int32)
                sentence_id = np.zeros((self.document_max_length,), dtype=np.int32)
                ems_info = np.zeros((130,7), dtype=np.int32)

                unkid = word2id["UNK"]
                for iii, w in enumerate(words):
                    word = word2id.get(w.lower(), word2id['UNK'])
                    word_id[iii] = word
                    if unkid == word:
                        unk_word_cnt += 1
                    total_word_cnt += 1

                mention_idx = len(entity_list)
                already_exist = set()  # dealing with NER overlapping problem
                for idx, vertex in enumerate(entity_list):
                    ems_info[idx] = np.array([1,ner2id[vertex[0]["type"]],-1,-1,idx,idx,-1])
                    for v in vertex:
                        sent_id, (pos0, pos1), ner_type = v['sent_id'], v['global_pos'], v['type']
                        if (pos0, pos1) in already_exist:
                            continue
                        if pos0 >= self.document_max_length:
                            continue
                        ner_id[pos0:pos1] = ner2id[ner_type]
                        pos_id[pos0:pos1] = idx+1
                        ems_info[mention_idx] = np.array([2,ner2id[ner_type],pos0,pos1,idx,mention_idx,sent_id])
                        mention_idx += 1
                        already_exist.add((pos0, pos1))
                for iii in range(1,len(Ls)):
                    sentence_id[Ls[iii-1]:Ls[iii]] = iii
                    ems_info[mention_idx] = np.array([3,-1,Ls[iii-1],Ls[iii],-1,mention_idx,iii-1])
                    mention_idx += 1

                # # construct graph
                graph_adj,graph_info,meta_path,tree_path = create_graph(ems_info)

                overlap = doc.get('overlap_entity_pair', [])
                new_overlap = [tuple(item) for item in overlap]

                self.data.append({
                    'title': title,
                    'entities': entity_list,
                    'labels': new_labels,
                    'na_triple': na_triple,
                    'word_id': word_id,
                    'pos_id': pos_id,
                    "sentence_id": sentence_id,
                    'ner_id': ner_id,
                    "ems_info": ems_info,
                    "index": i,
                    'graph_adj': graph_adj,
                    "graph_info":graph_info,
                    "meta_path": meta_path,
                    "tree_path": tree_path,
                    'overlap': new_overlap,
                })

            # save data
            with open(file=save_file, mode='wb') as fw:
                pickle.dump({'data': self.data, 'intrain_set': self.instance_in_train}, fw)
            print('finish reading {} consuming time {:.3f}s and save preprocessed data to {}.'.format(src_file, time.time()-start_time,save_file))

            print("UNK WORD Percentage:{:.3f}%".format(100*unk_word_cnt/total_word_cnt))
        if opt.k_fold != "none":
            k_fold = opt.k_fold.split(',')
            k, total = float(k_fold[0]), float(k_fold[1])
            a = (k - 1) / total * len(self.data)
            b = k / total * len(self.data)
            self.data = self.data[:a] + self.data[b:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)


class BERTDGLREDataset(IterableDataset):

    def __init__(self, src_file, save_file, word2id, ner2id, rel2id,
                 dataset_type='train', instance_in_train=None, opt=None, length_limit=None):

        super(BERTDGLREDataset, self).__init__()

        start_time = time.time()

        # record training set mention triples
        self.instance_in_train = set([]) if instance_in_train is None else instance_in_train
        self.data = None
        self.document_max_length = 512
        self.INFRA_EDGE = 0
        self.INTER_EDGE = 1
        self.LOOP_EDGE = 2
        self.count = 0

        print('Reading data from {}.'.format(src_file))
        if os.path.exists(save_file):
            with open(file=save_file, mode='rb') as fr:
                info = pickle.load(fr)
                self.data = info['data']
                self.instance_in_train = info['intrain_set']
            print('load preprocessed data from {}.'.format(save_file))

        else:
            bert = Bert(BertModel, opt.bert_path)

            with open(file=src_file, mode='r', encoding='utf-8') as fr:
                ori_data = json.load(fr)
            if length_limit is not None:
                ori_data = ori_data[:length_limit]
            print('loading..')
            self.data = []
            overlen_cnt = 0
            overlap_cnt = 0

            for i, doc in enumerate(ori_data):

                title, entity_list, labels, sentences = \
                    doc['title'], doc['vertexSet'], doc.get('labels', []), doc['sents']

                Ls = [0]
                L = 0
                for x in sentences:
                    L += len(x)
                    Ls.append(L)
                for j in range(len(entity_list)):
                    for k in range(len(entity_list[j])):
                        sent_id = int(entity_list[j][k]['sent_id'])
                        entity_list[j][k]['sent_id'] = sent_id

                        dl = Ls[sent_id]
                        pos0, pos1 = entity_list[j][k]['pos']
                        entity_list[j][k]['global_pos'] = (pos0 + dl, pos1 + dl)

                # generate positive examples
                train_triple = []
                new_labels = []
                for label in labels:
                    head, tail, relation, evidence = label['h'], label['t'], label['r'], label['evidence']
                    assert (relation in rel2id), 'no such relation {} in rel2id'.format(relation)
                    label['r'] = rel2id[relation]

                    train_triple.append((head, tail))

                    label['in_train'] = False

                    # record training set mention triples and mark it for dev and test set
                    for n1 in entity_list[head]:
                        for n2 in entity_list[tail]:
                            mention_triple = (n1['name'], n2['name'], relation)
                            if dataset_type == 'train':
                                self.instance_in_train.add(mention_triple)
                            else:
                                if mention_triple in self.instance_in_train:
                                    label['in_train'] = True
                                    break

                    new_labels.append(label)

                # generate negative examples
                na_triple = []
                for j in range(len(entity_list)):
                    for k in range(len(entity_list)):
                        if j != k and (j, k) not in train_triple:
                            na_triple.append((j, k))

                # generate document ids
                words = []
                for sentence in sentences:
                    for word in sentence:
                        words.append(word)

                bert_token, bert_starts, bert_subwords = bert.subword_tokenize_to_ids(words)
                if len(bert_subwords)>=511:
                    overlen_cnt+=1

                word_id = np.zeros((self.document_max_length,), dtype=np.int32)
                pos_id = np.zeros((self.document_max_length,), dtype=np.int32)
                ner_id = np.zeros((self.document_max_length,), dtype=np.int32)
                sentence_id = np.zeros((self.document_max_length,), dtype=np.int32)
                ems_info = np.zeros((130,7), dtype=np.int32)
                word_id[:] = bert_token[0]

                new_Ls = [0]
                for ii in range(1, len(Ls)):
                    new_Ls.append(bert_starts[Ls[ii]] if Ls[ii] < len(bert_starts) else len(bert_subwords))
                Ls = new_Ls

                mention_idx = len(entity_list)
                entity2mention = defaultdict(list)
                already_exist = set()  # dealing with NER overlapping problem
                for idx, vertex in enumerate(entity_list):
                    # nodetype nertype start end entityid mentioncnt sentid
                    ems_info[idx] = np.array([1,ner2id[vertex[0]["type"]],-1,-1,idx,idx,-1])
                    for v in vertex:

                        sent_id, (pos0, pos1), ner_type = v['sent_id'], v['global_pos'], ner2id[v['type']]

                        pos0 = bert_starts[pos0]
                        pos1 = bert_starts[pos1] if pos1 < len(bert_starts) else len(bert_subwords)

                        # assert pos1!=512

                        if pos0 >= self.document_max_length:
                            continue
                        if (pos0, pos1) in already_exist:
                            overlap_cnt += 1
                            continue

                        pos_id[pos0:pos1] = idx+1
                        ner_id[pos0:pos1] = ner_type
                        ems_info[mention_idx] = np.array([2,ner_type,pos0,pos1,idx,mention_idx,sent_id])
                        mention_idx += 1
                        entity2mention[idx+1].append(mention_idx)
                        already_exist.add((pos0, pos1))
                for iii in range(1,len(Ls)):
                    sentence_id[Ls[iii-1]:Ls[iii]] = iii
                    ems_info[mention_idx] = np.array([3,-1,Ls[iii-1],Ls[iii],-1,mention_idx,iii-1])
                    mention_idx += 1
                # replace_i = 0
                # idx = len(entity_list)
                # if entity2mention[idx] == []:
                #     entity2mention[idx].append(mention_idx)
                    # while mention_id[replace_i] != 0:
                    #     replace_i += 1
                    # mention_id[replace_i] = mention_idx
                    # entity_id[replace_i] = idx
                    # mention2sent[mention_idx] = sentence_id[replace_i]
                    # ner_id[replace_i] = ner2id[vertex[0]['type']]
                    # mention_idx += 1

                # # construct graph
                graph_adj,graph_info,meta_path,tree_path = create_graph(ems_info)

                overlap = doc.get('overlap_entity_pair', [])
                new_overlap = [tuple(item) for item in overlap]

                self.data.append({
                    'title': title,
                    'entities': entity_list,
                    'labels': new_labels,
                    'na_triple': na_triple,
                    'word_id': word_id,
                    "ner_id": ner_id,
                    "pos_id": pos_id,
                    "sentence_id": sentence_id,
                    'ems_info': ems_info,
                    "index": i,
                    'graph_adj': graph_adj,
                    "graph_info":graph_info,
                    'meta_path': meta_path,
                    "tree_path": tree_path,
                    'overlap': new_overlap,
                })

            # save data
            with open(file=save_file, mode='wb') as fw:
                pickle.dump({'data': self.data, 'intrain_set': self.instance_in_train}, fw)
            print('finish reading {} and consuming time{:.3f}s save preprocessed data to {}.'.format(src_file, time.time()-start_time,save_file))
            print("overmax len:{}".format(overlen_cnt))
            print("overlay cnt:{}".format(overlap_cnt))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)

class DGLREDataloader(DataLoader):

    def __init__(self, dataset, args, batch_size, shuffle=False, h_t_limit=1722,max_length=512, dataset_type='train'):
        super(DGLREDataloader, self).__init__(dataset, batch_size=batch_size)
        self.shuffle = shuffle
        self.length = len(self.dataset)
        self.max_length = max_length
        self.negativa_alpha = args.negativa_alpha
        self.dataset_type = dataset_type
        self.path_type = args.path_type

        self.h_t_limit = h_t_limit
        self.node_limit = 130
        self.entity_limit = 45
        self.mention_limit = 85
        self.relation_num = args.relation_nums
        self.dis2idx = np.zeros((max_length), dtype='int64')
        self.dis2idx[1] = 1
        self.dis2idx[2:] = 2
        self.dis2idx[4:] = 3
        self.dis2idx[8:] = 4
        self.dis2idx[16:] = 5
        self.dis2idx[32:] = 6
        self.dis2idx[64:] = 7
        self.dis2idx[128:] = 8
        self.dis2idx[256:] = 9
        self.dis_size = 20

        self.path_per_type = args.path_per_type

        self.order = list(range(self.length))

    def __iter__(self):
        # shuffle
        if self.shuffle:
            random.shuffle(self.order)
        self.data = self.dataset
        batch_num = math.ceil(self.length / self.batch_size)
        self.batches_order = [self.order[idx * self.batch_size: min(self.length, (idx + 1) * self.batch_size)]
                        for idx in range(0, batch_num)]

        # begin
        n_path = self.path_per_type
        context_word_ids = torch.LongTensor(self.batch_size, self.max_length).cpu()
        context_pos_ids = torch.LongTensor(self.batch_size, self.max_length).cpu()
        context_ner_ids = torch.LongTensor(self.batch_size, self.max_length).cpu()
        context_mention_ids = torch.LongTensor(self.batch_size, self.max_length).cpu()
        context_sent_ids = torch.LongTensor(self.batch_size, self.max_length).cpu()
        context_word_mask = torch.LongTensor(self.batch_size, self.max_length).cpu()
        context_word_length = torch.LongTensor(self.batch_size).cpu()
        ht_pairs = torch.LongTensor(self.batch_size, self.h_t_limit, 2).cpu()
        relation_multi_label = torch.Tensor(self.batch_size, self.h_t_limit, self.relation_num).cpu()
        ign_relation_multi_label = torch.Tensor(self.batch_size, self.h_t_limit, self.relation_num).cpu()
        relation_mask = torch.Tensor(self.batch_size, self.h_t_limit).cpu()
        relation_label = torch.LongTensor(self.batch_size, self.h_t_limit).cpu()
        relation_path = torch.LongTensor(self.batch_size, self.h_t_limit,3*n_path,4).cpu()
        ht_pair_distance = torch.LongTensor(self.batch_size, self.h_t_limit).cpu()
        ht_sent_distance = torch.LongTensor(self.batch_size, self.h_t_limit).cpu()
        batch_graph_adj = torch.LongTensor(self.batch_size, self.node_limit, self.node_limit).cpu()
        graph_node_num = torch.LongTensor(self.batch_size).cpu()
        batch_graph_info = torch.LongTensor(self.batch_size, self.node_limit,6).cpu()
        context_ems_info = torch.LongTensor(self.batch_size, self.node_limit,7).cpu()

        for idx, minibatch in enumerate(self.batches_order):
            start_time = time.time()
            cur_bsz = len(minibatch)

            for mapping in [context_word_ids,context_word_mask, context_word_length,ht_pairs, ht_pair_distance, ht_sent_distance,relation_multi_label, ign_relation_multi_label,relation_mask, relation_label,relation_path,batch_graph_adj,graph_node_num,batch_graph_info,context_pos_ids,context_ner_ids,context_sent_ids,context_mention_ids,context_ems_info]:
                if mapping is not None:
                    mapping.zero_()

            relation_label.fill_(IGNORE_INDEX)

            max_h_t_cnt = 0

            label_list = []
            L_vertex = []
            titles = []
            indexes = []
            overlaps = []
            

            for i, example_idx in enumerate(minibatch):
                example = self.data[example_idx]
                title, entities, labels, na_triple, word_id, ems_info, graph_adj, meta_path, tree_path, graph_info,ner_id,pos_id, sentence_id = \
                    example['title'], example['entities'], example['labels'], example['na_triple'],\
                    example['word_id'], example['ems_info'],example['graph_adj'], example['meta_path'], example["tree_path"],\
                    example["graph_info"], example["ner_id"], example["pos_id"], example["sentence_id"]
                nodes_num = graph_info.shape[0]
                batch_graph_adj[i,:nodes_num,:nodes_num].copy_(torch.from_numpy(graph_adj))
                batch_graph_info[i,:nodes_num].copy_(torch.from_numpy(graph_info))
                graph_node_num[i] = nodes_num
                overlaps.append(example['overlap'])

                entity2mention = defaultdict(list)
                entity_table = np.where(graph_info[:,3]==2)[0]
                for iii in range(entity_table.shape[0]):
                    idx = entity_table[iii]
                    entity2mention[graph_info[idx,2]].append(idx)
                    context_mention_ids[i,graph_info[idx,0]:graph_info[idx,1]] = graph_info[idx,4]

                L = len(entities)
                word_num = word_id.shape[0]

                context_word_ids[i, :word_num].copy_(torch.from_numpy(word_id))
                context_ner_ids[i, :word_num].copy_(torch.from_numpy(ner_id))
                context_pos_ids[i, :word_num].copy_(torch.from_numpy(pos_id))
                context_sent_ids[i, :word_num].copy_(torch.from_numpy(sentence_id))
                context_ems_info[i].copy_(torch.from_numpy(ems_info))

                idx2label = defaultdict(list)
                label_set = {}
                for label in labels:
                    head, tail, relation, intrain, evidence = \
                        label['h'], label['t'], label['r'], label['in_train'], label['evidence']
                    idx2label[(head, tail)].append(relation)
                    label_set[(head, tail, relation)] = intrain

                label_list.append(label_set)

                if self.dataset_type == 'train' and self.negativa_alpha > 0.0:
                    train_tripe = list(idx2label.keys())
                    for j, (h_idx, t_idx) in enumerate(train_tripe):
                        hlist, tlist = entities[h_idx], entities[t_idx]
                        ht_pairs[i, j, :] = torch.Tensor([h_idx, t_idx])
                        label = idx2label[(h_idx, t_idx)]

                        delta_dis = hlist[0]['global_pos'][0] - tlist[0]['global_pos'][0]
                        if delta_dis < 0:
                            ht_pair_distance[i, j] = -int(self.dis2idx[-delta_dis]) + self.dis_size // 2
                        else:
                            ht_pair_distance[i, j] = int(self.dis2idx[delta_dis]) + self.dis_size // 2
                        sent_dis = hlist[0]['sent_id'] - tlist[0]['sent_id']
                        if sent_dis < 0:
                            ht_sent_distance[i, j] = -int(self.dis2idx[-sent_dis]) + self.dis_size // 2
                        else:
                            ht_sent_distance[i, j] = int(self.dis2idx[sent_dis]) + self.dis_size // 2
                        for r in label:
                            relation_multi_label[i, j, r] = 1
                            ign_relation_multi_label[i,j,r] = label_set[(h_idx,t_idx,r)]

                        relation_mask[i, j] = 1
                        rt = np.random.randint(len(label))
                        relation_label[i, j] = label[rt]

                        relation_path[i, j] = mergy_all_path(h_idx,t_idx,meta_path,tree_path,entity2mention,self.path_type,n_path)
                        
                    random.shuffle(na_triple)
                    lower_bound = int(max(20, len(train_tripe) * self.negativa_alpha))
                    lower_bound = min(lower_bound,self.h_t_limit-len(train_tripe))

                    for j, (h_idx, t_idx) in enumerate(na_triple[:lower_bound], len(train_tripe)):
                        hlist, tlist = entities[h_idx], entities[t_idx]
                        ht_pairs[i, j, :] = torch.Tensor([h_idx, t_idx])

                        delta_dis = hlist[0]['global_pos'][0] - tlist[0]['global_pos'][0]
                        if delta_dis < 0:
                            ht_pair_distance[i, j] = -int(self.dis2idx[-delta_dis]) + self.dis_size // 2
                        else:
                            ht_pair_distance[i, j] = int(self.dis2idx[delta_dis]) + self.dis_size // 2
                        sent_dis = hlist[0]['sent_id'] - tlist[0]['sent_id']
                        if sent_dis < 0:
                            ht_sent_distance[i, j] = -int(self.dis2idx[-sent_dis]) + self.dis_size // 2
                        else:
                            ht_sent_distance[i, j] = int(self.dis2idx[sent_dis]) + self.dis_size // 2
                        relation_multi_label[i, j, 0] = 1
                        relation_label[i, j] = 0
                        relation_mask[i, j] = 1
                        relation_path[i, j] = mergy_all_path(h_idx,t_idx,meta_path,tree_path,entity2mention,self.path_type,n_path)

                    max_h_t_cnt = max(max_h_t_cnt, len(train_tripe) + lower_bound)
                else:
                    if "test_pair" not in example:
                        train_tripe = list(idx2label.keys())
                        j = 0
                        for h_idx in range(L):
                            for t_idx in range(L):
                                if h_idx != t_idx:
                                    hlist, tlist = entities[h_idx], entities[t_idx]
                                    ht_pairs[i, j, :] = torch.Tensor([h_idx, t_idx])

                                    relation_mask[i, j] = 1

                                    delta_dis = hlist[0]['global_pos'][0] - tlist[0]['global_pos'][0]
                                    if delta_dis < 0:
                                        ht_pair_distance[i, j] = -int(self.dis2idx[-delta_dis]) + self.dis_size // 2
                                    else:
                                        ht_pair_distance[i, j] = int(self.dis2idx[delta_dis]) + self.dis_size // 2
                                    sent_dis = hlist[0]['sent_id'] - tlist[0]['sent_id']
                                    if sent_dis < 0:
                                        ht_sent_distance[i, j] = -int(self.dis2idx[-sent_dis]) + self.dis_size // 2
                                    else:
                                        ht_sent_distance[i, j] = int(self.dis2idx[sent_dis]) + self.dis_size // 2
                                    if (h_idx,t_idx) in train_tripe:
                                        label = idx2label[(h_idx, t_idx)]
                                        for r in label:
                                            relation_multi_label[i, j, r] = 1
                                            ign_relation_multi_label[i,j,r] = label_set[(h_idx,t_idx,r)]


                                        rt = np.random.randint(len(label))
                                        relation_label[i, j] = label[rt]
                                        relation_path[i, j] = mergy_all_path(h_idx,t_idx,meta_path,tree_path,entity2mention,self.path_type,n_path)
                                    else:
                                        relation_multi_label[i, j, 0] = 1
                                        relation_label[i, j] = 0
                                        relation_path[i, j] = mergy_all_path(h_idx,t_idx,meta_path,tree_path,entity2mention,self.path_type,n_path)

                                    j += 1
                        assert example["index"] == self.data[example["index"]]["index"]
                        self.data[example["index"]]["test_pair"] =  ht_pairs[i].clone().numpy()
                        self.data[example["index"]]["test_mask"] =  relation_mask[i].clone().numpy()
                        self.data[example["index"]]["test_distance"] =  ht_pair_distance[i].clone().numpy()
                        self.data[example["index"]]["test_sent_distance"] =  ht_sent_distance[i].clone().numpy()
                        self.data[example["index"]]["test_label"] =  relation_label[i].clone().numpy()
                        self.data[example["index"]]["test_mul_label"] =  relation_multi_label[i].clone().numpy()
                        self.data[example["index"]]["test_ign_label"] =  ign_relation_multi_label[i].clone().numpy()
                        self.data[example["index"]]["test_path"] =  relation_path[i].clone().numpy()
                        self.data[example["index"]]["test_num"] =  j
                    else:
                        ht_pairs[i] = torch.from_numpy(self.data[example["index"]]["test_pair"])
                        relation_mask[i] = torch.from_numpy(self.data[example["index"]]["test_mask"])
                        ht_pair_distance[i] = torch.from_numpy(self.data[example["index"]]["test_distance"])
                        ht_sent_distance[i] = torch.from_numpy(self.data[example["index"]]["test_sent_distance"])
                        relation_label[i] = torch.from_numpy(self.data[example["index"]]["test_label"])
                        relation_multi_label[i] = torch.from_numpy(self.data[example["index"]]["test_mul_label"])
                        ign_relation_multi_label[i] = torch.from_numpy(self.data[example["index"]]["test_ign_label"])
                        relation_path[i] = torch.from_numpy(self.data[example["index"]]["test_path"])
                        j = self.data[example["index"]]["test_num"]

                    max_h_t_cnt = max(max_h_t_cnt, j)
                L_vertex.append(L)
                titles.append(title)
                indexes.append(example["index"])

            context_word_mask = context_word_ids > 0
            context_word_length = context_word_mask.sum(1)
            batch_max_length = context_word_length.max()

            # print("read time{:.3f}s".format(time.time()-start_time))

            # data_exam(context_mention_ids,context_sentence_ids,entity2mention_table,graph_list)

            yield {'context_idxs': get_cuda(context_word_ids[:cur_bsz, :batch_max_length].contiguous()),
                   'context_pos': get_cuda(context_pos_ids[:cur_bsz, :batch_max_length].contiguous()),
                   'context_ner': get_cuda(context_ner_ids[:cur_bsz, :batch_max_length].contiguous()),
                   "context_mention": get_cuda(context_mention_ids[:cur_bsz, :batch_max_length].contiguous()),
                   "context_sent": get_cuda(context_sent_ids[:cur_bsz, :batch_max_length].contiguous()),
                   'context_ems_info': get_cuda(context_ems_info[:cur_bsz].contiguous()),
                   'context_word_mask': get_cuda(context_word_mask[:cur_bsz, :batch_max_length].contiguous()),
                   'context_word_length': get_cuda(context_word_length[:cur_bsz].contiguous()),
                   'ht_pair_distance': get_cuda(ht_pair_distance[:cur_bsz, :max_h_t_cnt]),
                   'ht_sent_distance': get_cuda(ht_sent_distance[:cur_bsz, :max_h_t_cnt]),
                   "graph_adj": get_cuda(batch_graph_adj[:cur_bsz]),
                   "graph_info":get_cuda(batch_graph_info[:cur_bsz]),
                   "graph_node_num": get_cuda(graph_node_num[:cur_bsz]),
                   "relation_path":get_cuda(relation_path[:cur_bsz, :max_h_t_cnt]),
                   'h_t_pairs': get_cuda(ht_pairs[:cur_bsz, :max_h_t_cnt, :2]),
                   'relation_label': get_cuda(relation_label[:cur_bsz, :max_h_t_cnt]).contiguous(),
                   'relation_multi_label': get_cuda(relation_multi_label[:cur_bsz, :max_h_t_cnt]),
                   'ign_relation_multi_label': get_cuda(ign_relation_multi_label[:cur_bsz, :max_h_t_cnt]),
                   'relation_mask': get_cuda(relation_mask[:cur_bsz, :max_h_t_cnt]),
                   'labels': label_list,
                   'L_vertex': L_vertex,
                   'titles': titles,
                   'indexes': indexes,
                   'overlaps': overlaps,
                   }
def mergy_all_path(h_idx,t_idx,meta_path,tree_path,entity2mention,path_type,n_path): 
    all_path = torch.zeros(3*n_path,4).cpu()
    if path_type == "meta":
        for path_type in range(3):
            path_cnt = 0
            for h_men in entity2mention[h_idx]:
                if path_cnt >=n_path:
                    break
                for t_men in entity2mention[t_idx]:
                    if path_cnt >=n_path:
                        break
                    for tmp in meta_path[(h_men,t_men)][path_type]:
                        all_path[path_type*n_path+path_cnt,:len(tmp)+2] = torch.Tensor([h_men]+tmp+[t_men])
                        path_cnt += 1
                        if path_cnt >=n_path:
                            break
    else:
        cnt = 0
        for h_men in entity2mention[h_idx]:
            for t_men in entity2mention[t_idx]:
                for p in tree_path[(h_men,t_men)]:
                    if cnt>=9:
                        break
                    all_path[cnt,:len(p)+1] = torch.Tensor(p+[t_men])
                    cnt+=1
    return all_path

def data_exam(context_mention,context_sentence,entity2mention_table,graph_list):
    for ind in range(context_mention.shape[0]):
        graph = graph_list[ind]
        sentence_id = context_sentence[ind]
        mention_id = context_mention[ind]
        table = entity2mention_table[ind].cpu()
        sentence_num = torch.max(sentence_id)
        for node_id in range(1,1+sentence_num):
            sent_span = torch.where(sentence_id==node_id)[0].numpy().tolist()
            a = graph.successors(node_id,etype="global")
            assert a.shape[0] == sentence_num-1
            a = graph.successors(node_id,etype="contain")
            for i in range(a.shape[0]):
                word_span = torch.where(mention_id==a[i])[0].numpy().tolist()
                for sp in word_span:
                    assert (sp in sent_span)
        for node_id in range(1+sentence_num,torch.max(mention_id)+1):
            a = graph.successors(node_id,etype="intra").cpu().numpy().tolist()
            a.append(node_id)
            entity_id = torch.where(table[:,node_id]>0)[0]
            mention_set = torch.where(table[entity_id,:]>0)[1].numpy().tolist()
            assert len(a)==len(mention_set)
            for sp in a:
                assert sp in mention_set

def GetHeterAdj(graph):
    adj = None
    cnt = 1
    for stype, etype, dtype in graph.canonical_etypes:
        if adj is not None:
            adj = adj + cnt * graph[(stype,etype,dtype)].adjacency_matrix().to_dense().numpy()
        else:
            adj = graph[(stype,etype,dtype)].adjacency_matrix().to_dense().numpy()
        cnt+=1
    return adj
            
            



