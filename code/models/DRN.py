import numpy as np
import torch
import torch.nn as nn
from transformers import *
from utils import get_cuda,mergy_span_token,BiLSTM
from models.taskdecompose import Task_Decompose
from models.graph import GraphConvolutionLayer,GraphAttentionLayer,GraphMultiHeadAttention
from .attention import MultiHeadAttention


class DRN_GloVe(nn.Module):
    def __init__(self, config):
        super(DRN_GloVe, self).__init__()
        self.config = config

        word_emb_size = config.word_emb_size
        vocabulary_size = config.vocabulary_size
        encoder_input_size = word_emb_size
        self.activation = nn.Tanh() if config.activation == 'tanh' else nn.ReLU()

        self.word_emb = nn.Embedding(vocabulary_size, word_emb_size, padding_idx=config.word_pad)
        if config.pre_train_word:
            self.word_emb = nn.Embedding(config.data_word_vec.shape[0], word_emb_size, padding_idx=config.word_pad)
            self.word_emb.weight.data.copy_(torch.from_numpy(config.data_word_vec[:, :word_emb_size]))

        self.word_emb.weight.requires_grad = config.finetune_word
        if config.use_entity_type:
            encoder_input_size += config.entity_type_size
            self.entity_type_emb = nn.Embedding(config.entity_type_num, config.entity_type_size,
                                                padding_idx=config.entity_type_pad)

        if config.use_entity_id:
            encoder_input_size += config.entity_id_size
            self.entity_id_emb = nn.Embedding(config.max_entity_num + 1, config.entity_id_size,
                                              padding_idx=config.entity_id_pad)

        self.encoder = BiLSTM(encoder_input_size,config.lstm_hidden_size,config.nlayers,config.lstm_dropout,word_pad=config.word_pad)

        self.gcn_dim = config.gcn_dim
        assert self.gcn_dim == 2 * config.lstm_hidden_size, 'gcn dim should be the lstm hidden dim * 2'


        self.bank_size = self.gcn_dim
        if self.config.use_graph:
            self.rel_name_lists = ['intra', 'inter', 'global', "contain"]
            self.graph_reason = GraphReasonLayer(self.rel_name_lists, self.gcn_dim, self.gcn_dim,2,
                                        graph_type=config.graph_type,graph_drop=self.config.graph_dropout)

            self.bank_size = self.gcn_dim * (self.config.gcn_layers + 1)

        class_size = self.bank_size * 2
        if self.config.use_dis_embed:
            self.dis_embed = nn.Embedding(20, 20)
            class_size = class_size + 20

        if self.config.use_context:
            self.context_atten = MultiHeadAttention(1,self.gcn_dim,self.gcn_dim,self.gcn_dim,dropout=0.4)
            class_size = self.bank_size * 2 + 2*self.gcn_dim
            if self.config.use_dis_embed:
                class_size = class_size + 40
        else:
            class_size = self.bank_size * 2 + 40
        self.task_infer = Task_Decompose(config)

        self.predict = nn.Sequential(
            nn.Linear(class_size , self.bank_size * 2),  #
            self.activation,
            nn.Dropout(self.config.output_dropout),
            nn.Linear(self.bank_size * 2, config.relation_nums),
        )

    def forward(self, **params):
        '''
            words: [batch_size, max_length]
            src_lengths: [batchs_size]
            mask: [batch_size, max_length]
            entity_type: [batch_size, max_length]
            entity_id: [batch_size, max_length]
            mention_id: [batch_size, max_length]
            distance: [batch_size, max_length]
            entity2mention_table: list of [local_entity_num, local_mention_num]
            graphs: list of DGLHeteroGraph
            h_t_pairs: [batch_size, h_t_limit, 2]
        '''
        src = self.word_emb(params['words'])
        mask = params['mask']
        bsz, slen, _ = src.size()

        if self.config.use_entity_type:
            src = torch.cat([src, self.entity_type_emb(params['entity_type'])], dim=-1)

        if self.config.use_entity_id:
            src = torch.cat([src, self.entity_id_emb(params['entity_id'])], dim=-1)

        # src: [batch_size, slen, encoder_input_size]
        # src_lengths: [batchs_size]

        encoder_outputs, (output_h_t, _) = self.encoder(src, params['src_lengths'])
        encoder_outputs[mask == 0] = 0
        # encoder_outputs: [batch_size, slen, 2*encoder_hid_size]
        # output_h_t: [batch_size, 2*encoder_hid_size]

        graph_node_num = params["graph_node_num"]
        max_node = torch.max(graph_node_num).item()
        graph_adj = params['graph_adj'][:,:max_node,:max_node]
        graph_info = params["graph_info"]
        graph_fea = get_cuda(torch.zeros(bsz, max_node, self.gcn_dim))
        graph_fea.zero_()
        
        for i in range(graph_adj.shape[0]):
            encoder_output = encoder_outputs[i]  # [slen, 2*encoder_hid_size]
            info = graph_info[i]
            node_num = graph_node_num[i]
            graph_fea[i, :node_num] = mergy_span_token(encoder_output,info[:node_num,:2])

        if self.config.use_graph:
            assert torch.max(graph_adj) <= len(self.rel_name_lists)
            graph_feature = self.graph_reason(graph_fea,graph_adj)
        else:
            graph_feature = graph_fea

        # mention -> entity
        ems_info = params["context_ems_info"]
        entity_num = torch.max(torch.sum(ems_info[...,0]==1,dim=-1)).item()
        entity_bank = get_cuda(torch.Tensor(bsz, entity_num, self.bank_size))
        entity_bank.zero_()

        for i in range(graph_adj.shape[0]):
            # average mention -> entity
            s_num = torch.sum(graph_info[i,...,3]==1)
            m_num = torch.sum(graph_info[i,...,3]==2)
            e_num = torch.sum(ems_info[i,...,0]==1)
            e_id,m_id = torch.broadcast_tensors(torch.arange(e_num)[...,None].cuda(),\
                                                graph_info[i,s_num:s_num+m_num,2][None,...].cuda())
            select_metrix = (e_id == m_id).float()
            mention_nums = torch.sum(select_metrix, dim=-1).unsqueeze(-1).expand(-1, select_metrix.size(1))
            select_metrix = torch.where(mention_nums > 0, select_metrix / mention_nums, select_metrix)
            entity_bank[i,:e_num] = torch.mm(select_metrix, graph_feature[i,s_num:s_num+m_num])

        h_t_pairs = params['h_t_pairs']
        h_t_limit = h_t_pairs.size(1)

        # [batch_size, h_t_limit, bank_size]
        b_ind = torch.arange(h_t_pairs.shape[0]).cuda()[...,None].repeat(1,h_t_limit)
        h_entity = entity_bank[b_ind,h_t_pairs[...,0]]
        t_entity = entity_bank[b_ind,h_t_pairs[...,1]]

        if self.config.use_dis_embed:
            rel_embed = torch.cat((h_entity,t_entity,self.dis_embed(params["ht_pair_distance"])),dim=-1)
        else:
            rel_embed = torch.cat((h_entity, t_entity),dim=-1)

        
        if self.config.use_context:
            length_ind = torch.arange(encoder_outputs.shape[-2]).cuda().unsqueeze(0).expand(bsz,-1)
            context_mask = (length_ind<params['src_lengths'].unsqueeze(-1)).unsqueeze(-2)
            context_feature,_ = self.context_atten(encoder_outputs,encoder_outputs,encoder_outputs,mask=context_mask)
        else:
            context_feature = None

        relation_path = params["relation_path"]
        path_fea,path_mask = self.task_infer(relation_path,graph_info,graph_feature,context_feature)
        scores = self.predict(path_fea)
        scores_mask = scores.masked_fill(~path_mask.unsqueeze(-1),-1e9)
        predictions,task_select = torch.max(scores_mask,dim=-2)
        return {"predictions":predictions,"task_select":task_select}

class DRN_BERT(nn.Module):
    def __init__(self, config):
        super(DRN_BERT, self).__init__()
        self.config = config
        if config.activation == 'tanh':
            self.activation = nn.Tanh()
        elif config.activation == 'relu':
            self.activation = nn.ReLU()
        elif config.activation == "prelu":
            self.activation = nn.PReLU()
            self.config.activation = "relu"
        else:
            assert 1 == 2, "you should provide activation function."

        hidden = config.bert_hid_size
        if config.use_entity_type:
            hidden += config.entity_type_size
            self.entity_type_emb = nn.Embedding(config.entity_type_num, config.entity_type_size,
                                                padding_idx=config.entity_type_pad)
        if config.use_entity_id:
            hidden += config.entity_id_size
            self.entity_id_emb = nn.Embedding(config.max_entity_num + 1, config.entity_id_size,
                                              padding_idx=config.entity_id_pad)

        if config.use_sent_id:
            self.sent_id_emb = nn.Embedding(config.max_entity_num + 1, config.sent_id_size,
                                              padding_idx=config.sent_id_pad)
            hidden += config.sent_id_size

        bertconfig = AutoConfig.from_pretrained(config.bert_path)
        self.bert = AutoModel.from_pretrained(config.bert_path,config=bertconfig)

        if config.bert_fix:
            for p in self.bert.parameters():
                p.requires_grad = False

        
        self.gcn_dim = hidden
        # assert self.gcn_dim == config.bert_hid_size + config.entity_id_size + config.entity_type_size

        self.bank_size = self.gcn_dim
        if self.config.use_graph:
            self.rel_name_lists = ['intra', 'inter', 'global', "contain"]
            self.graph_reason = GraphReasonLayer(self.rel_name_lists, self.gcn_dim, self.gcn_dim,2,
                                    graph_type=config.graph_type,graph_drop=self.config.graph_dropout)
            self.bank_size = self.gcn_dim * (self.config.gcn_layers + 1)

        class_size = self.bank_size * 2
        if self.config.use_dis_embed:
            self.dis_embed = nn.Embedding(20, 20)
            class_size = class_size + 20

        if self.config.use_context:
            self.context_atten = MultiHeadAttention(1,self.gcn_dim,self.gcn_dim,self.gcn_dim,dropout=0.4)
            class_size = self.bank_size * 2 + 2*self.gcn_dim
            if self.config.use_dis_embed:
                class_size = class_size + 40
        else:
            class_size = self.bank_size * 2 + 40
        self.task_infer = Task_Decompose(config)

        self.predict = nn.Sequential(
            nn.Linear(class_size, self.bank_size * 2),
            self.activation,
            nn.Dropout(self.config.output_dropout),
            nn.Linear(self.bank_size * 2, config.relation_nums),
        )

    def forward(self, **params):
        '''
        words: [batch_size, max_length]
        src_lengths: [batchs_size]
        mask: [batch_size, max_length]
        entity_type: [batch_size, max_length]
        entity_id: [batch_size, max_length]
        mention_id: [batch_size, max_length]
        distance: [batch_size, max_length]
        entity2mention_table: list of [local_entity_num, local_mention_num]
        graphs: list of DGLHeteroGraph
        h_t_pairs: [batch_size, h_t_limit, 2]
        ht_pair_distance: [batch_size, h_t_limit]
        '''
        words = params['words']
        mask = params['mask']
        bsz, slen = words.size()

        bert_output = self.bert(input_ids=words, attention_mask=mask,output_hidden_states=True,return_dict=True)
        encoder_outputs = bert_output.last_hidden_state
        # encoder_outputs[mask == 0] = 0

        if self.config.use_entity_type:
            encoder_outputs = torch.cat([encoder_outputs, self.entity_type_emb(params['entity_type'])], dim=-1)

        if self.config.use_entity_id:
            encoder_outputs = torch.cat([encoder_outputs, self.entity_id_emb(params['entity_id'])], dim=-1)

        if self.config.use_sent_id:
            encoder_outputs = torch.cat([encoder_outputs, self.sent_id_emb(params['sentence_id'])], dim=-1)

        # encoder_outputs: [batch_size, slen, bert_hid+type_size+id_size]

        graph_node_num = params["graph_node_num"]
        max_node = torch.max(graph_node_num).item()
        graph_adj = params['graph_adj'][:,:max_node,:max_node]
        graph_fea = get_cuda(torch.zeros(bsz, max_node, encoder_outputs.shape[-1]))
        graph_fea.zero_()
        graph_info = params["graph_info"]
        
        for i in range(graph_adj.shape[0]):
            encoder_output = encoder_outputs[i]  # [slen, bert_hid]
            info = graph_info[i]
            node_num = graph_node_num[i]
            graph_fea[i, :node_num] = mergy_span_token(encoder_output,info[:node_num,:2])

        if self.config.use_graph:
            assert torch.max(graph_adj) <= len(self.rel_name_lists)
            graph_feature = self.graph_reason(graph_fea,graph_adj)
        else:
            graph_feature = graph_fea

        # mention -> entity
        ems_info = params["context_ems_info"]
        entity_num = torch.max(torch.sum(ems_info[...,0]==1,dim=-1)).item()
        entity_bank = get_cuda(torch.Tensor(bsz, entity_num, self.bank_size))
        entity_bank.zero_()

        for i in range(graph_adj.shape[0]):
            # average mention -> entity
            s_num = torch.sum(graph_info[i,...,3]==1)
            m_num = torch.sum(graph_info[i,...,3]==2)
            e_num = torch.sum(ems_info[i,...,0]==1)
            e_id,m_id = torch.broadcast_tensors(torch.arange(e_num)[...,None].cuda(),\
                                                graph_info[i,s_num:s_num+m_num,2][None,...].cuda())
            select_metrix = (e_id == m_id).float()
            mention_nums = torch.sum(select_metrix, dim=-1).unsqueeze(-1).expand(-1, select_metrix.size(1))
            select_metrix = torch.where(mention_nums > 0, select_metrix / mention_nums, select_metrix)
            entity_bank[i,:e_num] = torch.mm(select_metrix, graph_feature[i,s_num:s_num+m_num])

        h_t_pairs = params['h_t_pairs']
        h_t_limit = h_t_pairs.size(1)

        b_ind = torch.arange(h_t_pairs.shape[0]).cuda()[...,None].repeat(1,h_t_limit)
        h_entity = entity_bank[b_ind,h_t_pairs[...,0]]
        t_entity = entity_bank[b_ind,h_t_pairs[...,1]]

        if self.config.use_dis_embed:
            rel_embed = torch.cat((h_entity, t_entity,self.dis_embed(params["ht_pair_distance"])),dim=-1)
        else:
            rel_embed = torch.cat((h_entity, t_entity),dim=-1)

        if self.config.use_context:
            length_ind = torch.arange(encoder_outputs.shape[-2]).cuda().unsqueeze(0).expand(bsz,-1)
            context_mask = (length_ind<params['src_lengths'].unsqueeze(-1)).unsqueeze(-2)
            context_feature,_ = self.context_atten(encoder_outputs,encoder_outputs,encoder_outputs,mask=context_mask)
        else:
            context_feature = None

        relation_path = params["relation_path"]
        path_fea,path_mask = self.task_infer(relation_path,graph_info,graph_feature,context_feature)
        scores = self.predict(path_fea)
        scores_mask = scores.masked_fill(~path_mask.unsqueeze(-1),-1e9)
        predictions,task_select = torch.max(scores_mask,dim=-2)
        return {"predictions":predictions,"task_select":task_select}


class GraphReasonLayer(nn.Module):
    def __init__(self,edges,input_size,out_size,iters,graph_type="gat",graph_drop=0.0,graph_head=4):
        super(GraphReasonLayer, self).__init__()
        # self.W = nn.Parameter(nn.init.normal_(torch.empty(input_size, input_size)), requires_grad=True)
        # self.W_node = nn.ModuleList([nn.Linear(input_size,input_size) for i in range(iters)])
        # self.W_sum = nn.ModuleList([nn.Linear(input_size,input_size) for i in range(iters)])
        self.iters = iters
        self.edges = edges
        self.graph_type = graph_type
        if graph_type == "gat":
            self.block = nn.ModuleList([GraphAttentionLayer(edges,input_size,input_size,graph_drop=graph_drop) for i in range(iters)])
        elif graph_type == "gcn":
            self.block = nn.ModuleList([GraphConvolutionLayer(edges,input_size,input_size,graph_drop) for i in range(iters)])
        else:
            raise("graph choose error")
        
    def forward(self, nodes_embed,node_adj):

        hi = nodes_embed
        for cnt in range(0, self.iters):
            hi = self.block[cnt](hi,node_adj)
            nodes_embed = torch.cat((nodes_embed,hi),dim=-1)
    
        return nodes_embed
