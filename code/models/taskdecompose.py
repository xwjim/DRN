#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
author: wxu
"""

from torch import nn, torch

class Task_Decompose(nn.Module):
    def __init__(self,config):
        super(Task_Decompose, self).__init__()

        self.config = config
        # pattern recongnize
        if config.use_dis_embed:
            self.dis_embed = nn.Embedding(20, 20)
            self.dis_sent_embed = nn.Embedding(20, 20)
            self.dis2idx = torch.zeros(512).cuda().long()
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

    def patten_recon(self,ins_path,path_info,graph_feature,context_feature):
        N_bt = ins_path.shape[0]
        N_pair = ins_path.shape[1]
        Meta_num = ins_path.shape[2]
        hidden_size = 2*graph_feature.shape[-1]

        b_ind,p_ind,m_ind = torch.where(torch.sum(ins_path,dim=-1)>0)
        path_select = ins_path[b_ind,p_ind,m_ind]

        graph_rep = torch.cat((graph_feature[b_ind,path_select[:,0]],graph_feature[b_ind,path_select[:,2]]),dim=-1)
        if self.config.use_dis_embed:
            hidden_size += 40
            delta_dis = path_info[b_ind,path_select[:,0],0] - path_info[b_ind,path_select[:,2],0]
            delta_dis = torch.where(delta_dis<0,-self.dis2idx[delta_dis]+self.dis_size // 2,self.dis2idx[delta_dis]+ self.dis_size // 2) 
            graph_rep = torch.cat((graph_rep,self.dis_embed(delta_dis)),dim=-1)
            sent_dis = path_info[b_ind,path_select[:,0],5] - path_info[b_ind,path_select[:,2],5]
            sent_dis = torch.where(delta_dis<0,-self.dis2idx[delta_dis]+self.dis_size // 2,self.dis2idx[delta_dis]+ self.dis_size // 2) 
            graph_rep = torch.cat((graph_rep,self.dis_sent_embed(sent_dis)),dim=-1)

        if context_feature is not None:
            hidden_size += 2*context_feature.shape[-1]
            a_start = path_info[b_ind,path_select[:,0],0]
            b_start = path_info[b_ind,path_select[:,2],0]
            con_rep = torch.cat((context_feature[b_ind,a_start],context_feature[b_ind,b_start]),dim=-1)
            m1_fea = torch.cat((graph_rep,con_rep),dim=-1)
        else:
            m1_fea = graph_rep

        out = torch.zeros(N_bt,N_pair,Meta_num,hidden_size).cuda()
        out[b_ind,p_ind,m_ind] = m1_fea

        return out

    def coreference_reason(self,ins_path,path_info,graph_feature,context_feature):
        N_bt = ins_path.shape[0]
        N_pair = ins_path.shape[1]
        Meta_num = ins_path.shape[2]
        hidden_size = 2*graph_feature.shape[-1]

        b_ind,p_ind,m_ind = torch.where(torch.sum(ins_path,dim=-1)>0)
        path_select = ins_path[b_ind,p_ind,m_ind]

        graph_rep = torch.cat((graph_feature[b_ind,path_select[:,0]],graph_feature[b_ind,path_select[:,3]]),dim=-1)
        if self.config.use_dis_embed:
            hidden_size += 40
            delta_dis = path_info[b_ind,path_select[:,0],0] - path_info[b_ind,path_select[:,3],0]
            delta_dis = torch.where(delta_dis<0,-self.dis2idx[delta_dis]+self.dis_size // 2,self.dis2idx[delta_dis]+ self.dis_size // 2) 
            graph_rep = torch.cat((graph_rep,self.dis_embed(delta_dis)),dim=-1)
            sent_dis = path_info[b_ind,path_select[:,0],5] - path_info[b_ind,path_select[:,2],5]
            sent_dis = torch.where(delta_dis<0,-self.dis2idx[delta_dis]+self.dis_size // 2,self.dis2idx[delta_dis]+ self.dis_size // 2) 
            graph_rep = torch.cat((graph_rep,self.dis_sent_embed(sent_dis)),dim=-1)

        if context_feature is not None:
            hidden_size += 2*context_feature.shape[-1]
            a_start = path_info[b_ind,path_select[:,0],0]
            b_start = path_info[b_ind,path_select[:,2],0]
            con_rep = torch.cat((context_feature[b_ind,a_start],context_feature[b_ind,b_start]),dim=-1)
            m2_fea = torch.cat((graph_rep,con_rep),dim=-1)
        else:
            m2_fea = graph_rep

        out = torch.zeros(N_bt,N_pair,Meta_num,hidden_size).cuda()
        out[b_ind,p_ind,m_ind] = m2_fea

        return out
    def logical_reason(self,ins_path,path_info,graph_feature,context_feature):
        N_bt = ins_path.shape[0]
        N_pair = ins_path.shape[1]
        Meta_num = ins_path.shape[2]
        hidden_size = 2*graph_feature.shape[-1]

        b_ind,p_ind,m_ind = torch.where(torch.sum(ins_path,dim=-1)>0)
        path_select = ins_path[b_ind,p_ind,m_ind]

        graph_rep = torch.cat((graph_feature[b_ind,path_select[:,0]],graph_feature[b_ind,path_select[:,3]]),dim=-1)
        if self.config.use_dis_embed:
            hidden_size += 40
            delta_dis = path_info[b_ind,path_select[:,0],0] - path_info[b_ind,path_select[:,3],0]
            delta_dis = torch.where(delta_dis<0,-self.dis2idx[delta_dis]+self.dis_size // 2,self.dis2idx[delta_dis]+ self.dis_size // 2) 
            graph_rep = torch.cat((graph_rep,self.dis_embed(delta_dis)),dim=-1)
            sent_dis = path_info[b_ind,path_select[:,0],5] - path_info[b_ind,path_select[:,2],5]
            sent_dis = torch.where(delta_dis<0,-self.dis2idx[delta_dis]+self.dis_size // 2,self.dis2idx[delta_dis]+ self.dis_size // 2) 
            graph_rep = torch.cat((graph_rep,self.dis_sent_embed(sent_dis)),dim=-1)

        if context_feature is not None:
            hidden_size += 2*context_feature.shape[-1]
            a_start = path_info[b_ind,path_select[:,0],0]
            b_start = path_info[b_ind,path_select[:,1],0]
            c_start = path_info[b_ind,path_select[:,2],0]
            d_start = path_info[b_ind,path_select[:,3],0]
            con_rep = torch.cat((context_feature[b_ind,a_start]+context_feature[b_ind,b_start],context_feature[b_ind,c_start]+context_feature[b_ind,d_start]),dim=-1)
            m3_fea = torch.cat((graph_rep,con_rep),dim=-1)
        else:
            m3_fea = graph_rep

        out = torch.zeros(N_bt,N_pair,Meta_num,hidden_size).cuda()
        out[b_ind,p_ind,m_ind] = m3_fea

        return out

    def forward(self,relation_path,path_info,graph_feature,context_feature=None):

        ## Pattern Recongnize
        m1_path = relation_path[...,0:self.config.path_per_type,:]
        m1_mask = torch.sum(m1_path>0,dim=-1)>0
        m1_fea = self.patten_recon(m1_path,path_info,graph_feature,context_feature)

        ## Coreference Reason
        m2_path = relation_path[...,self.config.path_per_type:2*self.config.path_per_type,:]
        m2_mask = torch.sum(m2_path>0,dim=-1)>0
        m2_fea = self.coreference_reason(m2_path,path_info,graph_feature,context_feature)

        ## Logical Recongnize
        m3_path = relation_path[...,2*self.config.path_per_type:3*self.config.path_per_type,:]
        m3_mask = torch.sum(m3_path>0,dim=-1)>0
        m3_fea = self.logical_reason(m3_path,path_info,graph_feature,context_feature)
        ##
        path_fea = torch.cat((m1_fea,m2_fea,m3_fea),dim=-2)
        mask = torch.cat((m1_mask,m2_mask,m3_mask),dim=-1)

        return path_fea,mask

