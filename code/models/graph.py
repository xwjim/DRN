import numpy as np
import torch
import torch.nn as nn
from transformers import *
import torch.nn.functional as F

class GraphConvolutionLayer(nn.Module):
    def __init__(self,edges,input_size,hidden_size,graph_drop):
        super(GraphConvolutionLayer, self).__init__()
        self.W = nn.Parameter(torch.Tensor(size=(input_size, hidden_size)))
        nn.init.xavier_uniform_(self.W, gain=nn.init.calculate_gain('relu'))
        self.edges = edges
        self.W_edge = nn.ModuleList([nn.Linear(hidden_size,hidden_size,bias=False) for i in (self.edges)])
        for m in self.W_edge:
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))
        nn.init.zeros_(self.bias)
        self.loop_weight = nn.Parameter(torch.Tensor(input_size, hidden_size))
        nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
        self.drop = torch.nn.Dropout(p=graph_drop, inplace=False)

    def forward(self, nodes_embed,node_adj):

        N_bt = nodes_embed.shape[0]
        N = nodes_embed.shape[1]
        h = torch.matmul(nodes_embed,self.W.unsqueeze(0))
        sum_nei = torch.zeros_like(h)
        for edge_type in range(len(self.edges)):
            mask = (node_adj==(edge_type+1)).float()
            sum_nei += torch.matmul(mask,self.W_edge[edge_type](h))
        degs = torch.sum(node_adj>0,dim=-1).float().clamp(min=1).unsqueeze(dim=-1)
        norm = 1.0 / degs
        dst = sum_nei*norm + self.bias
        dst = dst + torch.matmul(nodes_embed, self.loop_weight)
        out = self.drop(torch.relu(dst))
        return out

class GraphMultiHeadAttention(nn.Module):
    def __init__(self,edges,input_size,hidden_size,nhead=4,graph_drop=0.0):
        super(GraphMultiHeadAttention, self).__init__()
        assert hidden_size%nhead == 0
        ho = int(hidden_size/nhead)
        self.head_graph = nn.ModuleList([GraphAttentionLayer(edges,input_size,ho,graph_drop) for _ in range(nhead)])
        self.nhead = nhead
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)

    def forward(self, nodes_embed,node_adj):

        x = []
        for cnt in range(0, self.nhead):
            x.append(self.head_graph[cnt](nodes_embed,node_adj))
    
        return torch.cat(x,dim=-1)

class GraphAttentionLayer(nn.Module):
    def __init__(self,edges,input_size,hidden_size,graph_drop):
        super(GraphAttentionLayer, self).__init__()
        self.W = nn.Parameter(torch.Tensor(size=(input_size, hidden_size)))
        nn.init.xavier_uniform_(self.W, gain=nn.init.calculate_gain('relu'))
        self.edges = edges
        self.W_edge = nn.ModuleList([nn.Linear(2*hidden_size,1,bias=False) for i in (self.edges)])
        for m in self.W_edge:
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        
        
        self.bias = nn.Parameter(torch.Tensor(hidden_size))
        nn.init.zeros_(self.bias)
        self.self_loop = False
        self.loop_weight = nn.Linear(hidden_size, 1, bias=False)
        self.hidden_size = hidden_size
        nn.init.xavier_uniform_(self.loop_weight.weight, gain=nn.init.calculate_gain('relu'))
        self.drop = torch.nn.Dropout(p=graph_drop, inplace=False)

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, nodes_embed,node_adj):

        N_bt = nodes_embed.shape[0]
        N = nodes_embed.shape[1]
        h = torch.matmul(nodes_embed,self.W.unsqueeze(0))
        a_input = torch.cat([h.repeat(1,1,N).view(N_bt,N * N, -1), h.repeat(1,N,1)], dim=-1)
        weight = torch.zeros(N_bt,N*N).cuda()
        for edge_type in range(len(self.edges)):
            mask = (node_adj==(edge_type+1)).float().view(N_bt,-1)
            weight += mask * self.W_edge[edge_type](a_input).squeeze(dim=-1)
        
        if self.self_loop:
            sl_mask = torch.zeros_like(node_adj)
            sl_mask[:,torch.arange(node_adj.shape[1]).cuda(),torch.arange(node_adj.shape[2]).cuda()] = 1
            sl_mask = sl_mask.view(N_bt,-1)
            weight += mask * self.loop_weight(a_input[...,:self.hidden_size]).squeeze(dim=-1)

        weight = F.leaky_relu(weight).view(N_bt,N,N)
        weight = weight.masked_fill(node_adj==0, -1e9)
        attention = F.softmax(weight, dim=-1)
        dst = torch.matmul(attention, h) + self.bias

        out = self.drop(torch.relu(dst)) + h
        return self.layer_norm(out)

