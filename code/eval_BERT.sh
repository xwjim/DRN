#! /bin/bash
export CUDA_VISIBLE_DEVICES=$1

# binary classification threshold, automatically find optimal threshold when -1, default:-1
input_theta=${2--1}
batch_size=5
test_batch_size=5
dataset=dev

model_name=DRN_BERT_base

python3 -m pdb test.py \
  --dataset docred \
  --model_name ${model_name} \
  --use_model bert \
  --pretrain_model checkpoint/DRN_BERT_base_best.pt \
  --batch_size ${batch_size} \
  --test_batch_size ${test_batch_size} \
  --gcn_dim 808 \
  --gcn_layers 2 \
  --bert_hid_size 768 \
  --bert_path bert-base-uncased \
  --use_entity_type \
  --use_entity_id \
  --use_graph \
  --use_dis_embed \
  --use_context \
  --graph_type gcn \
  --activation relu \
  --test_type ${dataset} \
  --input_theta ${input_theta}
