#! /bin/bash
export CUDA_VISIBLE_DEVICES=$1

# -------------------GAIN_BERT_base Training Shell Script--------------------

model_name=DRN_BERT_base
lr=1e-3
bert_lr=1e-5
batch_size=3
test_batch_size=2
epoch=300
test_epoch=1
log_step=20
save_model_freq=5
negativa_alpha=4
clip=-1

python3 -m pdb train.py \
  --dataset docred \
  --use_model bert \
  --model_name ${model_name} \
  --lr ${lr} \
  --bert_lr ${bert_lr} \
  --batch_size ${batch_size} \
  --clip ${clip} \
  --test_batch_size ${test_batch_size} \
  --epoch ${epoch} \
  --test_epoch ${test_epoch} \
  --log_step ${log_step} \
  --save_model_freq ${save_model_freq} \
  --negativa_alpha ${negativa_alpha} \
  --gcn_dim 808 \
  --gcn_layers 2 \
  --bert_hid_size 768 \
  --bert_path bert-base-uncased \
  --use_entity_type \
  --use_entity_id \
  --graph_type gcn \
  --activation relu \
  --use_dis_embed \
  --use_graph \
  --use_context \
  --coslr 
