#! /bin/bash
export CUDA_VISIBLE_DEVICES=$1

# -------------------GAIN_GloVe Training Shell Script--------------------

model_name=DRN_Glove
lr=0.001
batch_size=12
test_batch_size=5
epoch=200
test_epoch=1
log_step=20
save_model_freq=100
negativa_alpha=-1
weight_decay=0.0001

nohup python3 -u train.py \
  --dataset docred \
  --use_model bilstm \
  --model_name ${model_name} \
  --lr ${lr} \
  --batch_size ${batch_size} \
  --test_batch_size ${test_batch_size} \
  --epoch ${epoch} \
  --test_epoch ${test_epoch} \
  --log_step ${log_step} \
  --save_model_freq ${save_model_freq} \
  --negativa_alpha ${negativa_alpha} \
  --word_emb_size 100 \
  --pre_train_word \
  --gcn_dim 256 \
  --gcn_layers 2 \
  --lstm_hidden_size 128 \
  --use_entity_type \
  --use_entity_id \
  --finetune_word \
  --activation relu \
  --graph_type gcn \
  --use_context \
  --use_graph \
  --no_na_loss \
  --steplr \
  --use_dis_embed \
  --use_wandb \
  --wandb_name origin \
  --weight_decay ${weight_decay} \
  >logs/train_${model_name}.log 2>&1 &
