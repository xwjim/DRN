#! /bin/bash
export CUDA_VISIBLE_DEVICES=$1

# -------------------GAIN_GloVe Training Shell Script--------------------

model_name=DRN_GloVe
lr=0.001
batch_size=8
test_batch_size=4
epoch=200
test_epoch=1
log_step=20
save_model_freq=100
negativa_alpha=-1
weight_decay=0.00 #01

python3 -m pdb train.py \
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
  --graph_type gcn \
  --activation relu \
  --use_dis_embed \
  --use_graph \
  --use_context \
  --no_na_loss \
  --steplr \
  --weight_decay ${weight_decay} 

# -------------------additional options--------------------

# option below is used to resume training, it should be add into the shell scripts above
# --pretrain_model checkpoint/GAIN_GloVe_10.pt \
