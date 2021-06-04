import argparse
import json
import os

import numpy as np

class path_set():
    # datasets path
    def __init__(self):
        self.docred_train_set = 'train_annotated.json'
        self.docred_dev_set ="dev.json"
        self.docred_test_set='test.json'
        self.docred_relation_nums=97

def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default="docred", choices=['docred'],
                        help='you should choose between docred and wiki80')
    parser.add_argument('--data_dir', type=str, default="../data")
    parser.add_argument('--prepro_dir', type=str, default="../data/prepro_data")
    # datasets path
    parser.add_argument('--train_set', type=str, default='train_annotated.json')
    parser.add_argument('--dev_set', type=str, default='dev.json')
    parser.add_argument('--test_set', type=str, default='test.json')
    parser.add_argument('--test_type', type=str, default='dev')
    parser.add_argument('--train_set_save', type=str, default='train.pkl')
    parser.add_argument('--dev_set_save', type=str, default='dev.pkl')
    parser.add_argument('--test_set_save', type=str, default='test.pkl')
    parser.add_argument('--train_set_glove_save', type=str, default='train_glove.pkl')
    parser.add_argument('--dev_set_glove_save', type=str, default='dev_glove.pkl')
    parser.add_argument('--test_set_glove_save', type=str, default='test_glove.pkl')
    parser.add_argument('--train_set_bert_save', type=str, default='train_bert.pkl')
    parser.add_argument('--dev_set_bert_save', type=str, default='dev_bert.pkl')
    parser.add_argument('--test_set_bert_save', type=str, default='test_bert.pkl')
    parser.add_argument('--relation_nums', type=int, default=97)

    # checkpoints
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
    parser.add_argument('--model_name', type=str, default='train_model')
    parser.add_argument('--pretrain_model', type=str, default='')
    parser.add_argument('--no_na_loss', action='store_true')

    # task/Dataset-related
    parser.add_argument('--vocabulary_size', type=int, default=200000)
    parser.add_argument('--entity_type_num', type=int, default=7)
    parser.add_argument('--max_entity_num', type=int, default=80)

    # padding
    parser.add_argument('--word_pad', type=int, default=0)
    parser.add_argument('--entity_type_pad', type=int, default=0)
    parser.add_argument('--entity_id_pad', type=int, default=0)
    parser.add_argument('--sent_id_pad', type=int, default=0)

    # word embedding
    parser.add_argument('--word_emb_size', type=int, default=100)
    parser.add_argument('--pre_train_word', action='store_true')
    parser.add_argument('--data_word_vec', type=str)
    parser.add_argument('--finetune_word', action='store_true')

    # entity type embedding
    parser.add_argument('--use_entity_type', action='store_true')
    parser.add_argument('--entity_type_size', type=int, default=20)

    # entity id embedding, i.e., coreference embedding in DocRED original paper
    parser.add_argument('--use_entity_id', action='store_true')
    parser.add_argument('--entity_id_size', type=int, default=20)

    # sent id embedding, i.e., coreference embedding in DocRED original paper
    parser.add_argument('--use_sent_id', action='store_true')
    parser.add_argument('--sent_id_size', type=int, default=20)

    # BiLSTM
    parser.add_argument('--nlayers', type=int, default=1)
    parser.add_argument('--lstm_hidden_size', type=int, default=32)
    parser.add_argument('--lstm_dropout', type=float, default=0.5)

    # training settings
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--bert_lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--test_epoch', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--negativa_alpha', type=float, default=0.0)  # negative example nums v.s positive example num
    parser.add_argument('--log_step', type=int, default=50)
    parser.add_argument('--save_model_freq', type=int, default=10)

    # gcn
    parser.add_argument('--gcn_layers', type=int, default=2)
    parser.add_argument('--gcn_dim', type=int, default=808)
    parser.add_argument('--graph_dropout', type=float, default=0.5)
    parser.add_argument('--activation', type=str, default="relu")
    parser.add_argument('--output_dropout', type=float, default=0.5)

    # BERT
    parser.add_argument('--bert_hid_size', type=int, default=768)
    parser.add_argument('--bert_path', type=str, default="")
    parser.add_argument('--bert_fix', action='store_true')
    parser.add_argument('--coslr', action='store_true')
    parser.add_argument('--steplr', action='store_true')
    parser.add_argument('--dynlr', action='store_true')
    parser.add_argument('--linearlr', action='store_true')
    parser.add_argument('--clip', type=float, default=-1)
    parser.add_argument('--warmup_epoch', type=int, default=10)
    parser.add_argument('--path_per_type', type=int, default=3)

    parser.add_argument('--k_fold', type=str, default="none")

    # other
    parser.add_argument('--use_dis_embed', action='store_true', default=False)
    parser.add_argument('--path_type', type=str, default="meta")
    parser.add_argument('--graph_type', type=str, default="gcn")
    parser.add_argument('--load_model', action='store_true')

    parser.add_argument('--use_context', action='store_true', default=False)
    parser.add_argument('--use_graph', action='store_true', default=False)

    # use BiLSTM / BERT encoder, default: BiLSTM encoder
    parser.add_argument('--use_model', type=str, default="bilstm", choices=['bilstm', 'bert'],
                        help='you should choose between bert and bilstm')

    # binary classification threshold, automatically find optimal threshold when -1
    parser.add_argument('--input_theta', type=float, default=-1)

    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--wandb_name', type=str, default="")

    args = parser.parse_args()

    path_opt = path_set()

    if args.dataset == "docred":
        args.data_dir = os.path.join(args.data_dir,args.dataset)
        args.prepro_dir = os.path.join(args.data_dir,"prepro_data")
        args.train_set = os.path.join(args.data_dir,path_opt.docred_train_set)
        args.udata_set = os.path.join(args.data_dir,"train_distant.json")
        args.dev_set = os.path.join(args.data_dir,path_opt.docred_dev_set)
        args.test_set = os.path.join(args.data_dir,path_opt.docred_test_set)
        args.relation_nums = path_opt.docred_relation_nums
    else:
        pass
    if args.use_model == "bilstm":
        args.train_set_save = os.path.join(args.prepro_dir,args.train_set_glove_save)
        args.dev_set_save = os.path.join(args.prepro_dir,args.dev_set_glove_save)
        args.test_set_save = os.path.join(args.prepro_dir,args.test_set_glove_save)
        args.entity_type_size = 20
        args.entity_id_size = 20
        args.sent_id_size = 20
    elif args.use_model == "bert":
        args.train_set_save = os.path.join(args.prepro_dir,"train_"+"_".join(args.bert_path.split("-"))+".pkl")
        args.dev_set_save = os.path.join(args.prepro_dir,"dev_"+"_".join(args.bert_path.split("-"))+".pkl")
        args.test_set_save = os.path.join(args.prepro_dir,"test_"+"_".join(args.bert_path.split("-"))+".pkl")
        args.entity_type_size = 20
        args.entity_id_size = 20
        args.sent_id_size = 20
    else:
        raise("Error")
    if not os.path.exists(args.prepro_dir):
        os.mkdir(args.prepro_dir)
    if args.use_wandb:
        import wandb
        wandb_name = args.wandb_name
        if args.use_graph:
            if wandb_name == "":
                wandb_name = args.graph_type
            else:
                wandb_name += "_" + args.graph_type
        else:
            if wandb_name == "":
                wandb_name = "dir"
            else:
                wandb_name += "_" + "dir"
        wandb_name += "_" + args.use_model
        if args.use_context:
            wandb_name += "_context"
        if args.negativa_alpha>0:
            wandb_name += "_s" + str(args.negativa_alpha)

        wandb.init(name=wandb_name,project="docre",config=args.__dict__)

    return args
