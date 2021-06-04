import sklearn.metrics
import torch

from config import *
from data import DGLREDataset, DGLREDataloader, BERTDGLREDataset
from models.DRN import DRN_GloVe, DRN_BERT
from utils import get_cuda, logging, print_params,logging, print_params, Metrics
from torch import nn
from collections import  Counter,defaultdict
import pickle


def test(model, dataloader, modelname, id2rel, output_file=False, test_prefix='dev',lr_rate=0,global_step=0,config=None):
    # ours: inter-sentence F1 in LSR

    relation_num = config.relation_nums
    input_theta = config.input_theta

    BCELogit = nn.BCEWithLogitsLoss(reduction='none')


    test_result = []
    test_metric = Metrics("Re Test",logging,use_wandb=config.use_wandb)
    test_metric.reset()

    theta_list = input_theta*np.ones((1,relation_num),np.float32)
    
    for cur_i, d in enumerate(dataloader):
        # print('step: {}/{}'.format(cur_i, total_steps))

        with torch.no_grad():
            relation_multi_label = d['relation_multi_label']
            ign_relation_multi_label = d['ign_relation_multi_label']
            relation_mask = d['relation_mask']
            labels = d['labels']
            L_vertex = d['L_vertex']
            titles = d['titles']
            indexes = d['indexes']
            overlaps = d['overlaps']
            relation_path = d["relation_path"]
            output = model(words=d['context_idxs'],
                                src_lengths=d['context_word_length'],
                                mask=d['context_word_mask'],
                                context_ems_info=d['context_ems_info'],
                                h_t_pairs=d['h_t_pairs'],
                                entity_type=d['context_ner'],
                                entity_id=d['context_pos'],
                                sentence_id=d["context_sent"],
                                mention_id=d["context_mention"],
                                relation_mask=relation_mask,
                                ht_pair_distance=d['ht_pair_distance'],
                                ht_sent_distance=d["ht_sent_distance"],
                                graph_adj=d['graph_adj'],
                                graph_info=d["graph_info"],
                                graph_node_num=d["graph_node_num"],
                                relation_path=d["relation_path"],
                                )
            predictions = output["predictions"]
            loss = torch.sum(BCELogit(predictions, relation_multi_label) * relation_mask.unsqueeze(2)) / (
                    relation_num * torch.sum(relation_mask))
            

            ## Relation
            scores = predictions
            total_mask = relation_mask>0
           
            test_metric.roc_record(loss,scores[...,1:],relation_multi_label[...,1:],total_mask,ign=ign_relation_multi_label[...,1:])


            if output_file:
                scores = predictions.data.cpu().numpy()
                for i in range(len(titles)):
                    j = 0
                    for h_idx in range(L_vertex[i]):
                        for t_idx in range(L_vertex[i]):
                            if h_idx == t_idx:
                                continue
                            for label_idx in range(1,relation_num):
                                if scores[i,j,label_idx]>theta_list[0,label_idx-1]:
                                    test_result.append({"title":titles[i],"h_idx":h_idx,"t_idx":t_idx,"r":id2rel[label_idx]})
                            j+=1

    loss,test_acc,test_recall,test_ign_f1,test_f1,theta = test_metric.cal_roc_metric(global_step,lr_rate,log=True)

    if output_file:
        json.dump(test_result, open(test_prefix + "_index.json", "w"))
        print("result file finish")

    return loss,test_ign_f1, test_f1,theta, test_acc, test_recall


if __name__ == '__main__':
    opt = get_opt()
    print('processId:', os.getpid())
    print('prarent processId:', os.getppid())
    print(json.dumps(opt.__dict__, indent=4))
    rel2id = json.load(open(os.path.join(opt.data_dir, 'rel2id.json'), "r"))
    id2rel = {v: k for k, v in rel2id.items()}
    word2id = json.load(open(os.path.join(opt.data_dir, 'word2id.json'), "r"))
    ner2id = json.load(open(os.path.join(opt.data_dir, 'ner2id.json'), "r"))
    opt.data_word_vec = np.load(os.path.join(opt.data_dir, 'vec.npy'))

    if opt.use_model == 'bert':
        # datasets
        train_set = BERTDGLREDataset(opt.train_set, opt.train_set_save, word2id, ner2id, rel2id, dataset_type='train',opt=opt)
        if opt.test_type == "dev":
            test_set = BERTDGLREDataset(opt.dev_set, opt.dev_set_save, word2id, ner2id, rel2id, dataset_type='test',instance_in_train=train_set.instance_in_train, opt=opt)
        else:
            test_set = BERTDGLREDataset(opt.test_set, opt.test_set_save, word2id, ner2id, rel2id, dataset_type='test',instance_in_train=train_set.instance_in_train, opt=opt)

        test_loader = DGLREDataloader(test_set, opt, batch_size=opt.test_batch_size, dataset_type='test')

        if "DRN" in opt.model_name:
            model = DRN_BERT(opt)
        elif "MPR" in opt.model_name:
            model = MPR_BERT(opt)
        else:
            raise("Error")
    elif opt.use_model == 'bilstm':
        # datasets
        train_set = DGLREDataset(opt.train_set, opt.train_set_save, word2id, ner2id, rel2id, dataset_type='train',opt=opt)
        if opt.test_type == "dev":
            test_set = DGLREDataset(opt.dev_set, opt.dev_set_save, word2id, ner2id, rel2id, dataset_type='test',instance_in_train=train_set.instance_in_train, opt=opt)
        else:
            test_set = DGLREDataset(opt.test_set, opt.test_set_save, word2id, ner2id, rel2id, dataset_type='test',instance_in_train=train_set.instance_in_train, opt=opt)

        test_loader = DGLREDataloader(test_set, opt,batch_size=opt.test_batch_size, dataset_type='test')

        if "DRN" in opt.model_name:
            model = DRN_GloVe(opt)
        elif "MPR" in opt.model_name:
            model = MPR_GloVe(opt)
        else:
            raise("Error")
    else:
        assert 1 == 2, 'please choose a model from [bert, bilstm].'

    import gc

    del train_set
    gc.collect()

    # print(model.parameters)
    print_params(model)

    start_epoch = 1
    pretrain_model = opt.pretrain_model
    lr = opt.lr
    model_name = opt.model_name

    if pretrain_model != '':
        chkpt = torch.load(pretrain_model, map_location=torch.device('cpu'))
        model.load_state_dict(chkpt['checkpoint'])
        logging('load checkpoint from {}'.format(pretrain_model))
    else:
        assert 1 == 2, 'please provide checkpoint to evaluate.'

    model = get_cuda(model)
    model.eval()

    loss,ign_f1, f1, theta, pr_x, pr_y = test(model, test_loader, model_name, id2rel=id2rel,
                            output_file=True, test_prefix='test',config=opt)
    print('finished')
