from datetime import datetime

import numpy as np
import torch
import torch.nn as nn


def get_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor

def mergy_span_token(context_output,info):
    
    word_size =  context_output.shape[-2]
    if len(info.shape) == 3:
        start, end, w_ids = torch.broadcast_tensors(info[...,0].unsqueeze(-1),
                                                info[...,1].unsqueeze(-1),
                                                torch.arange(0, word_size).cuda()[None,None])
        e_mapping = (torch.ge(w_ids, start) & torch.lt(w_ids, end)).float()
        embed = torch.einsum("abcd,ade->abce",e_mapping,context_output)
        spancnt = torch.sum(e_mapping,dim=-1).unsqueeze(-1).clamp(min=1)
        embed = torch.div(embed,spancnt)
    elif len(info.shape) == 2:
        start, end, w_ids = torch.broadcast_tensors(info[...,0].unsqueeze(-1),
                                                info[...,1].unsqueeze(-1),
                                                torch.arange(0, word_size).cuda()[None])
        e_mapping = (torch.ge(w_ids, start) & torch.lt(w_ids, end)).float()
        spancnt = torch.sum(e_mapping,dim=-1).unsqueeze(-1).clamp(min=1)
        e_mapping = torch.div(e_mapping,spancnt).unsqueeze(-2)
        embed = torch.matmul(e_mapping,context_output).squeeze(-2)
    else:
        raise("dim mismatch")

    return embed

def mergy_token(encoder_output,token_id,id_start=0):
    slen = encoder_output.shape[0]
    token_num = torch.max(token_id).item() - id_start
    token_index = get_cuda(
        (torch.arange(token_num) + 1+id_start).unsqueeze(1).expand(-1, slen))  # [mention_num, slen]
    mentions = token_id.unsqueeze(0).expand(token_num, -1)  # [mention_num, slen]
    select_metrix = (token_index == mentions).float()  # [mention_num, slen]
    # average word -> mention
    word_total_numbers = torch.sum(select_metrix, dim=-1).unsqueeze(-1).expand(-1, slen)  # [mention_num, slen]
    select_metrix = torch.where(word_total_numbers > 0, select_metrix / word_total_numbers, select_metrix)
    x = torch.mm(select_metrix, encoder_output)  # [mention_num, 2*encoder_hid_size]
    return x


def logging(s):
    print(datetime.now(), s)


class Accuracy(object):
    def __init__(self):
        self.correct = 0
        self.total = 0

    def add(self, is_correct):
        self.total += 1
        if is_correct:
            self.correct += 1

    def get(self):
        if self.total == 0:
            return 0.0
        else:
            return float(self.correct) / self.total

    def clear(self):
        self.correct = 0
        self.total = 0


def print_params(model):
    print('total parameters:', sum([np.prod(list(p.size())) for p in model.parameters() if p.requires_grad]))


class Metrics():
    def __init__(self,prefix,logout,use_wandb=False):
        self.prefix = prefix
        self.total_acc = 0
        self.total_predict = 0
        self.total_truth = 0
        self.total_loss = 0
        self.batch_cnt = 0
        self.logger = logout
        self.res = []
        self.use_wandb = use_wandb
        self.task_decompose = False
    def record(self,loss,predict,labels,mask,binmode=True):
        acc,predict_truth,total_truth = self.cal_bin_data(predict,labels,mask)
        self.total_predict += predict_truth
        self.total_acc += acc
        self.total_truth += total_truth
        self.total_loss += loss.item()
        self.batch_cnt += 1
    def cal_bin_data(self,predict,label,mask):
        acc = torch.sum((torch.gt(label,0) & torch.eq(predict,label.long())).float()*mask) #torch.gt(output_exist,0)
        predict_truth = torch.sum(torch.gt(predict,0).float()*mask).item()
        total_truth = torch.sum(torch.gt(label,0).float()*mask).item()
        return acc.item(),predict_truth,total_truth
    def reset(self):
        self.total_acc = 0
        self.total_predict = 0
        self.total_truth = 0
        self.total_loss = 0
        self.batch_cnt = 0
        self.res = []
    def cal_metric(self,global_step,lr_rate,log=True):
        if self.batch_cnt == 0:
            return
        acc = 1e-9 if int(self.total_predict)==0 else 1.0*self.total_acc/self.total_predict+1e-9
        recall = 1e-9 if int(self.total_truth)==0 else 1.0*self.total_acc/self.total_truth+1e-9
        f1 = 2*acc*recall/(acc+recall)
        loss = self.total_loss/self.batch_cnt
        if log:
            self.logger("{:15} Loss{:8.3f} Predict:{:8d}Total:{:8d}True:{:8d} Acc:{:8.3f}%Recall:{:8.3f}%F1:{:8.3f}%".format(self.prefix,loss,int(self.total_predict),int(self.total_truth),int(self.total_acc),100*acc,100*recall,100*f1))
        if self.use_wandb:
            import wandb
            wandb.log({ self.prefix+'accuracy': acc, 
                    self.prefix+"recall":recall,
                    self.prefix+'f1': f1,
                    self.prefix+"loss":loss,
                    self.prefix+"lr":lr_rate},step=global_step)
        return loss,acc,recall,f1
    def roc_record(self,loss,scores,label,mask,multi_theta=False,ign=None,task_decompose=False):
        self.total_loss += loss.item()
        if multi_theta:
            relation_num = scores.shape[-1]
            if ign is not None:
                self.res.append(np.concatenate((scores.cpu().numpy().reshape(-1,relation_num,1),
                                        label.cpu().numpy().reshape(-1,relation_num,1),
                                        ign.cpu().numpy().reshape(-1,relation_num,1),
                                        mask.cpu().numpy().reshape(-1,1,1).repeat(relation_num,axis=1)),axis=2))
            else:
                self.res.append(np.concatenate((scores.cpu().numpy().reshape(-1,relation_num,1),
                                        label.cpu().numpy().reshape(-1,relation_num,1),
                                        mask.cpu().numpy().reshape(-1,1,1).repeat(relation_num,axis=1)),axis=2))
        elif task_decompose:
            self.task_decompose=True
            if ign is not None:
                task1 = torch.max(scores[...,0:3,:],dim=-2)[0]
                task2 = torch.max(scores[...,3:6,:],dim=-2)[0]
                task3 = torch.max(scores[...,6:9,:],dim=-2)[0]
                self.res.append(np.hstack((task1[mask].cpu().numpy().reshape(-1,1),
                            task2[mask].cpu().numpy().reshape(-1,1),
                            task3[mask].cpu().numpy().reshape(-1,1),
                            label[mask].cpu().numpy().reshape(-1,1),
                            ign[mask].cpu().numpy().reshape(-1,1))))
            else:
                self.res.append(np.hstack((task1[mask].cpu().numpy().reshape(-1,1),
                            task2[mask].cpu().numpy().reshape(-1,1),
                            task3[mask].cpu().numpy().reshape(-1,1),
                            label[mask].cpu().numpy().reshape(-1,1))))
        else:
            if ign is not None:
                self.res.append(np.hstack((scores[mask].cpu().numpy().reshape(-1,1),
                            label[mask].cpu().numpy().reshape(-1,1),
                            ign[mask].cpu().numpy().reshape(-1,1))))
            else:
                self.res.append(np.hstack((scores[mask].cpu().numpy().reshape(-1,1),
                            label[mask].cpu().numpy().reshape(-1,1))))

        self.batch_cnt += 1
    def cal_roc_metric(self,global_step,lr_rate,log=True):
        if self.res == []:
            self.cal_metric(global_step,log)
            return
        if self.res[0].ndim==3:
            rel_res = np.concatenate(self.res,axis=0)
            theta,acc,recall,ign_f1,f1 = circle_optim(rel_res)
        elif self.res[0].ndim==2 and self.task_decompose:
            rel_res = np.concatenate(self.res,axis=0)
            theta,acc,recall,ign_f1,f1 = task_optim(rel_res)
        else:
            rel_res = np.vstack(self.res)
            theta,acc,recall,ign_f1,f1 = roc_cal(rel_res)
        loss = self.total_loss/self.batch_cnt
        if log:
            if isinstance(theta,list):
                self.logger("Theta1:{:8.3f}Theta2:{:8.3f}Theta3:{:8.3f}".format(theta[0],theta[1],theta[2]))
                theta = theta[0]
            self.logger("{:15} Loss{:8.3f} Theta:{:8.3f}{:12} Ign F1:{:8.3f}% Acc:{:8.3f}%Recall:{:8.3f}%F1:{:8.3f}%".format(self.prefix,loss,theta," ",100*ign_f1,100*acc,100*recall,100*f1))
        if self.use_wandb:
            import wandb
            wandb.log({ self.prefix+'accuracy': acc, 
                    self.prefix+"recall":recall,
                    self.prefix+'f1': f1,
                    self.prefix+"loss":loss,
                    self.prefix+"lr":lr_rate},step=global_step)
        return loss,acc,recall,ign_f1,f1,theta
    
def circle_optim(res):
    mask = res[...,3]
    rel_res = res[...,:3]
    theta,acc,recall,ign_f1,f1 = roc_cal(rel_res[mask>0])
    rel_num = res.shape[-2]
    rel_theta = theta * np.ones((1,rel_num),np.float32)
    for circle_cnt in range(2):
        for rel_cnt in range(rel_num):
            rel_mask = res[:,rel_cnt,3]
            rel_res = res[:,rel_cnt,:2]
            total_pre = np.sum((res[...,0]>rel_theta)*res[...,3],axis=0)
            total_pre[rel_cnt] = 0
            pre_cnt = np.sum(total_pre)
            total_acc = np.sum((res[...,0]>rel_theta)*res[...,3]*res[...,1],axis=0)
            total_acc[rel_cnt] = 0
            acc_cnt = np.sum(total_acc)
            total_recall = np.sum(res[...,1]*res[...,3],axis=0)
            total_recall[rel_cnt] = 0
            recall_cnt = np.sum(total_recall)
            theta,acc,recall,ign_f1,f1 = roc_cal(rel_res[rel_mask>0],his_acc=acc_cnt,his_pre=pre_cnt,his_recall=recall_cnt)
            rel_theta[0,rel_cnt] = theta
    np.save("circle_theta_list.npy",rel_theta)
    return theta,acc,recall,ign_f1,f1
def task_optim(res):
    def cal_total_performance(scores,threshold,label):
        predict = np.sum(scores>threshold,axis=1)>0
        total_predict = np.sum(predict)
        total_recall = np.sum(label)
        total_correct  = np.sum(predict*label)
        prx = total_correct/total_recall
        pry = total_correct/total_predict
        f1 = 2*prx*pry/(prx+pry)
        return f1
    max_score = np.expand_dims(np.max(res[:,:3],axis=1),1)
    theta,acc,recall,ign_f1,f1 = roc_cal(np.hstack((max_score,res[:,3:])))
    theta_list = theta * np.ones((3,1),np.float32)
    theta_table = theta * np.ones((max_score.shape[0],1),np.float32)
    max_ind = np.argmax(res[:,:3],axis=1)
    for n_iter in range(1):
        for task in range(3):
            other_res = max_score[max_ind!=task]
            other_theta = theta_table[max_ind!=task]
            other_label = (other_res>other_theta)
            acc_cnt = np.sum(other_label*res[max_ind!=task,3][:,None])
            pre_cnt = np.sum(other_label)
            recall_cnt = np.sum(res[max_ind!=task,3])
            select_res = np.hstack((max_score[max_ind==task],res[max_ind==task,3:]))
            theta,acc,recall,ign_f1,f1 = roc_cal(select_res,his_acc=acc_cnt,his_pre=pre_cnt,his_recall=recall_cnt)
            theta_list[task]=theta
            theta_table[max_ind==task]=theta
    # f1 = cal_total_performance(max_score,theta_table,res[:,3])

    np.save("task_theta_list.npy",theta_list)
    return theta,acc,recall,ign_f1,f1

def roc_cal(res,his_acc = 0,his_pre = 0,his_recall = 0):

    res = res[np.argsort(-res[:,0])]
    pr_x = []
    pr_y = []
    correct = np.cumsum(res[:,1]) + his_acc

    total_recall = np.sum(res[:,1]) + his_recall
    if total_recall == 0:
        total_recall = 1  # for test

    total_predict = np.arange(1,len(res)+1) + his_pre
    
    pr_y = correct/total_predict
    pr_x = correct/total_recall

    f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
    f1 = f1_arr.max()

    if res.shape[1] == 3:
        ign = np.cumsum(res[:,2])
    else:
        ign = 0

    total_predict = np.maximum(total_predict-ign,1)
    
    pr_y = (correct-ign)/total_predict
    pr_x = (correct)/total_recall

    f1_arr = (2 * pr_x * pr_y / (pr_x + pr_y + 1e-20))
    ign_f1 = f1_arr.max()
    f1_pos = f1_arr.argmax()
    theta = res[f1_pos][0]
    acc = pr_x[f1_pos]
    recall = pr_y[f1_pos]


    return theta,acc,recall,ign_f1,f1

class BiLSTM(nn.Module):
    def __init__(self, input_size, lstm_hidden_size,nlayers,dropout,word_pad=0,bidir=True):
        super().__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.nlayers = nlayers
        self.word_pad = word_pad
        self.bidir = bidir
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=lstm_hidden_size,
                            num_layers=nlayers, batch_first=True,
                            bidirectional=bidir)
        self.in_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, src, src_lengths,hidden_init = None,ceil_init = None):
        '''
        src: [batch_size, slen, input_size]
        src_lengths: [batch_size]
        '''

        self.lstm.flatten_parameters()
        bsz, slen, input_size = src.size()

        src = self.in_dropout(src)

        new_src_lengths, sort_index = torch.sort(src_lengths, dim=-1, descending=True)
        new_src = torch.index_select(src, dim=0, index=sort_index)

        packed_src = nn.utils.rnn.pack_padded_sequence(new_src, new_src_lengths.cpu(), batch_first=True, enforce_sorted=True)
        if hidden_init is not None and ceil_init is not None:
            packed_outputs, (src_h_t, src_c_t) = self.lstm(packed_src,(hidden_init,ceil_init))
        else:
            packed_outputs, (src_h_t, src_c_t) = self.lstm(packed_src)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True,
                                                      padding_value=self.word_pad)

        unsort_index = torch.argsort(sort_index)
        outputs = torch.index_select(outputs, dim=0, index=unsort_index)

        if not self.bidir:
            src_h_t = src_h_t.view(self.nlayers, bsz, self.lstm_hidden_size)
            src_c_t = src_c_t.view(self.nlayers, bsz, self.lstm_hidden_size)
            output_h_t = src_h_t[-1]
            output_c_t = src_c_t[-1]
        else:
            src_h_t = src_h_t.view(self.nlayers, 2, bsz, self.lstm_hidden_size)
            src_c_t = src_c_t.view(self.nlayers, 2, bsz, self.lstm_hidden_size)
            output_h_t = torch.cat((src_h_t[-1, 0], src_h_t[-1, 1]), dim=-1)
            output_c_t = torch.cat((src_c_t[-1, 0], src_c_t[-1, 1]), dim=-1)
        output_h_t = torch.index_select(output_h_t, dim=0, index=unsort_index)
        output_c_t = torch.index_select(output_c_t, dim=0, index=unsort_index)

        outputs = self.out_dropout(outputs)
        output_h_t = self.out_dropout(output_h_t)
        output_c_t = self.out_dropout(output_c_t)

        return outputs, (output_h_t, output_c_t)

if __name__ == "__main__":
    res = np.load("total_scores.npy")
    theta,acc,recall,ign_f1,f1=task_optim(res)
    print("F1:{:.3f}% Ign F1:{:.3f}".format(f1*100,100*ign_f1))

