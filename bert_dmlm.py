from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel
from transformers import DebertaTokenizer, DebertaModel
from transformers import BartTokenizer, BartModel
import torch
# import torch_scatter
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from operator import itemgetter


# 先encode行，然后输入行embedding组成的列，和embed好的candidate type 和 mask 和 gt，再encode列
class KBLink(nn.Module):
    def __init__(self, num_labels=77, model_name="bert-base-uncased",
                 learn_weight=False, LM='bert', feature_vec=False, drop_out=0.2):  # "deepset/bert-base-cased-squad2"
        super(KBLink, self).__init__()
        self.model_name = model_name
        self.LM = LM
        if self.LM == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.bert_model_contextual = BertModel.from_pretrained(self.model_name)
            self.bert_model = BertModel.from_pretrained(self.model_name)
        elif self.LM == 'roberta':
            self.model_name = 'roberta-base'
            self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
            self.bert_model_contextual = RobertaModel.from_pretrained(self.model_name)
            self.bert_model = RobertaModel.from_pretrained(self.model_name)
        elif self.LM == 'deberta':
            self.model_name = 'microsoft/deberta-base'
            self.tokenizer = DebertaTokenizer.from_pretrained(self.model_name)
            self.bert_model_contextual = DebertaModel.from_pretrained(self.model_name)
            self.bert_model = DebertaModel.from_pretrained(self.model_name)
        elif self.LM == 'bart':
            self.model_name = 'facebook/bart-base'
            self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
            self.bert_model_contextual = BartModel.from_pretrained(self.model_name)
            self.bert_model = BartModel.from_pretrained(self.model_name)
        self.bert_model.eval()
        self.counter = 0
        self.learn_weight = learn_weight
        self.feature_vec = feature_vec
        self.projector = nn.Linear(768, len(self.tokenizer))
        self.projector_gt = nn.Linear(768, len(self.tokenizer))

        # if self.feature_vec:
        # self.classifier = nn.Linear(768 * 2, num_labels)
        # else:
        # if self.feature_vec:
        # self.feat_translator = nn.Linear(768, 768)
        # self.attn = nn.MultiheadAttention(768, 4, drop_out, batch_first=True)
        self.classifier = nn.Linear(768, num_labels)
        self.loss_func = nn.CrossEntropyLoss(reduction='sum')
        self.dropout = nn.Dropout(drop_out)
        self.activation = nn.Tanh()
        # self.loss_func_doduo = nn.BCELoss()
        self.softmax = nn.Softmax(dim=-1)
        if learn_weight:
            grad = True
        else:
            grad = False
        self.sigma = nn.Parameter(torch.ones(2), requires_grad=grad)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        # self.dropout = nn.Dropout(0.1)

    def dmlm_loss(self, softmax_gt, softmax_contextual):
        # softmax_gt = self.log_softmax(bert_gt_output)
        # softmax_contextual = self.softmax(contextual_prediction_output)
        bert_gt_sum = torch.sum(softmax_gt, dim=-1)
        contextual_prediction_sum = torch.sum(softmax_contextual, dim=-1)
        bert_gt = torch.div(softmax_gt.t(), bert_gt_sum).t()
        contextual_prediction = torch.div(softmax_contextual.t(), contextual_prediction_sum).t()
        loss = torch.sum(-1 * contextual_prediction * bert_gt, dim=-1)
        return torch.mean(loss)

    def data_in_one(self, input_data):
        min_val = np.nanmin(input_data)
        max_val = np.nanmax(input_data)
        return (input_data - min_val) / (max_val - min_val)

    def forward(self, input_data):
        self.counter += 1
        classfi_loss, batch_col_count, batch_count, dmlm_loss = 0, 0, 0, 0
        bert_hidden_states = torch.Tensor().to('cuda')
        bert_hidden_states_gt = torch.Tensor().to('cuda')
        classfi_output = torch.Tensor().to('cuda')
        total_label = []
        total_prediction = torch.Tensor().to('cuda')
        total_gt_tensor = torch.LongTensor().to('cuda')
        # total_prediction = []
        bert_output = self.bert_model_contextual(input_ids=input_data['input_ids'],
                                                 attention_mask=input_data['attention_mask'],
                                                 output_hidden_states=False)
        if self.learn_weight:
            with torch.no_grad():
                bert_gt_output = self.bert_model(input_ids=input_data['input_ids_msk'],
                                                 attention_mask=input_data['attention_mask_msk'],
                                                 output_hidden_states=False)
        for batch_id, cls_idx_list in enumerate(input_data['cls_idx'].tolist()):
            batch_count += 1
            label_list = input_data['labels'].tolist()[batch_id]
            pred_col_num = input_data['predicted_column_num'].tolist()[batch_id]

            # print(label_list)
            # print(input_data['predicted_column_num'].tolist()[batch_id])
            target_cls_idx = []
            target_msk_idx = []
            gt_label = []
            break_from = False
            for idx in range(pred_col_num):
                if label_list[idx] == -1:
                    break_from = True
                    break_idx = idx
                    break
                batch_col_count += 1
                target_col_idx = int(input_data['predicted_column_idx'].tolist()[batch_id][idx])
                target_cls_idx.append(int(cls_idx_list[target_col_idx]))
                target_msk_idx.append(int(cls_idx_list[target_col_idx]) + 1)

                gt_label.append(label_list[idx])
            if break_from:
                pred_col_num = break_idx
            cls_vector = bert_output.last_hidden_state[batch_id, target_cls_idx, :]
            if self.feature_vec:
                row_num = input_data['feature_vec'].shape[2]
                token_length = input_data['feature_vec'].shape[3]
                # vec_tensor = input_data['feature_vec'][batch_id, :pred_col_num, :, :].view(pred_col_num * row_num, -1)
                # vec_tensor_msk = input_data['feature_vec_msk'][batch_id, :pred_col_num, :, :].view(
                #   pred_col_num * row_num, -1)
                vec_tensor = input_data['feature_vec'][batch_id, :pred_col_num, 0, :]
                vec_tensor_msk = input_data['feature_vec_msk'][batch_id, :pred_col_num, 0, :]
                with torch.no_grad():
                    feature_vec = self.bert_model(input_ids=vec_tensor,
                                                  attention_mask=vec_tensor_msk,
                                                  output_hidden_states=False)
                # feature_vec = feature_vec.last_hidden_state[:, 0, :].view(pred_col_num, row_num, -1)
                feature_vec = feature_vec.last_hidden_state[:, 0, :]
                # print(feature_vec.shape)
                # print(cls_vector.shape)
                # feature_vec = torch.sum(feature_vec, dim=1)
                # feature_vec = self.feat_translator(feature_vec)
                # cls_vector = torch.cat((cls_vector, feature_vec), dim=-1)
                cls_vector = (cls_vector + feature_vec) / 2
                # cls_vector = self.attn(cls_vector.unsqueeze(1), feature_vec.unsqueeze(1), feature_vec.unsqueeze(1))[0]
            # print(cls_vector.shape)
            classfi_vec = self.classifier(cls_vector)
            # classfi_vec=self.classifier(feature_vec)
            # classfi_output = torch.cat((classfi_output, classfi_vec), 0)
            prediction = self.activation(self.dropout(classfi_vec))
            gt_tensor = torch.tensor(gt_label, dtype=torch.long).to('cuda')
            total_gt_tensor = torch.cat((total_gt_tensor, gt_tensor), 0)
            # print(prediction.shape)
            # print(gt_tensor.shape)
            classfi_loss += self.loss_func(prediction, gt_tensor)
            # total_prediction = torch.cat((total_prediction, prediction), 0)
            padded_prediction = F.pad(prediction,
                                      (0, 0,
                                       0, 16 - prediction.shape[0]),
                                      mode='constant',
                                      value=-1)
            # total_prediction.append(padded_prediction)
            total_prediction = torch.cat((total_prediction, torch.unsqueeze(padded_prediction, 0)), 0)
            # total_label.extend(gt_label)
            total_label = input_data['labels'].cpu().tolist()
            if self.learn_weight:
                bert_hidden_states = torch.cat(
                    (bert_hidden_states, bert_output.last_hidden_state[batch_id, target_msk_idx, :]), 0)
                bert_hidden_states_gt = torch.cat(
                    (bert_hidden_states_gt, bert_gt_output.last_hidden_state[batch_id, target_msk_idx, :]), 0)
        classfi_loss = classfi_loss / batch_col_count
        if self.learn_weight:
            msk_vector = self.projector(bert_hidden_states)
            msk_vector_gt = self.projector(bert_hidden_states_gt)
            # print(softmax_gt.cpu().shape)
            # Ori dmlm loss
            softmax_gt = self.log_softmax(0.5 * msk_vector_gt)
            softmax_contextual = self.softmax(0.5 * msk_vector)
            #  bert_gt_sum = torch.sum(softmax_gt, dim=-1)
            # contextual_prediction_sum = torch.sum(softmax_contextual, dim=-1)
            # dmlm_part_1 = torch.div(torch.squeeze(softmax_gt.t()), bert_gt_sum).t()
            # dmlm_part_2 = torch.div(torch.squeeze(softmax_contextual.t()),
            # contextual_prediction_sum).t()
            # dmlm_loss = torch.mean(-1 * torch.sum(dmlm_part_1 * dmlm_part_2, dim=-1))

            # Self defined dmlm loss
            '''
            softmax_gt = self.log_softmax(0.5 * msk_vector_gt)
            softmax_contextual = self.dropout(self.log_softmax(0.5 * msk_vector))
            # softmax_gt = 0.5 * msk_vector_gt
            # softmax_contextual = self.dropout(0.5 * msk_vector)

            bert_gt_sum = torch.sum(softmax_gt, dim=-1)
            contextual_prediction_sum = torch.sum(softmax_contextual, dim=-1)
            dimension = softmax_gt.shape[-1]
            
            func = torch.div(torch.squeeze(softmax_gt.t()), bert_gt_sum).t() / self.sigma[
                0].exp()  # - 0.5 * self.sigma[0].expand(dimension)

            dmlm_part_1 = torch.exp(func)
            dmlm_part_2 = torch.div(torch.squeeze(softmax_contextual.t()),
                                    contextual_prediction_sum).t() / self.sigma[
                              0].exp()  # - 0.5 * self.sigma[0].expand(dimension)

            # print(dmlm_part_1.shape)
            # print((dmlm_part_1 * dmlm_part_2).shape)
            dmlm_loss = torch.mean(-1 * torch.sum(dmlm_part_1 * dmlm_part_2, dim=-1))
            '''

            # print(dmlm_loss.shape)

            # log_classfi_output = self.log_softmax(classfi_output)
            # print(log_classfi_output.shape)
            # print(total_gt_tensor.shape)

            # classfi_loss_ll = torch.mean(log_classfi_output[:, total_gt_tensor], dim=-1)
            # print(classfi_loss_ll.shape)
            # classfi_loss = classfi_loss / batch_count
            # classfi_loss_adaptive = -1 * classfi_loss_ll / self.sigma[1].exp()
            # classfi_loss_adaptive = classfi_loss / self.sigma[1].exp()  # + 0.5 * self.sigma[1]
            # total_loss = dmlm_loss + torch.mean(classfi_loss_adaptive, dim=-1)

            dmlm_loss = self.dmlm_loss(softmax_gt, softmax_contextual)
            total_loss = (dmlm_loss / (2 * self.sigma[0].exp())
                          + classfi_loss / (2 * self.sigma[1].exp())
                          + 0.5 * (self.sigma[0] + self.sigma[1]))
            # print(total_loss)
            # print(classfi_loss_adaptive)
            # print(dmlm_loss)
            # total_loss = torch.mul(total_loss, torch.ones_like(total_loss).to('cuda')).sum()
            # total_loss.requires_grad_()
            return {'predictions': total_prediction,
                    'label_ids': torch.tensor(total_label),
                    'total_loss': total_loss}
        else:

            # classfi_loss = classfi_loss / batch_count
            return {'label_ids': torch.tensor(total_label),
                    # 'bert_gt_output': bert_gt_output,
                    'predictions': total_prediction,
                    'total_loss': classfi_loss}
