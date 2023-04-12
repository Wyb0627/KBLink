from transformers import BertTokenizer, BertModel, BertConfig, BertForSequenceClassification
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class KBLink(nn.Module):
    def __init__(self, num_labels=77, model_name="bert-base-uncased",
                 learn_weight=False):  # "deepset/bert-base-cased-squad2"
        super(KBLink, self).__init__()
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.bert_model_contextual = BertModel.from_pretrained(self.model_name)
        self.bert_model = BertModel.from_pretrained(self.model_name)
        self.bert_model.eval()
        self.counter = 0
        self.learn_weight = learn_weight
        self.projector = nn.Linear(768, len(self.tokenizer))
        self.classifier = nn.Linear(768, num_labels)
        self.loss_func = nn.CrossEntropyLoss(reduction='sum')
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        if learn_weight:
            grad = True
        else:
            grad = False
        self.sigma = nn.Parameter(0.5 * torch.ones(2), requires_grad=grad)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def dmlm_loss(self, softmax_gt, softmax_contextual):
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
        bert_output = self.bert_model_contextual(input_ids=input_data['input_ids'],
                                                 attention_mask=input_data['attention_mask'],
                                                 output_hidden_states=True)
        if self.learn_weight:
            with torch.no_grad():
                bert_gt_output = self.bert_model(input_ids=input_data['input_ids_msk'],
                                                 attention_mask=input_data['attention_mask_msk'],
                                                 output_hidden_states=True)
        for batch_id, cls_idx_list in enumerate(input_data['cls_idx'].tolist()):
            batch_count += 1
            label_list = input_data['labels'].tolist()[batch_id]
            target_cls_idx = []
            target_msk_idx = []
            gt_label = []
            for idx in range(input_data['predicted_column_num'].tolist()[batch_id]):
                batch_col_count += 1
                target_col_idx = int(input_data['predicted_column_idx'].tolist()[batch_id][idx])
                target_cls_idx.append(int(cls_idx_list[target_col_idx]))
                target_msk_idx.append(int(cls_idx_list[target_col_idx]) + 1)
                gt_label.append(label_list[idx])
            cls_vector = bert_output.last_hidden_state[batch_id, target_cls_idx, :]
            classfi_vec = self.classifier(cls_vector)
            classfi_output = torch.cat((classfi_output, classfi_vec), 0)
            prediction = self.dropout(self.activation(classfi_vec))
            gt_tensor = torch.tensor(gt_label, dtype=torch.long).to('cuda')
            total_gt_tensor = torch.cat((total_gt_tensor, gt_tensor), 0)
            classfi_loss += self.loss_func(prediction, gt_tensor)
            padded_prediction = F.pad(prediction,
                                      (0, 0,
                                       0, 16 - prediction.shape[0]),
                                      mode='constant',
                                      value=-1)
            total_prediction = torch.cat((total_prediction, torch.unsqueeze(padded_prediction, 0)), 0)
            total_label = input_data['labels'].tolist()
            if self.learn_weight:
                bert_hidden_states = torch.cat(
                    (bert_hidden_states, bert_output.hidden_states[0][batch_id, target_msk_idx, :]), 0)
                bert_hidden_states_gt = torch.cat(
                    (bert_hidden_states_gt, bert_gt_output.hidden_states[0][batch_id, target_msk_idx, :]), 0)
        classfi_loss = classfi_loss / batch_col_count
        if self.learn_weight:
            msk_vector = self.projector(bert_hidden_states)
            msk_vector_gt = self.projector(bert_hidden_states_gt)
            softmax_gt = self.softmax(0.5 * msk_vector_gt)
            softmax_contextual = self.dropout(self.log_softmax(0.5 * msk_vector))
            dmlm_loss = self.dmlm_loss(softmax_gt, softmax_contextual)
            total_loss = dmlm_loss / 2 * self.sigma[0].exp() + classfi_loss / 2 * self.sigma[1].exp() + 0.5 * (
                    self.sigma[0] + self.sigma[1])
            return {'predictions': total_prediction,
                    'label_ids': torch.tensor(total_label),
                    'total_loss': total_loss}
        else:
            return {'label_ids': torch.tensor(total_label),
                    'predictions': total_prediction,
                    'total_loss': classfi_loss}
