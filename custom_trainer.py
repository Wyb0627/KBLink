import torch
from torch import nn
from transformers import Trainer


class CustomTrainer(Trainer):
    def __init__(self, **kwargs):
        super(CustomTrainer, self).__init__(**kwargs)
        self.loss_func = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)

    def dmlm_loss(self, bert_gt_output, contextual_prediction_output):
        softmax_gt = self.softmax(bert_gt_output)
        softmax_contextual = self.softmax(contextual_prediction_output)
        bert_gt_sum = torch.sum(softmax_gt.sqrt(), dim=-1)
        contextual_prediction_sum = torch.sum(softmax_contextual.sqrt(), dim=-1)
        bert_gt = torch.div(softmax_gt.t(), bert_gt_sum).t()
        contextual_prediction = torch.div(softmax_contextual.sqrt().t(), contextual_prediction_sum.sqrt()).t()
        loss = torch.sum(-1 * contextual_prediction * torch.log(bert_gt), dim=-1)
        return torch.mean(loss)

    def compute_loss(self, model, inputs, return_outputs=False, alpha=0.1, beta=0.9):
        outputs = model(inputs)
        # print(outputs['bert_output'].hidden_states[0].shape)
        # print(outputs['bert_output'].hidden_states[1].shape)
        msk_bert_gt = torch.squeeze(outputs['bert_gt_output'].hidden_states[0][:, 1, :])
        msk_contextual = torch.squeeze(outputs['bert_contextual_output'].hidden_states[0][:, 1, :])
        classfi_loss = outputs['bert_contextual_output'].loss
        # classfi_loss = self.loss_func(self.softmax(outputs['bert_contextual_output'].logits), inputs['labels'])
        dmlm_loss = self.dmlm_loss(msk_bert_gt, msk_contextual)
        total_loss = alpha * dmlm_loss + beta * classfi_loss
        # total_loss = classfi_loss
        # print(outputs['bert_contextual_output'].logits.shape)
        return (total_loss, {'predictions': outputs['bert_contextual_output'].logits,
                             'label_ids': inputs['labels']}) if return_outputs else total_loss


class JLTrainer(Trainer):
    def __init__(self, **kwargs):
        super(JLTrainer, self).__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(inputs)
        # print(outputs['label_ids'].shape)
        if return_outputs:
            # print(outputs['predictions'])
            return outputs['total_loss'], {'predictions': outputs['predictions'], 'label_ids': outputs['label_ids']}
        else:
            return outputs['total_loss']
