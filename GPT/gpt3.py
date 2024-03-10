import os
import warnings
from argparse import ArgumentParser

warnings.filterwarnings("ignore")

parser = ArgumentParser()
parser.add_argument("--dataset_name", default='iswc')
parser.add_argument("--lr", default=5e-3, type=float)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--dim", default=1536, type=int)
parser.add_argument("--gpu", default=1, type=int)
parser.add_argument("--drop_out", default=0.2, type=float)
parser.add_argument("--no_ct", action="store_true")
parser.add_argument("--feature_vec", action="store_true")
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score, \
    precision_recall_fscore_support
import time
from tqdm import tqdm
from collator import DataCollator

import torch.nn.functional as func


class Model(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, dropout=args.drop_out):
        super(Model, self).__init__()
        # self.hidden = torch.nn.Linear(n_feature, n_hidden).cuda()
        self.activaton = torch.nn.ReLU().cuda()
        # self.activaton = torch.nn.PReLU().cuda()
        # self.activaton = torch.nn.LeakyReLU().cuda()
        self.dropout = torch.nn.Dropout(dropout)
        self.out = torch.nn.Linear(n_hidden, n_output).cuda()
        self.n_hidden = n_hidden
        self.loss_func = nn.CrossEntropyLoss()
        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=n_hidden, nhead=8, batch_first=True)

    def forward(self, labels, embedding, mask):
        # hidden = self.hidden(embedding)
        # hidden = self.activaton(embedding)

        embedding = self.transformer_encoder(embedding, src_key_padding_mask=mask)
        # embedding = torch.mean(embedding, dim=1)
        embedding = embedding[:, 0, :]
        # hidden = self.dropout(embedding)
        out = self.out(embedding)
        loss = self.loss_func(out, labels)
        return {'label_ids': labels,
                'predictions': out,
                'loss': loss}


class GPTDataset(Dataset):

    def get_ids(self):
        return self.id

    def __init__(self, dataset: list, use_feature_vec=True, KGLink=True):
        self.label = []
        self.embedding = []
        self.mask = []
        self.id = []
        # print(dataset[0].keys())
        for table in dataset:
            for col_idx, column in table.items():
                self.id.append(column['id'])
                if not column['label']:
                    continue
                self.label.append(int(column['label']))
                if KGLink:
                    table_emb = torch.tensor(column['column_emb'])

                    if column['feature_vec_emb'] and use_feature_vec:
                        feature_vec = torch.tensor(column['feature_vec_emb'])
                        input_emb = (feature_vec + table_emb) / 2
                    else:
                        input_emb = table_emb

                else:
                    input_emb = torch.tensor(column['column_emb_no_ct'])
                input_emb_total = [input_emb]
                for col_idx_2, column_2 in table.items():
                    if col_idx_2 != col_idx:
                        if KGLink:
                            table_emb_2 = torch.tensor(column_2['column_emb'])
                            if column_2['feature_vec_emb'] and use_feature_vec:
                                feature_vec_2 = torch.tensor(column_2['feature_vec_emb'])
                                input_emb_2 = (feature_vec_2 + table_emb_2) / 2
                            else:
                                input_emb_2 = table_emb_2
                        else:
                            input_emb_2 = torch.tensor(column_2['column_emb_no_ct'])
                        input_emb_total.append(input_emb_2)
                pad_len = 16 - len(input_emb_total)
                input_emb = torch.stack(input_emb_total, dim=0)
                if pad_len > 0:
                    dim = input_emb.shape[1]
                    pad = torch.zeros(pad_len, dim)
                    msk = torch.zeros(16 - pad_len)
                    input_emb_pad = torch.cat([input_emb, pad], dim=0)
                    msk_pad = torch.cat([msk, torch.ones(pad_len)], dim=0)
                else:
                    input_emb_pad = input_emb[:16, :]
                    msk_pad = torch.zeros_like(input_emb)

                # if input_emb.shape[0] != 768:
                #    print(input_emb.shape)
                self.embedding.append(input_emb_pad)
                self.mask.append(msk_pad)
        self.label = torch.tensor(self.label)
        self.embedding = torch.stack(self.embedding, dim=0)
        self.mask = torch.stack(self.mask, dim=0)

    def __getitem__(self, idx):
        item = {'labels': self.label[idx],
                'embedding': self.embedding[idx],
                'mask': self.mask[idx]}
        return item

    def __len__(self):
        return len(self.label)


def load_dataset(input_dir: str):
    total_data = []
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            with open(os.path.join(input_dir, filename), "r") as file:
                batch = json.load(file)
            total_data += batch
            # print(f"Loaded {filename}")
    return total_data


def compute_metrics(pred):
    # print(labels)
    print(torch.tensor(pred.predictions[0]).shape)
    print(torch.tensor(pred.predictions[1]).shape)
    # labels = torch.tensor(pred.predictions[1])
    labels = pred.label_ids
    preds = torch.tensor(pred.predictions[1]).argmax(-1)
    # preds = torch.tensor(pred.predictions[0])
    # print(torch.tensor(pred.predictions[0]).shape)
    print(labels.shape)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    _, _, f1_macro, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'weighted_f1': f1,
        'macro_f1': f1_macro,
        'precision': precision,
        'recall': recall
    }


def start_train_hf(model, train_set, eval_set):
    # if args.dataset_name == 'iswc':
    epochs = args.epochs
    # else:
    #    epochs = 20
    training_args = TrainingArguments(
        output_dir='../GPT3/gpt_' + args.dataset_name + '_' + str(args.lr) + '_' + str(args.epochs) + '_' + str(
            args.batch_size) + '_' + str(args.dim),  # 存储结果文件的目录
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        # per_gpu_train_batch_size=4,
        learning_rate=args.lr,
        # label_names=['label_ids', 'labels'],
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="weighted_f1",  # 最后载入最优模型的评判标准，这里选用precision最高的那个模型参数
        # weight_decay=0,
        # warmup_steps=100,
        evaluation_strategy="epoch",  # 这里设置每100个batch做一次评估，也可以为“epoch”，也就是每个epoch进行一次
        logging_strategy="epoch",
        save_strategy='epoch',
        eval_accumulation_steps=20,
        save_total_limit=10,
        seed=3407,
        logging_dir='./log'  # 存储logs的目录
    )
    trainer = Trainer(
        model=model,
        # data_collator=DataCollator(),
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],  # 早停Callback
    )
    trainer.train()
    return trainer


def generate_data(full_dataset):
    train_size = int(0.7 * len(full_dataset))
    eval_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - eval_size - train_size
    torch.manual_seed(3407)
    train_dataset, eval_dataset, test_dataset = torch.utils.data.random_split(full_dataset,
                                                                              [train_size, eval_size, test_size])
    return train_dataset, eval_dataset, test_dataset


def generate_num_testset(dataset, test_dataset, whole_dataset: list):
    with open(f'./data_final/num_list_{dataset}.json', 'r') as f:
        num_id = json.load(f)
    test_list = []
    test_dict = {}
    num_dataset = []
    for cols in num_id:
        if cols[0] in test_dataset.get_ids():
            test_list.append(cols)
            if cols[0] not in test_dict:
                test_dict[cols[0]] = [cols[1]]
            else:
                test_dict[cols[0]].append(cols[1])
    for table in whole_dataset:
        table_new = {}
        for col_idx, column in table.items():
            if column['id'] in test_dict:
                if col_idx in test_dict[table['id']]:
                    table_new[col_idx] = column
        num_dataset.append(table_new)
    return num_dataset


def load_num_dataset():
    with open(f'./GPT_emb/viznet_num/emb_data_num.json', 'r') as file:
        dataset_num = json.load(file)
    return dataset_num


if __name__ == '__main__':
    dataset_name = args.dataset_name
    if args.dataset_name == 'viznet':
        data = load_dataset(f"./GPT_emb/{dataset_name}_{args.dim}")
    else:
        data = load_dataset(f"./GPT_emb/{dataset_name}_{args.dim}")
    dim = args.dim
    random.seed(3407)
    dataset = GPTDataset(data, use_feature_vec=args.feature_vec, KGLink=not args.no_ct)
    if dataset_name == 'viznet':
        dataset_num_test = load_num_dataset()
        num_data_test = random.sample(dataset_num_test, int(0.2 * (len(dataset_num_test))))
        dataset_num_test = GPTDataset(num_data_test, use_feature_vec=args.feature_vec, KGLink=not args.no_ct)
        print('Num length: ', len(dataset_num_test))
    print('Dataset length: ', len(dataset))
    train_dataset, eval_dataset, test_dataset = generate_data(dataset)
    # num_data_test = generate_num_testset(dataset_name, test_dataset, data)

    if args.dataset_name == 'iswc':
        model = Model(n_feature=dim, n_hidden=dim, n_output=275)
    else:
        model = Model(n_feature=dim, n_hidden=dim, n_output=77)
    start_time = time.time()
    print('Start train')
    trainer = start_train_hf(model, train_dataset, eval_dataset)
    end_time = time.time()
    print('Train time used:{} mins'.format((end_time - start_time) / 60))
    with torch.no_grad():
        pred = trainer.predict(test_dataset)
        metrics = compute_metrics(pred)
        print(metrics)
        if dataset_name == 'viznet':
            pred_num = trainer.predict(dataset_num_test)
            metrics_num = compute_metrics(pred_num)
            print(metrics_num)
        print('Prediction Finished')
    inference_time = time.time()
    print('Inference time used:{} mins'.format((inference_time - end_time) / 60))
