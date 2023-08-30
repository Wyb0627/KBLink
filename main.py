import os
import time
from argparse import ArgumentParser
import warnings

warnings.filterwarnings('ignore')


def find_gpus(nums=4):
    usingGPUs = []
    while len(usingGPUs) == 0:
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >~/.tmp_free_gpus')
        # If there is no ~ in the path, return the path unchanged
        with open(os.path.expanduser('~/.tmp_free_gpus'), 'r') as lines_txt:
            frees = lines_txt.readlines()
            idx_freeMemory_pair = [(idx, int(x.split()[2]))
                                   for idx, x in enumerate(frees)]
        idx_freeMemory_pair.sort(key=lambda my_tuple: my_tuple[1], reverse=True)
        usingGPUs = [str(idx_memory_pair[0])
                     for idx_memory_pair in idx_freeMemory_pair[:nums]]
        if len(usingGPUs) == 0:
            time.sleep(1)
    usingGPUs = ','.join(usingGPUs)
    print('using GPU idx: #', usingGPUs)
    return usingGPUs


parser = ArgumentParser()
parser.add_argument("--gpu_count", help="How many GPU to use", type=int, default=1)
parser.add_argument("--label_count", help="How many labels in the dataset", type=int, default=275)
parser.add_argument("--lr", help="Learning rate", type=float, default=3e-5)
parser.add_argument("--epochs", help="Epochs", type=int, default=50)
parser.add_argument("--learn_weight", help="Whether to learn weight", action="store_true")
parser.add_argument("--batch_size", help="Batch size", type=int, default=16)
parser.add_argument("--exp_name", help="experiment name", type=str, default="iswc_msk")
parser.add_argument("--LM", type=str, default="bert")
parser.add_argument("--seed", type=int, default=3407)
parser.add_argument("--manual_GPU", action="store_true")
parser.add_argument("--gpu_num", help="If manually assign GPU", type=int, default=1)
args = parser.parse_args()

if not args.manual_GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = find_gpus(nums=args.gpu_count)  # 必须在import torch前面
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from util_original import Util
import transformers
import tqdm
import pandas as pd
import datetime
import json
import numpy as np
import random
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from dataset_multicol import TableDataset
from torch.utils.data import Dataset, DataLoader
from bert_dmlm import KBLink
from custom_trainer import JLTrainer

if 'iswc' in args.exp_name:
    label_count = 275
    epochs = 50
else:
    label_count = 77
    epochs = 20
bert_embedder = KBLink(num_labels=label_count, learn_weight=args.learn_weight, LM=args.LM).to('cuda')
Tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
total_param = []


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def generate_data(full_dataset):
    train_size = int(0.7 * len(full_dataset))
    eval_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - eval_size - train_size
    torch.manual_seed(0)
    train_dataset, eval_dataset, test_dataset = torch.utils.data.random_split(full_dataset,
                                                                              [train_size, eval_size, test_size])
    return train_dataset, eval_dataset, test_dataset


def preprocess_logits(logits):
    logits_0 = torch.unsqueeze(logits[0], 0)
    logits_1 = torch.unsqueeze(logits[1], 0)
    return (logits_0, logits_1)


def compute_metrics(pred, test_dir='', test_dataset=None, exp_name=''):
    prediction_vector = torch.tensor(pred.predictions[0])
    prediction_list = prediction_vector.tolist()
    logit_list = []
    for lines in prediction_list:
        for seq in lines:
            if seq[0] == -1:
                continue
            logit_list.append(seq)
    labels = pred.label_ids
    length_list = []
    for label_list in labels:
        for label_idx in label_list:
            if label_idx < 0:
                break
            length_list.append(label_idx)
    labels = length_list
    preds = torch.tensor(logit_list).argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    _, _, f1_macro, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    metric_dict = {
        'accuracy': acc,
        'weighted_f1': f1,
        'macro_f1': f1_macro,
        'precision': precision,
        'recall': recall
    }
    if test_dir:
        wrong_dict = []
        dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        for idx, input_data in enumerate(dataloader):
            label_list = labels.tolist()
            preds_list = preds.tolist()
            if label_list[idx] != preds_list[idx]:
                wrong_dict.append({
                    'table_col': input_data['table_col'],
                    'label': label_list[idx],
                    'pred': preds_list[idx]
                })
        with open(test_dir + '/bad_case_{}.json'.format(exp_name), mode='w') as file:
            file.write(json.dumps(wrong_dict, indent=4))

    return metric_dict


def save_testset(test_data: TableDataset):
    with open('./data/test_data.json', mode='w') as file:
        file.write(json.dumps(test_data.dataset.table_list, indent=4))


def start_train(train_set, eval_set, exp_name):
    training_args = TrainingArguments(
        output_dir='./' + exp_name,  # 存储结果文件的目录
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="weighted_f1",
        weight_decay=0,
        warmup_steps=100,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy='epoch',
        eval_accumulation_steps=20,
        save_total_limit=3,
        seed=args.seed,
        logging_dir='./bert_log'
    )
    trainer = JLTrainer(
        model=bert_embedder,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    trainer.train()
    return trainer


def set_cpu_num(cpu_num):
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)


def load_data(data_list: list):
    dataset = Util.load_dict('./data/find_similar_10_match_only.jsonl', 'index')
    test_dict_total = {}
    for key, value in dataset.items():
        if value['id'] in data_list:
            test_dict_total[key] = value
    return test_dict_total


if __name__ == '__main__':
    setup_seed(args.seed)
    set_cpu_num(32)
    exp_name = args.exp_name
    print('endfix: {} begin'.format(exp_name))
    start_time = datetime.datetime.now()
    train_dataset = torch.load('./dataset/train_dataset_' + exp_name)
    eval_dataset = torch.load('./dataset/eval_dataset_' + exp_name)
    test_dataset = torch.load('./dataset/test_dataset_' + exp_name)
    print('Start training!')
    print('length train {}'.format(len(train_dataset)))
    trainer = start_train(train_dataset, eval_dataset, exp_name)
    print('endfix: {} finished'.format(exp_name))
    pred = trainer.predict(test_dataset)
    for name, value in bert_embedder.state_dict().items():
        if name == 'sigma':
            print('Name: {}, Value: {}'.format(name, value))
            break
    metrics = compute_metrics(pred)
    print('Prediction Finished')
    print(metrics)
