import os
import time
from argparse import ArgumentParser


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
# parser.add_argument("--train_csv_dir", help="input csv dir for training", default="./data/ft_cell/train_csv")
parser.add_argument("--gpu_count", help="How many GPU to use", type=int, default=1)
parser.add_argument("--label_count", help="How many labels in the dataset", type=int, default=275)
# parser.add_argument("--train_csv_dir", help="input csv dir for training", default="./data/ft_table/train_csv")
# parser.add_argument("--train_label_path", help="train label path", default="./data/ft_cell/train_label.csv")
parser.add_argument("--lr", help="Learning rate", type=float, default=1e-4)
parser.add_argument("--epochs", help="Epochs", type=int, default=50)
parser.add_argument("--learn_weight", help="Whether to learn weight", action="store_true")
parser.add_argument("--batch_size", help="Batch size", type=int, default=16)
# parser.add_argument("--train_label_path", help="train label path", default="./data/ft_table/train_label.csv")
parser.add_argument("--end_fix", help="The first end fix", type=str, default="iswc_msk")
parser.add_argument("--end_fix_2", help="The second end fix", type=str, default="")  # Modify for different tasks
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = find_gpus(nums=args.gpu_count)  # 必须在import torch前面
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from util_original import Util
import transformers
import tqdm
import pandas as pd
import datetime
import json
import numpy as np
import random
import torch
import functools
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score, \
    precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from dataset_multicol import TableDataset
from torch.utils.data import Dataset, DataLoader
# from bert_dmlm import KBLink
# from bert_dmlm_col_only import KBLink
from bert_dmlm import KBLink
from custom_trainer import CustomTrainer, JLTrainer

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# bert_embedder = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=args.label_count).to(
#  'cuda')
bert_embedder = KBLink(num_labels=args.label_count, learn_weight=args.learn_weight).to('cuda')
Tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
total_param = []

'''
for name, param in bert_embedder.named_parameters():
    total_param.append([name, param.requires_grad])
with open('./bert_param.json', mode='w') as file:
    file.write(json.dumps(total_param, indent=4))
'''


# for name, param in bert_embedder.named_parameters():
#    if 'bert' in name:
#       param.requires_grad = False


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


def preprocess_logits(logits, labels):
    print(len(labels))
    logits_0 = torch.unsqueeze(logits[0], 0)
    logits_1 = torch.unsqueeze(logits[1], 0)
    # print(logits_0.shape)
    # print(logits_1.shape)
    return (logits_0, logits_1)


def compute_metrics(pred, test_dir='', test_dataset=None, end_fix=''):
    # torch.save(pred, './pred_class')
    # shape = pred.predictions[0]
    # prediction_vector = torch.tensor(pred.predictions[0]).view(shape[0] * shape[1], shape[2], shape[3])
    prediction_vector = torch.tensor(pred.predictions[0])
    prediction_list = prediction_vector.tolist()
    logit_list = []
    for lines in prediction_list:
        for seq in lines:
            if seq[0] == -1:
                break
            logit_list.append(seq)

    print('logit reshape: {}'.format(len(logit_list)))
    '''
    answer_dict = {'label_ids': pred.label_ids.tolist()}
    answer_list = []
    for item in pred.predictions:
        if type(item) is np.ndarray:
            answer_list.append(item.tolist())
        else:
            answer_list.append(item)
    answer_dict['prediction'] = answer_list
    with open('./pred_class_list.json', mode='w') as file:
        file.write(json.dumps(answer_dict, indent=4))
    '''

    labels = pred.label_ids
    length_list = []
    for label_list in labels:
        for label_idx in label_list:
            if label_idx < 0:
                break
            length_list.append(label_idx)
    labels = length_list

    # labels = pred.label_ids
    # print(length_list)
    print('label_length: {}'.format(len(labels)))
    # print(len(labels))
    print('logit shape: {}'.format(torch.tensor(pred.predictions[0]).shape))
    # print(pred.predictions)
    # print(len(pred.predictions[1]))
    preds = torch.tensor(logit_list).argmax(-1)
    # print(len(preds))
    # preds = pred.predictions.argmax(-1)
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
        with open(test_dir + '/bad_case{}.json'.format(end_fix), mode='w') as file:
            file.write(json.dumps(wrong_dict, indent=4))

    return metric_dict


def save_testset(test_data: TableDataset):
    with open('./data/test_data.json', mode='w') as file:
        file.write(json.dumps(test_data.dataset.table_list, indent=4))


def start_train(train_set, eval_set, end_fix, end_fix_2):
    training_args = TrainingArguments(
        output_dir='./bert_result_' + end_fix + end_fix_2,  # 存储结果文件的目录
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        # per_gpu_train_batch_size=4,
        learning_rate=args.lr,
        # label_names=['label_ids', 'labels'],
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="weighted_f1",  # 最后载入最优模型的评判标准，这里选用precision最高的那个模型参数
        weight_decay=0,
        warmup_steps=100,
        evaluation_strategy="epoch",  # 这里设置每100个batch做一次评估，也可以为“epoch”，也就是每个epoch进行一次
        logging_strategy="epoch",
        save_strategy='epoch',
        eval_accumulation_steps=20,
        save_total_limit=3,
        seed=3407,
        logging_dir='./bert_log'  # 存储logs的目录
    )
    # sigma_param = list(map(id, bert_embedder.sigma))
    # base_param = filter(lambda x: id(x) not in sigma_param, bert_embedder.parameters())
    # optim_tuple = (transformers.AdamW(
    #    params=[{'params': bert_embedder.bert_model_contextual.parameters()},
    #            {'params': bert_embedder.bert_model.parameters()},
    #            {'params': bert_embedder.sigma, "lr": 5 * args.lr}],
    #    lr=args.lr,
    #    weight_decay=args.lr),None),
    trainer = JLTrainer(
        model=bert_embedder,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        # optimizers=optim_tuple,
        # tokenizer=Tokenizer,
        compute_metrics=compute_metrics,
        # preprocess_logits_for_metrics=preprocess_logits,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  # 早停Callback
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
    setup_seed(3407)
    set_cpu_num(32)
    # end_fix = 'iswc_msk'
    # end_fix_2 = '_jl'
    end_fix = args.end_fix
    end_fix_2 = args.end_fix_2
    print('endfix: {} begin'.format(end_fix + end_fix_2))
    # label = pd.read_csv(base_dir + '/label.csv')
    start_time = datetime.datetime.now()
    train_dataset = torch.load('./dataset_final/train_dataset_' + end_fix)
    eval_dataset = torch.load('./dataset_final/eval_dataset_' + end_fix)
    # eval_dataset, _ = torch.utils.data.random_split(eval_dataset_whole,
    #                                               [16, len(eval_dataset_whole) - 16])
    test_dataset = torch.load('./dataset_final/test_dataset_' + end_fix)

    print('Start training!')
    print('length train {}'.format(len(train_dataset)))
    trainer = start_train(train_dataset, eval_dataset, end_fix, end_fix_2)
    # torch.save(bert_embedder.bert_model_contextual.state_dict(),
    #          './bert_result_' + end_fix + end_fix_2 + '/bert_state_dict')

    # torch.save(bert_embedder.bert_model.state_dict(), './bert_result' + end_fix + '/bert_state_dict')
    print('endfix: {} finished'.format(end_fix + end_fix_2))
    pred = trainer.predict(test_dataset)
    for name, value in bert_embedder.state_dict().items():
        if name == 'sigma':
            print('Name: {}, Value: {}'.format(name, value))
            break
    # metrics = compute_metrics(pred, test_dir='./bert_result_' + end_fix + end_fix_2 + '/', test_dataset=test_dataset,
    #                         end_fix=end_fix_2)
    metrics = compute_metrics(pred)
    print('Prediction Finished')
    print(metrics)
    # for name, parameter in bert_embedder.named_parameters():
    #    print('{}'.format(name))

    # if not args.end_fix_2:

    # print('Start testing!')
    # test(test_dataset)
