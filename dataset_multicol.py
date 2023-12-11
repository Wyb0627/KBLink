import os
# CUDA_VISIBLE_DEVICES'] ='1'
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as func
import pandas as pd
import json
import numpy as np
import tqdm
import unicodedata
from argparse import ArgumentParser


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html


class TableDataset(Dataset):
    def __init__(self, dataset: dict, tokenizer, LM='bert', filter_out=True, use_ct=True, ct_length=3, use_mask=False,
                 id_2_label=None, connect_token=' ', max_length=64, dataset_name='iswc'):
        if id_2_label is None:
            id_2_label = {}
        self.tokenizer = tokenizer
        self.device = 'cuda:1'
        self.LM = LM
        self.max_length = max_length
        self.connect_token = connect_token
        if self.LM == 'bert' or self.LM == 'deberta':
            self.pad_tok = '[PAD]'
            self.sep_tok = '[SEP]'
            self.msk_tok = '[MASK]'
            self.pad_num = 0
            # self.bert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        elif self.LM == 'roberta' or self.LM == 'bart':
            self.pad_tok = '<pad>'
            self.sep_tok = '</s>'
            self.msk_tok = '<mask>'
            self.pad_num = 1
        numeric_col = 0
        labels = []
        table_list = []
        table_col = []
        no_candidate_type = []
        count = 0
        no_ct_table = 0
        encodings_msk = []
        encodings = []
        feature_vec_msk = []
        feature_vec = []
        predicted_column_idx = []
        predicted_column_num = []
        cls_idx, cls_idx_msk = [], []
        table_idx_list = []
        # labels_df = pd.read_csv('./data/label.csv', index_col=0)
        table_idx_dict = {}
        no_ct_vec = 0
        total_col = 0
        for table_idx, table in tqdm.tqdm(dataset.items()):
            if len(table['label']) == 0:
                continue
            total_col += len(table['label'])
            table_idx_list.append(int(table_idx))
            table_idx_dict[int(table_idx)] = table['id']
            table_text, table_text_ct = {}, {}
            if not table['linked_cell']:
                no_ct_vec += len(table['label'])
            else:
                no_ct_vec += len(table['label']) - len(table['linked_cell'])
            for col_idx, l in table['col_list_dict'].items():
                if int(col_idx) in table['predicted_column_idx']:
                    temp_list, texts_temp, texts_ct_temp = [], [], []
                    if use_ct:
                        if not table['candidate_type_top_k']:
                            candidate_type = [self.pad_tok] * 3
                        else:
                            # print('table_idx {}'.format(table_idx))
                            count += 1
                            kb_dict_prediction = table['candidate_type_top_k'][col_idx]
                            candidate_type = []
                            if not kb_dict_prediction:
                                is_num = True
                                vector = []
                                for cell in l:
                                    if not is_number(cell):
                                        is_num = False
                                        break
                                    else:
                                        vector.append(float(cell))
                                if is_num:
                                    numeric_col += 1
                                    try:
                                        candidate_type += [str(np.mean(vector)), str(np.var(vector)),
                                                           str(np.max(vector))]
                                    except RuntimeWarning:
                                        print(vector)
                                else:
                                    candidate_type += [self.pad_tok] * 3

                            for ct_dict in kb_dict_prediction:
                                if filter_out and ct_dict['filter']:
                                    candidate_type.append(self.pad_tok)
                                    continue
                                '''
                                # 多个pad填充
                                ct_label_cut = ct_dict['ct_label'].split()[:ct_length] 
                                # while len(ct_label_cut) < ct_length:
                                #   ct_label_cut.append('[PAD]')
                                ct_label_cut.append('[SEP]')
                                candidate_type += ' '.join(ct_label_cut)
                                '''
                                candidate_type.append(ct_dict['ct_label'])
                                # print('******************************')
                                # print(candidate_type)``

                        temp_list.extend(candidate_type)
                        temp_list.append(self.sep_tok)
                        temp_list.extend(l)
                        if use_mask:
                            gt = id_2_label[str(table['label'][col_idx])]
                            texts_ct_temp.append(gt)
                            texts_ct_temp.append(self.sep_tok)
                            texts_ct_temp.extend(temp_list)
                            texts_temp.append(self.msk_tok)
                            texts_temp.append(self.sep_tok)
                            table_text_ct[int(col_idx)] = texts_ct_temp
                        texts_temp.extend(temp_list)
                        table_text[int(col_idx)] = texts_temp

                    else:
                        if use_mask:
                            gt = id_2_label[str(table['label'][col_idx])]
                            texts_ct_temp.append(gt)
                            texts_ct_temp.append(self.sep_tok)
                            texts_ct_temp.extend(l)
                            texts_temp.append(self.msk_tok)
                            texts_temp.append(self.sep_tok)
                            table_text_ct[int(col_idx)] = texts_ct_temp
                        texts_temp.extend(l)
                        table_text[int(col_idx)] = texts_temp
                else:
                    if use_mask:
                        table_text_ct[int(col_idx)] = l
                    table_text[int(col_idx)] = l
            if use_mask:
                texts_ct_dict = self.process_tokenize(table_text_ct, max_length=self.max_length, return_tensor=True,
                                                      connect_token=self.connect_token, end_fix='_msk')
                encodings_msk.append(texts_ct_dict)
                cls_idx_msk.append(self.calculate_token_idx(texts_ct_dict['input_ids_msk']))
            feature_vector_token, feature_vector_token_msk = self.generate_feature_vec(table['linked_cell'])
            texts_dict = self.process_tokenize(table_text, max_length=self.max_length, connect_token=self.connect_token,
                                               return_tensor=True)
            cls_idx.append(self.calculate_token_idx(texts_dict['input_ids']))
            encodings.append(texts_dict)
            feature_vec.append(feature_vector_token)
            feature_vec_msk.append(feature_vector_token_msk)
            label_list = []
            predicted_column_idx_t = table['predicted_column_idx']
            predicted_column_num.append(len(table['predicted_column_idx']))
            if len(table['predicted_column_idx']) < 16:
                predicted_column_idx_t.extend([-1] * (16 - len(table['predicted_column_idx'])))
            for key, values in table['label'].items():
                label_list.append(values)
            label_list_t = label_list
            if len(label_list) < 16:
                label_list_t.extend([-1] * (16 - len(label_list)))
            predicted_column_idx.append(predicted_column_idx_t)
            labels.append(label_list_t)
            table_list.append(table['id'])
        # print('Count: {}'.format(count))
        print('No ct col: {}, Total col: {}'.format(no_ct_vec, total_col))
        print('Numeric col: {}'.format(numeric_col))
        self.predicted_column_num = predicted_column_num
        self.predicted_column_idx = predicted_column_idx
        self.encodings = encodings
        self.feature_vec = feature_vec
        self.feature_vec_msk = feature_vec_msk
        self.use_mask = False
        self.cls_idx = cls_idx
        if use_mask:
            self.use_mask = True
            self.encodings_msk = encodings_msk
            self.cls_idx_msk = cls_idx_msk
        self.labels = labels
        # self.table_col = table_col
        self.table_id = table_list
        self.table_idx = table_idx_list

        with open('./data/no_candidate_type.json', mode='w') as file:
            file.write(json.dumps(no_candidate_type, indent=4))
        with open('./data/id_2_table_{}.json'.format(dataset_name), mode='w') as file:
            file.write(json.dumps(table_idx_dict, indent=4))
        print('****** {} ******'.format(len(labels)))

    def return_table_name(self):
        return self.table_col

    def calculate_token_idx(self, encoding: torch.Tensor):
        if self.LM == 'bert':
            token = 101
        elif self.LM == 'roberta' or self.LM == 'bart':
            token = 0
        elif self.LM == 'deberta':
            token = 1
        idx_list = [index for (index, value) in enumerate(encoding.tolist()) if value == token]
        if len(idx_list) < 16:
            idx_list.extend([self.pad_num] * (16 - len(idx_list)))
        return idx_list

    def process_tokenize(self, list_dict: dict, max_length=64, connect_token=' ', return_tensor=False,
                         end_fix=''):
        if self.LM == 'bert':
            token = 102
        elif self.LM == 'roberta' or self.LM == 'bart' or self.LM == 'deberta':
            token = 2
        if self.LM == 'bert' or self.LM == 'deberta':
            table_dict = {
                'input_ids' + end_fix: [],
                'attention_mask' + end_fix: [],
                'token_type_ids' + end_fix: []
            }
        else:
            table_dict = {
                'input_ids' + end_fix: [],
                'attention_mask' + end_fix: []
            }
        for col_idx, cell_list in list_dict.items():
            encodings = self.tokenizer(connect_token.join(cell_list), padding='max_length', truncation=True,
                                       max_length=max_length)
            for name, encode_list in encodings.items():
                del encode_list[-1]
                table_dict[name + end_fix].extend(encode_list)
        table_dict['attention_mask' + end_fix].append(1)
        table_dict['input_ids' + end_fix].append(token)
        if self.LM == 'bert' or self.LM == 'deberta':
            table_dict['token_type_ids' + end_fix].append(0)
        if return_tensor:
            for name, encode_list in table_dict.items():
                if len(encode_list) > 512:
                    encode_list = encode_list[:512]
                    encode_list[-1] = token
                encode_tensor = torch.tensor(encode_list)
                table_dict[name] = func.pad(encode_tensor,
                                            (0, 512 - encode_tensor.shape[-1]),
                                            mode='constant',
                                            value=self.pad_num)
        return table_dict

    def generate_feature_vec(self, input_dict: dict):
        feature_vec_total = []
        feature_vec_msk_total = []
        for idx, col_list in enumerate(input_dict.values()):
            if col_list:
                encodings = self.tokenizer(col_list, padding='max_length', truncation=True, return_tensors='pt')
                # print(encodings['input_ids'].shape)
                # self.bert_model.eval()
                feature_vec_token = func.pad(encodings['input_ids'],
                                             (0, 0, 0, 25 - encodings['input_ids'].shape[0]),
                                             mode='constant',
                                             value=0)
                feature_vec_token_msk = func.pad(encodings['attention_mask'],
                                                 (0, 0, 0, 25 - encodings['attention_mask'].shape[0]),
                                                 mode='constant',
                                                 value=0)
                # [25, 512]
                feature_vec_total.append(feature_vec_token)
                feature_vec_msk_total.append(feature_vec_token_msk)
                # print(feature_vec_token.shape)
                '''
                feature_vec = self.bert_model(input_ids=encodings['input_ids'].to(self.device),
                                              attention_mask=encodings['attention_mask'].to(self.device),
                                              output_hidden_states=True).last_hidden_state
                feature_vec = torch.mean(feature_vec[:, 0, :], dim=0)
                feature_vec_total[idx, :] = feature_vec.cpu()
                '''
            else:
                noise_vec = torch.zeros((25, 512), dtype=torch.long)
                noise_vec[:, 0] = 101
                noise_vec[:, -1] = 102
                feature_vec_total.append(noise_vec)
                feature_vec_msk_total.append(torch.zeros((25, 512), dtype=torch.long))
        row_num = len(feature_vec_total)
        if row_num == 0:
            feature_vec_total = torch.zeros((16, 25, 512), dtype=torch.long)
            feature_vec_total[:, :, 0] = 101
            feature_vec_total[:, :, -1] = 102
            feature_vec_msk_total = torch.zeros((16, 25, 512), dtype=torch.long)
        else:
            feature_vec_total = func.pad(torch.stack(feature_vec_total),
                                         (0, 0, 0, 0, 0, 16 - row_num),
                                         mode='constant',
                                         value=0)
            feature_vec_msk_total = func.pad(torch.stack(feature_vec_msk_total),
                                             (0, 0, 0, 0, 0, 16 - row_num),
                                             mode='constant',
                                             value=0)
        # print(feature_vec_total.shape)
        return feature_vec_total, feature_vec_msk_total

    def get_label_length(self):
        length_list = []
        for label_list in self.labels:
            for label_idx in label_list:
                if label_idx < 0:
                    break
                length_list.append(label_idx)
        return len(length_list)

    def __getitem__(self, idx):
        item_dict = self.encodings[idx]
        item = {key: val for key, val in item_dict.items()}
        if self.use_mask:
            for key, val in self.encodings_msk[idx].items():
                item[key] = val
            # item['cls_idx_msk'] = self.cls_idx_msk[idx]
        item['cls_idx'] = self.cls_idx[idx]
        item['labels'] = torch.tensor(self.labels[idx])
        item['table_id'] = self.table_id[idx]
        item['predicted_column_idx'] = self.predicted_column_idx[idx]
        item['max_length'] = self.max_length
        item['predicted_column_num'] = self.predicted_column_num[idx]
        item['feature_vec'] = self.feature_vec[idx]
        item['feature_vec_msk'] = self.feature_vec_msk[idx]
        item['table_idx'] = self.table_idx[idx]
        return item

    def __len__(self):
        return len(self.labels)


def generate_data(full_dataset, propotion=1.0, dataset_name='viznet'):
    torch.manual_seed(3407)
    test_size = int(0.2 * len(full_dataset))
    if dataset_name == 'iswc':
        eval_size = int(0.1 * len(full_dataset))
        train_size = len(full_dataset) - test_size - eval_size
        train_dataset, eval_dataset, test_dataset = torch.utils.data.random_split(full_dataset,
                                                                                  [train_size, eval_size, test_size])
    else:
        input_dataset, test_dataset = torch.utils.data.random_split(full_dataset,
                                                                    [len(full_dataset) - test_size, test_size])
        if propotion < 1.0:
            propotion_size = int(propotion * len(input_dataset))
            train_eval, _ = torch.utils.data.random_split(input_dataset,
                                                          [propotion_size, len(input_dataset) - propotion_size])
        else:
            train_eval = input_dataset
        train_size = int(0.8 * len(train_eval))
        train_dataset, eval_dataset = torch.utils.data.random_split(train_eval,
                                                                    [train_size, len(train_eval) - train_size])
    return train_dataset, eval_dataset, test_dataset


if __name__ == '__main__':
    parser = ArgumentParser()
    # parser.add_argument("--train_csv_dir", help="input csv dir for training", default="./data/ft_cell/train_csv")
    parser.add_argument("--dataset_name", default='iswc')
    parser.add_argument("--propotion", type=float, default=1.0)
    parser.add_argument("--filter_size", help="row filter", default='25')
    parser.add_argument("--model", help="row filter", default='KBLink')
    parser.add_argument("--max_length", help="row filter", type=int, default=64)
    parser.add_argument("--LM", help="language model used", type=str, default='bert')
    parser.add_argument("--no_endfix", help="language model used", action="store_true")
    parser.add_argument("--end_fix", help="exp_name", type=str, default='')
    args = parser.parse_args()
    filter_size = args.filter_size
    dataset_name = args.dataset_name
    model = args.model
    endfix = '{}_{}_{}_{}'.format(dataset_name, filter_size, model, args.max_length)
    if not args.no_endfix:
        print('endfix: {}_{} begin'.format(endfix, str(args.propotion)))
    else:
        print('LM: {} begin for {}'.format(args.LM, dataset_name))
    with open('./data_final/processed_dataset_{}_{}_no_filter.json'.format(dataset_name, filter_size), 'r') as file:
        dataset = json.load(file)
    with open('./data/label_{}.json'.format(dataset_name), 'r') as file:
        id_2_label = json.load(file)
    if model == 'KBLink':
        filter_out = True
        use_ct = True
        use_mask = True
    elif model == 'KBLink_no_msk':
        filter_out = True
        use_ct = True
        use_mask = False
    elif model == 'KBLink_no_ct':
        filter_out = False
        use_ct = False
        use_mask = True
    else:
        filter_out = False
        use_ct = False
        use_mask = False
    if args.LM.lower() == 'bert':
        from transformers import BertTokenizer, BertModel

        Tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    elif args.LM.lower() == 'roberta':
        from transformers import RobertaTokenizer

        Tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    elif args.LM.lower() == 'deberta':
        from transformers import DebertaTokenizer

        Tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
    elif args.LM.lower() == 'bart':
        from transformers import BartTokenizer

        Tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

    dataset = TableDataset(dataset, Tokenizer, args.LM.lower(), filter_out=filter_out, use_ct=use_ct, ct_length=3,
                           use_mask=use_mask,
                           id_2_label=id_2_label, connect_token=' ', max_length=args.max_length,
                           dataset_name=dataset_name)
    print('Generate complete')
    print('{} columns in total'.format(len(dataset)))
    # raise KeyError
    train_dataset, eval_dataset, test_dataset = generate_data(dataset, args.propotion, dataset_name)
    print('Saving datasets')
    if not args.no_endfix:
        torch.save(train_dataset, './dataset_final/train_dataset_' + endfix + '_no_filter')
        torch.save(eval_dataset, './dataset_final/eval_dataset_' + endfix + '_no_filter')
        torch.save(test_dataset, './dataset_final/test_dataset_' + endfix + '_no_filter')
        print('endfix: {} finished'.format(endfix + '_filter'))
    else:
        torch.save(train_dataset,
                   './dataset_final/train_dataset_' + str(dataset_name) + '_' + args.LM.lower() + args.end_fix)
        torch.save(eval_dataset,
                   './dataset_final/eval_dataset_' + str(dataset_name) + '_' + args.LM.lower() + args.end_fix)
        torch.save(test_dataset,
                   './dataset_final/test_dataset_' + str(dataset_name) + '_' + args.LM.lower() + args.end_fix)
        print('train_dataset_' + str(dataset_name) + '_' + args.LM.lower() + args.end_fix + ' finished')
