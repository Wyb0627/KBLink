import os

os.environ["OPENAI_API_KEY"] = '<Your key here>'
import openai
from openai import OpenAI
import random
import tqdm

openai.api_key = os.environ["OPENAI_API_KEY"]
client = openai.OpenAI()
import numpy as np
import json
import unicodedata

dataset_name = 'viznet'


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


def get_embedding(text: str, model="text-embedding-3-small", dim=-1):
    text = text.replace('\n', ' ')
    whole_emb = client.embeddings.create(input=text, model=model).data[0].embedding
    return whole_emb


def normalize_l2(x):
    x = np.array(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm
    else:
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm)


def load_data(dataset: str):
    with open(dataset, 'r') as file:
        data = json.load(file)
    return data


def create_label_string(filename: str):
    with open(filename, 'r') as file:
        data = json.load(file)
    label_list = []
    for key, value in data.items():
        label_list.append(' ' + key + ': ' + value)
    label_string = ', '.join(label_list)
    return label_string


num_list = []


def generate_prompt(table_dict: dict, sep_token='\t', process_num=False):
    candidate_type_top_k = table_dict['candidate_type_top_k']
    table_prompt_dict = {}
    new_dict = {}

    for col_idx_int in table_dict['predicted_column_idx']:
        col_idx = str(col_idx_int)
        table_ct_list = []
        if col_idx in candidate_type_top_k:
            for item in candidate_type_top_k[str(col_idx)]:
                if not item['filter']:
                    table_ct_list.append(item['ct_label'])
        feature_vec = []
        if col_idx in table_dict['linked_cell'] and table_dict['linked_cell'][col_idx]:
            for cols in table_dict['linked_cell'][col_idx]:
                feature_vec.append(cols.split('\t')[:25])
        # table_ct_list.extend(triple_list)
        col_list_dict = table_dict['col_list_dict'][col_idx][:25]
        table_ct_list.extend(col_list_dict)
        is_num = None
        if process_num:
            is_num = True
            number_list = []
            for cells in col_list_dict:
                if not is_number(cells):
                    is_num = False
                    break
                else:
                    number_list.append(float(cells))
            if is_num:
                number_ct = [str(np.mean(number_list)), str(np.var(number_list)), str(np.max(number_list))]
                table_ct_list = number_ct + table_ct_list
                num_list.append([table_dict['id'], col_idx])
        if feature_vec:
            feature_vec_input = feature_vec[0]
        else:
            feature_vec_input = []
        new_dict[col_idx] = {'feature_vec': feature_vec_input, 'table_cells': table_ct_list,
                             'label': table_dict['label'][col_idx], 'id': table_dict['id'],
                             'col_list_dict': table_dict['col_list_dict'][col_idx][:25], 'is_num': is_num}
    return new_dict


def generate_emb(input_list: list):
    new_list = []
    for table in tqdm.tqdm(input_list, desc='Fetch embedding'):
        table_dict_new = {}
        for col_idx, column_dict in table.items():
            column_dict_new = {}
            cell_string = ' '.join(column_dict['table_cells'])
            column_emb = get_embedding(cell_string)
            cell_string_no_ct = ' '.join(column_dict['col_list_dict'])
            column_emb_no_ct = get_embedding(cell_string_no_ct)
            if column_dict['feature_vec']:
                if isinstance(column_dict['feature_vec'], list):
                    feature_vec_str = column_dict['feature_vec'][0]
                else:
                    feature_vec_str = column_dict['feature_vec']
                feature_vec_emb = get_embedding(feature_vec_str)
            else:
                feature_vec_emb = []
            column_dict_new['label'] = column_dict['label']
            column_dict_new['id'] = column_dict['id']
            column_dict_new['feature_vec_emb'] = feature_vec_emb
            column_dict_new['column_emb'] = column_emb
            column_dict_new['column_emb_no_ct'] = column_emb_no_ct
            table_dict_new[col_idx] = column_dict_new
        new_list.append(table_dict_new)
    return new_list


def generate_emb_num(input_list: list):
    new_list = []
    for table in tqdm.tqdm(input_list, desc='Fetch embedding'):
        table_dict_new = {}
        for col_idx, column_dict in table.items():
            if not column_dict['is_num']:
                continue
            column_dict_new = {}
            cell_string = ' '.join(column_dict['table_cells'])
            column_emb = get_embedding(cell_string)
            cell_string_no_ct = ' '.join(column_dict['col_list_dict'])
            column_emb_no_ct = get_embedding(cell_string_no_ct)
            if column_dict['feature_vec']:
                if isinstance(column_dict['feature_vec'], list):
                    feature_vec_str = column_dict['feature_vec'][0]
                else:
                    feature_vec_str = column_dict['feature_vec']
                feature_vec_emb = get_embedding(feature_vec_str)
            else:
                feature_vec_emb = []
            column_dict_new['label'] = column_dict['label']
            column_dict_new['id'] = column_dict['id']
            column_dict_new['feature_vec_emb'] = feature_vec_emb
            column_dict_new['column_emb'] = column_emb
            column_dict_new['column_emb_no_ct'] = column_emb_no_ct
            table_dict_new[col_idx] = column_dict_new
        new_list.append(table_dict_new)
    return new_list


def convert_to_list(data: dict):
    data_list = []
    for key, value in data.items():
        data_list.append(value)
    return data_list


if __name__ == '__main__':
    dataset = load_data(f'./data_final/processed_dataset_{dataset_name}_25_feat.json')
    dataset_list = convert_to_list(dataset)
    random.seed(3407)

    input_list = []
    print('Load complete')
    # random.sample(dataset_list, int(0.1 * len(dataset_list))
    '''
    for value in dataset_list:
        input_list.append(generate_prompt(value, process_num=True))
    with open(f'./data_final/num_list_{dataset_name}.json', 'w') as file:
        json.dump(num_list, file)
    '''
    print('Start fetch embedding')
    for count in tqdm.tqdm(range(0, len(input_list), 300), desc='Blocks'):
        if os.path.exists(f'./GPT_emb/{dataset_name}/emb_data_{count}.json'):
            continue
        batch_data = generate_emb(input_list[count:count + 300])
        with open(f'./GPT_emb/{dataset_name}/emb_data_{count}.json', 'w') as file:
            json.dump(batch_data, file)
    batch_data_num = generate_emb_num(input_list)
    with open(f'./GPT_emb/{dataset_name}_num/emb_data_num.json', 'w') as file:
        json.dump(batch_data_num, file)
    print('finish')
