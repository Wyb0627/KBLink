import os
import pandas as pd
from util_original import Util
import datetime
import torch
import numpy as np
import spacy
import random
import jsonlines
import kb
import json
import tqdm
import operator
from itertools import combinations
import networkx as nx
from argparse import ArgumentParser
import requests
from feature_vec import proecess_neighbor

parser = ArgumentParser()
# parser.add_argument("--train_csv_dir", help="input csv dir for training", default="./data/ft_cell/train_csv")
parser.add_argument("--filter", help="row filter", type=int, default=25)
parser.add_argument("--dataset", help="dataset", type=str, default='iswc')
args = parser.parse_args()


def set_cpu_num(cpu_num):
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)


nlp = spacy.load("en_core_web_sm")


def setup_seed(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


def get_key(dict, value):
    return [k for k, v in dict.items() if v == value]


def calculate_filter(col_pair_0, col_pair_1, filtered_linkage, filtered_linkage_total_by_col, link_dict):
    # 初始化所有dict
    match_id = {}
    if col_pair_0['col_idx'] not in filtered_linkage:  # 如果该行不在dict中，创建该行dict
        filtered_linkage[col_pair_0['col_idx']] = {}
    if col_pair_0['col_idx'] not in filtered_linkage_total_by_col:
        filtered_linkage_total_by_col[col_pair_0['col_idx']] = {}
    if col_pair_1['col_idx'] not in filtered_linkage:
        filtered_linkage[col_pair_1['col_idx']] = {}
    if col_pair_1['col_idx'] not in filtered_linkage_total_by_col:
        filtered_linkage_total_by_col[col_pair_1['col_idx']] = {}
    if col_pair_0['col_idx'] not in link_dict:
        link_dict[col_pair_0['col_idx']] = {'s_set': set(),
                                            't_set': set()}
    if col_pair_1['col_idx'] not in link_dict:
        link_dict[col_pair_1['col_idx']] = {'s_set': set(),
                                            't_set': set()}
    for match0 in col_pair_0['match']:
        match_id[match0['_source']['id']] = match0['_source']
    for match1 in col_pair_1['match']:
        for edge in match1['_source']['edges']:
            if 'Q' not in edge[1] or edge[1] == 'Q4167410':
                # G.add_node(match1['_source']['id'])
                continue
            elif edge[1] in match_id:
                add_to_dict(filtered_linkage[col_pair_0['col_idx']], edge[1])
                add_to_dict(filtered_linkage_total_by_col[col_pair_0['col_idx']], edge[1])
                link_dict[col_pair_0['col_idx']]['s_set'].add(edge[1])
                for edge_entity in match_id[edge[1]]['edges']:
                    if 'Q' in edge_entity[1] and edge_entity[1] != 'Q4167410':
                        link_dict[col_pair_0['col_idx']]['t_set'].add(edge_entity[1])
                        add_to_dict(filtered_linkage[col_pair_0['col_idx']], edge_entity[1])
                        add_to_dict(filtered_linkage_total_by_col[col_pair_0['col_idx']], edge_entity[1])

                add_to_dict(filtered_linkage[col_pair_1['col_idx']], match1['_source']['id'])
                add_to_dict(filtered_linkage_total_by_col[col_pair_1['col_idx']], match1['_source']['id'])
                link_dict[col_pair_1['col_idx']]['s_set'].add(match1['_source']['id'])
                for edge_entity in match1['_source']['edges']:
                    if 'Q' in edge_entity[1] and edge_entity[1] != 'Q4167410':
                        link_dict[col_pair_1['col_idx']]['t_set'].add(edge_entity[1])
                        add_to_dict(filtered_linkage[col_pair_1['col_idx']], edge_entity[1])
                        add_to_dict(filtered_linkage_total_by_col[col_pair_1['col_idx']], edge_entity[1])

                # G.add_edge(match1['_source']['id'], edge[1])


def add_to_dict(target_dict: dict, element, add_num=1):
    if element in target_dict:
        target_dict[element] += add_num
    else:
        target_dict[element] = add_num


def source_col_filter(dataset: dict, top_k_dict: dict, k: int = 5):
    total_dict_list = {}
    for idx, table in tqdm.tqdm(dataset.items(), desc='source_col_filter'):
        table_index_dict = {}
        total_dict = {}
        linkage_dict = {}
        filtered_linkage_total = []
        # total_G = nx.DiGraph()
        total_graph_list_row = []
        col_pr_id_dict = []
        filtered_linkage_total_by_col = {}
        top_k_info = top_k_dict[idx]
        total_index_count = {}
        table['top_k_sequence'] = top_k_info['top_k_sequence'][:k]
        for row_idx in top_k_info['top_k_sequence'][:k]:
            row = table['linked_cell'][str(row_idx)]
            # for row in table['linked_cell'].values():
            col_id_dict_row = {}
            filtered_linkage = {}
            # G = nx.DiGraph()
            for col_pairs in list(combinations(row, 2)):
                if 'match' in col_pairs[0]:
                    for match0 in col_pairs[0]['match']:
                        if match0['_source']['id'] not in table_index_dict:
                            match0['_source']['score'] = match0['_score']
                            table_index_dict[match0['_source']['id']] = match0['_source']
                            # for edges in match0['_source']['edges']:

                if 'match' in col_pairs[1]:
                    for match1 in col_pairs[1]['match']:
                        if match1['_source']['id'] not in table_index_dict:
                            match1['_source']['score'] = match1['_score']
                            table_index_dict[match1['_source']['id']] = match1['_source']

                if col_pairs[0]['col_idx'] in table['predicted_column_idx']:
                    if col_pairs[0]['col_idx'] not in col_id_dict_row:
                        col_id_dict_row[int(col_pairs[0]['col_idx'])] = []
                        for item0 in col_pairs[0]['match']:
                            # G.add_node(item0['_source']['id'])
                            col_id_dict_row[int(col_pairs[0]['col_idx'])].append(item0['_source']['id'])
                    calculate_filter(col_pairs[0], col_pairs[1], filtered_linkage,
                                     filtered_linkage_total_by_col, linkage_dict)
                if col_pairs[1]['col_idx'] in table['predicted_column_idx']:
                    if col_pairs[1]['col_idx'] not in col_id_dict_row:
                        col_id_dict_row[int(col_pairs[1]['col_idx'])] = []
                        for item1 in col_pairs[1]['match']:
                            # G.add_node(item1['_source']['id'])
                            col_id_dict_row[int(col_pairs[1]['col_idx'])].append(item1['_source']['id'])
                    calculate_filter(col_pairs[1], col_pairs[0], filtered_linkage,
                                     filtered_linkage_total_by_col, linkage_dict)
            for col_idx, value in filtered_linkage.items():
                if not value:
                    for col in row:
                        if col['match']:
                            if col['col_idx'] == col_idx:
                                if col['match']:
                                    value[col['match'][0]['_source']['id']] = col['match'][0]['_score']

            for link_dict in filtered_linkage.values():
                for link_id, link_count in link_dict.items():
                    if link_id not in total_index_count:
                        total_index_count[link_id] = 0
                    if isinstance(link_count, int):
                        total_index_count[link_id] += link_count + 1
                    else:
                        total_index_count[link_id] += 1
            filtered_linkage_total.append(filtered_linkage)
        total_dict['total_index_count'] = total_index_count
        total_dict['filtered_linkage_total'] = filtered_linkage_total
        total_dict['filtered_linkage_total_by_col'] = filtered_linkage_total_by_col
        total_dict['table_idx'] = table_index_dict
        total_dict['top_k_index'] = top_k_info
        total_dict['linkage_dict'] = linkage_dict
        total_dict_list[idx] = total_dict
    return total_dict_list


def convert_pr_to_per_col(col_pr_id_list: list):
    pr_per_row = {}
    for key in col_pr_id_list[0].keys():
        pr_per_row[key] = {}
    for rows in col_pr_id_list:
        for col_idx, l in rows.items():
            for tup in l:
                pr_per_row[col_idx][tup[0]] = tup[1]
    return pr_per_row


def calculate_acc(total_dict_list, dataset, label_2_id):
    T, F = 0, 0
    for table_idx, table in enumerate(total_dict_list):
        table_dict = {}
        table_dataset = dataset[str(table_idx)]
        for col_idx, col in table['filtered_linkage_total_by_col'].items():
            if str(col_idx) not in table_dataset['label']:
                continue
            if col_idx not in table_dict:
                table_dict[col_idx] = {}
            for wiki_id in col.keys():
                entity = table['table_idx'][wiki_id]
                for edge in entity['edges']:
                    if edge[1] not in table_dict:
                        table_dict[col_idx][edge[1]] = 1
                    else:
                        table_dict[col_idx][edge[1]] += 1
            gt_label = table_dataset['label'][str(col_idx)]
            candidate_type = sorted(table_dict[col_idx].items(), key=lambda x: x[1], reverse=True)
            found = False
            for tups in candidate_type:
                if tups[0] in label_2_id:
                    if int(gt_label) == int(label_2_id[tups[0]]):
                        # print(gt_label)
                        T += 1
                    else:
                        F += 1
                    found = True
                    break
            if not found:
                F += 1
    print('Acc: {}'.format(T / (T + F)))
    return T / (T + F)


def top_k_filter(dataset: dict):
    answer_dict = {}
    total_rows = 0
    for table_idx, table in tqdm.tqdm(dataset.items(), desc='top_k_filter'):
        row_linking_score_total = []
        cell_linking_score_total = []
        total_rows += len(table['linked_cell'])
        for col_idx, rows in table['linked_cell'].items():
            cell_linking_score_per_row = []
            for cell in rows:
                cell_linking_score_list = []
                if 'match' in cell:
                    for linkage in cell['match']:
                        cell_linking_score_list.append(linkage['_score'])
                    if cell_linking_score_list:
                        cell_linking_score = np.max(cell_linking_score_list)
                    else:
                        cell_linking_score = 0.0
                else:
                    cell_linking_score = 0.0
                cell_linking_score_per_row.append(cell_linking_score)
            row_linking_score = np.sum(cell_linking_score_per_row) / len(rows)
            cell_linking_score_total.append(cell_linking_score_per_row)
            row_linking_score_total.append(row_linking_score)
        answer_dict[table_idx] = {'row_linking_score': row_linking_score_total,
                                  'cell_linking_score': cell_linking_score_total,
                                  'top_k_sequence': sorted(range(len(row_linking_score_total)),
                                                           key=lambda k: row_linking_score_total[k], reverse=True)}
    print('Average row number: {}'.format(total_rows / len(table)))
    return answer_dict


def filter_num_date(total_dict_list: dict, dataset: dict, k=3, connect_KB=False, max_flow=False):
    for table_idx, table in tqdm.tqdm(total_dict_list.items(), desc='calculate_max_flow'):
        flow_per_col, ct_per_col = {}, {}
        overlap_score = table['total_index_count']
        if max_flow:
            for col_idx, col in table['filtered_linkage_total_by_col'].items():
                linkage_dict = table['linkage_dict'][col_idx]
                graph = nx.DiGraph()
                col_ct_dict = {}
                for wiki_id, overlap_count in col.items():
                    if wiki_id not in table['table_idx']:
                        continue
                    entity = table['table_idx'][wiki_id]
                    for edges in entity['edges']:
                        if 'Q' not in edges[1] or edges[1] == 'Q4167410':
                            continue
                        else:
                            if wiki_id not in overlap_score or edges[1] not in overlap_score:
                                continue
                            col_ct_dict[wiki_id] = overlap_score[wiki_id]
                            graph.add_edge(wiki_id, edges[1])
                graph.add_node('S')
                graph.add_node('T')
                for i in linkage_dict['s_set']:
                    if i not in overlap_score:
                        continue
                    graph.add_edge('S', i, capacity=overlap_score[i])
                    update_dict = {}
                    for succ in graph.successors(i):
                        update_dict[(i, succ)] = {'capacity': overlap_score[succ]}
                    nx.set_edge_attributes(graph, update_dict)
                for j in linkage_dict['t_set'] - linkage_dict['s_set']:
                    graph.add_edge(j, 'T')
                flow_value, flow_dict = nx.maximum_flow(graph, 'S', 'T')
                answer_dict = {}
                for candidate_type, f_dict in flow_dict.items():
                    if 'T' in f_dict:
                        if f_dict['T'] > 0:
                            answer_dict[candidate_type] = f_dict['T']
                answer_dict_sorted = sorted(col_ct_dict.items(), key=lambda x: x[1], reverse=True)
                flow_per_col[col_idx] = {'flow_value': flow_value,
                                         'flow_dict': flow_dict,
                                         'candidate_type': answer_dict_sorted,
                                         }
                ct_per_col[col_idx] = answer_dict_sorted[:k]
                edge_label = {}
                for i in flow_dict.keys():
                    for j in flow_dict[i].keys():
                        edge_label[(i, j)] = '({}, {})'.format(i, j)
        if not connect_KB:
            dataset[table_idx]['candidate_type_top_k'] = ct_per_col
        else:
            ct_connected_per_col = {}
            for col_num, ct in ct_per_col.items():
                if col_num not in ct_connected_per_col:
                    ct_connected_per_col[col_num] = []
                for tup in ct:
                    filter_out = False
                    entity_label = kb.search(tup[0])
                    # print('Start nlp')
                    if entity_label:
                        doc = nlp(entity_label)
                        for ent in doc.ents:
                            if ent.label_ in ['PERSON', 'DATE']:
                                filter_out = True
                                break
                        ct_connected_per_col[col_num].append({'ct_label': entity_label,
                                                              'over_lap_score': tup[1],
                                                              'filter': filter_out})
                    # print('end nlp')
            dataset[table_idx]['candidate_type_top_k'] = ct_connected_per_col

        table['flow_info'] = flow_per_col
    print('Finish calculate flow')
    return total_dict_list


def apply_filter(dataset: dict, filter_size: int):
    for table_idx, table in tqdm.tqdm(dataset.items(), desc='apply filter'):
        if len(table['linked_cell']) > filter_size:
            linked_cell_new, col_list_dict_new, row_list_dict_new = {}, {}, {}
            for row_idx in table['top_k_sequence']:
                linked_cell_new[str(row_idx)] = table['linked_cell'][str(row_idx)]
                row_list_dict_new[str(row_idx)] = table['cell'][int(row_idx)]
                for cols in table['col_list_dict'].keys():
                    if cols not in col_list_dict_new:
                        col_list_dict_new[cols] = []
                    try:
                        col_list_dict_new[cols].append(table['cell'][int(row_idx)][int(cols)])
                    except IndexError:
                        print(table['row_list_dict'][str(row_idx)])
                        print('Col {}'.format(cols))
                        return False
            table['linked_cell'] = linked_cell_new
            table['col_list_dict'] = col_list_dict_new
            table['row_list_dict'] = row_list_dict_new
    return dataset


def http_lookup(key: str):
    url = "https://www.wikidata.org/w/api.php"
    if isinstance(key, list):
        if key:
            key = key[0]
        else:
            return ''
    params = {
        'action': 'wbgetentities',
        'format': 'json',
        'ids': key,
        'language': 'en',
    }

    # 访问
    r = requests.get(url=url, params=params).json()
    try:
        if not r['success']:
            return ''
        elif 'entities' in r:
            # print(key)
            return r['entities'][key]['labels']['en']['value']
            # return r['search'][0]['label']
        else:
            # print(list(r['search'][0].keys()))
            return ''
    except KeyError:
        return ''


def look_up_neighbor(entity: dict):
    neighbor_list = []
    if entity['types']:
        if isinstance(entity['types'], list):
            entity['types'] = entity['types'][0]
        label_edge = kb.search(entity['types'])
        if label_edge:
            neighbor_list.append(entity['label'] + ' types ' + label_edge)
    for edge in entity['edges']:
        if isinstance(edge[1], list):
            entity_id = edge[1][0]
        else:
            entity_id = edge[1]
        if isinstance(edge[0], list):
            edge_id = edge[0][0]
        else:
            edge_id = edge[0]
        if entity_id == entity['types'] or 'Q' not in entity_id or entity_id == 'Q4167410':
            continue
        else:
            label_edge = kb.search(edge_id)
            label_entity = kb.search(entity_id)
            if not label_edge or not label_entity:
                continue
            neighbor_list.append(' ' + label_edge + ' ' + label_entity)
    if neighbor_list:
        return ' '.join(neighbor_list)
    else:
        return ''


if __name__ == '__main__':
    setup_seed(0)
    set_cpu_num(32)
    base_dir = './data'
    dataset_name = args.dataset
    dataset = Util.load_dict(base_dir + '/10_match_with_score_{}.jsonl'.format(dataset_name))
    print('{} columns in total'.format(len(dataset)))
    filter_size = args.filter
    start_time = datetime.datetime.now()
    answer_dict = top_k_filter(dataset)
    total_dict_list = source_col_filter(dataset, answer_dict, filter_size)
    total_dict_list = filter_num_date(total_dict_list, dataset, connect_KB=True, max_flow=True)
    if filter_size != None:
        dataset = apply_filter(dataset, filter_size)
    dataset = proecess_neighbor(dataset, mode='all', filter_size=filter_size)
    end_time = datetime.datetime.now()
    ten_dict = {}
    if filter_size == None:
        filter_size = 'all'
    for i in range(3):
        ten_dict[i] = dataset[i]
    if filter_size >= 10000:
        filter_size = 'all'
    with open('./data_final/processed_dataset_3_{}_{}.json'.format(dataset_name, filter_size), mode='w') as file:
        file.write(json.dumps(ten_dict, indent=4))
    with open('./data_final/processed_dataset_{}_{}.json'.format(dataset_name, filter_size),
              mode='w') as file:
        file.write(json.dumps(dataset, indent=4))
    print('Time cost: {}'.format(end_time - start_time))
    print('Finish')
