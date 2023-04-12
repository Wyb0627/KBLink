from elasticsearch import Elasticsearch
import tqdm
import jsonlines
import re
import json
import itertools
import string
import Levenshtein
import unicodedata

es = Elasticsearch(['localhost'], port=9201)
doc_type_name = 'wiki_entities'
index_name = 'wikidata_entity_linking'
month = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sept', 'oct', 'nov', 'dec',
         'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november',
         'december']


def remove_brackets(s):
    new = ""
    for l in s:
        if l == "(":
            return new.strip()
        else:
            new += l
    return new.strip()


def levenshtein_similarity(s1, s2):
    s1 = s1.lower().strip().replace("–", "")
    s2 = s2.lower().strip().replace("–", "")
    if s1 and s2 and len(s1) > 1:
        new_s2 = ""
        if s1[1] == ".":
            split_s2 = s2.split(" ")
            new_s2 += split_s2[0][0] + ". " + " ".join(split_s2[1:])
            s2 = new_s2
    s1 = remove_brackets(s1).translate(str.maketrans('', '', string.punctuation))
    s2 = remove_brackets(s2).translate(str.maketrans('', '', string.punctuation))
    if s1 and s2:
        return 1 - (Levenshtein.distance(s1, s2) / max(len(s1), len(s2)))
    else:
        return 0.0


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


def is_number_or_date(s):
    if len(s) < 3:
        return True
    for word in re.split('[-?!,=+`~*#$%^&@|/\\.:;\n\t]', s):
        if len(word) < 3:
            continue
        elif not is_number(word):
            if word.lower() not in month:
                return False
    return True


def clean_link(dataset: dict):
    for item in tqdm.tqdm(dataset.values(), desc='clean_link_filter'):
        linked_cell = {}
        col_list_dict = {}
        row_list_dict = {}
        for row_ind, lines in enumerate(item['cell']):
            linked_line = []
            row_list_dict[row_ind] = []
            for ind, cell in enumerate(lines):
                if ind in item['predicted_column_idx']:
                    row_list_dict[row_ind].append(cell)
                    if cell.isdigit() or is_number_or_date(cell):
                        linked_line.append({'value': '', 'col_idx': ind})
                    else:
                        linked_line.append({'value': cell, 'col_idx': ind})
                    if ind in col_list_dict:
                        col_list_dict[ind].append(cell)
                    else:
                        col_list_dict[ind] = [cell]
                elif not is_number_or_date(cell):
                    row_list_dict[row_ind].append(cell)
                    linked_line.append({'value': cell, 'col_idx': ind})
                    if ind in col_list_dict:
                        col_list_dict[ind].append(cell)
                    else:
                        col_list_dict[ind] = [cell]
            linked_cell[row_ind] = linked_line
        # print(linked_cell)
        item['linked_cell'] = linked_cell
        item['col_list_dict'] = col_list_dict
        item['row_list_dict'] = row_list_dict
    return dataset


def link(dataset: dict, link_key: str, link_only: bool = False, size1: int = 30, size2: int = 10, hop: int = 1):
    total_link_dict = {}
    total_cells = 0
    one_hop_cells = 0
    two_hop_cells = 0
    single_col_cells = 0
    two_hop_count = 0
    for item in tqdm.tqdm(dataset.values(), desc='link'):
        id_dict = {}
        if len(item['label']) == 0:
            continue
        for lines in item[link_key].values():
            for cell_dict in lines:
                link_str = list(cell_dict.values())
                if link_str[0] in total_link_dict:
                    result_list = total_link_dict[link_str[0]]
                else:
                    if not link_str[0]:
                        result_list = []
                    else:
                        dsl = return_dsl('label', link_str[0])
                        result = es.search(index=index_name, doc_type=doc_type_name, body=dsl, size=size2)['hits'][
                            'hits']
                        # result_filtered = get_max_sim(link_str[0], result, size2)
                        result_list = []
                        result_set = set()
                        for res in result:
                            result_list.append(res)
                            result_set.add(res['_source']['id'])
                            if res['_source']['id'] not in id_dict:
                                id_dict[res['_source']['id']] = res['_source']
                                if hop == 2:
                                    for edges in res['_source']['edges']:
                                        if edges[1] != 'Q5' and edges[1] not in result_set:
                                            if edges[1] not in id_dict:
                                                dsl_hop2 = return_dsl('id', edges[1])
                                                result_hop2 = \
                                                    es.search(index=index_name, doc_type=doc_type_name, body=dsl_hop2,
                                                              size=1)['hits']['hits']
                                                if len(result_hop2) > 0:
                                                    id_dict[edges[1]] = result_hop2[0]['_source']
                                                    result_list.append(result_hop2[0]['_source'])
                                                    result_set.add(result_hop2[0]['_source']['id'])
                                                    two_hop_count += 1
                                                else:
                                                    id_dict[edges[1]] = None
                                            else:
                                                if id_dict[edges[1]] is not None:
                                                    result_list.append(id_dict[edges[1]])
                                                    result_set.add(id_dict[edges[1]]['id'])
                                                    two_hop_count += 1

                    total_link_dict[link_str[0]] = result_list
                cell_dict['match'] = result_list
                cell_dict['edges'] = {}
        if not link_only:
            for rows in tqdm.tqdm(item[link_key], desc='rows'):
                if len(rows) < 2:
                    single_col_cells += find_common_type(item[link_key])
                    break
                else:
                    for tuples in itertools.combinations(rows, 2):
                        l = find_hops(tuples[0], tuples[1])
                        one_hop_cells += l[0]
                        two_hop_cells += l[1]
            total_cells += len(rows) * len(item[link_key])
    print('Two hop count: {}'.format(two_hop_count))
    return dataset, [total_cells, one_hop_cells, two_hop_cells, single_col_cells]


def process_count_list(count_list: list, type_set: set):
    type_dict_total = {}
    type_dict_total_sorted = {}
    for types in type_set:
        type_dict_total[types] = {'id': [], 'count': 0}
    for line_dict in count_list:
        # print(line_dict.keys())
        for key, value in line_dict.items():
            # print('******* key is {}********'.format(key))
            # print(type_dict_total[key])
            type_dict_total[key]['id'] += value['id']
            type_dict_total[key]['count'] += value['count'] / len(count_list)
    if type_dict_total:
        type_dict_sorted = sorted(type_dict_total.items(), key=lambda x: x[1]['count'], reverse=True)
        for tup in type_dict_sorted:
            type_dict_total_sorted[tup[0]] = tup[1]
    return type_dict_total_sorted


def find_common_type(linked_cell: list, threshold: float = 0.5):
    cell_type_dict_total = []
    signle_col_connected = 0
    type_set = set()
    for rows in linked_cell:
        cell_type_dict_row = {}
        for matches in rows[0]['match']:
            for types in matches['types']:
                if types != 'Q4167410':
                    if types in cell_type_dict_row:
                        cell_type_dict_row[types]['id'].append(matches['id'])
                        cell_type_dict_row[types]['count'] += 1
                    else:
                        type_set.add(types)
                        cell_type_dict_row[types] = {'id': [matches['id']], 'count': 1}
        cell_type_dict_total.append(cell_type_dict_row)
    total_list = []
    for ind, item in enumerate(cell_type_dict_total):
        other_list = cell_type_dict_total[:ind] + cell_type_dict_total[ind + 1:]
        type_dict_total = process_count_list(other_list, type_set)
        candidate_types = []
        for key, value in type_dict_total.items():
            if value['count'] < threshold:
                break
            else:
                candidate_types.append([key, value['count']])
                if key in item:
                    signle_col_connected += 1
                    for candidate in item[key]['id']:
                        linked_cell[ind][0]['edges'][candidate] = [candidate, value['count'], key]
                        total_list.append([candidate, value['count'], key])
    with open('./test/single_column_result.json', mode='w') as file:
        file.write(json.dumps(total_list, indent=4))
    return signle_col_connected


def inter(a, b):
    return list(set(a).intersection(set(b)))


def link_KB(dataset: dict):
    count = 0
    for item in tqdm.tqdm(dataset.values(), desc='link'):
        item['related_info'] = {}
        for keys, cols in item['related_column_dot'].items():
            count += 1
            max_score = {}
            for index, lines in enumerate(item['cell']):
                function_list = []
                for c in cols:
                    function_list.append({
                        "filter": {"match": {"desc": lines[int(c)]}},
                        "weight": 4
                    })
                #   query = query + " " + lines[int(c)]
                dsl_modified = {
                    "query": {
                        "function_score": {
                            "query": {"match": {'label': lines[int(keys)]}},
                            "boost": 5,
                            "functions": function_list,
                            "score_mode": "multiply",
                            "boost_mode": "multiply",
                        }
                    }
                }
                if lines[int(keys)] == 'W.Garne':
                    print(dsl_modified)
                dsl = {
                    'query': {
                        'match': {'label': lines[int(keys)]}
                    }
                }
                candidate_entities = es.search(index=index_name, body=dsl_modified, size=1)
                max_score[index] = candidate_entities['hits']['max_score']
                if not max_score[index]:
                    max_score[index] = 0.0
                item['related_info'][lines[int(keys)]] = candidate_entities
            item['max_score_' + str(keys)] = sorted(max_score.items(), key=lambda kv: (str(kv[1]), str(kv[0])),
                                                    reverse=True)
    with jsonlines.open('./test/train_processed_KBlinked_total_linked.jsonl', mode='w') as file:
        for _, value in dataset.items():
            file.write(value)
    print('Total {} columns'.format(count))
    return dataset


def find_path(dataset: dict):
    for item in tqdm.tqdm(dataset.values(), desc='find path'):
        for keys, cols in item['related_column_dot'].items():
            max_score = item['max_score_' + keys]
            linked_entity_0 = item['related_info'][item['cell'][max_score[0][0]][int(keys)]]
            linked_entity_1 = item['related_info'][item['cell'][max_score[1][0]][int(keys)]]
            id_0 = linked_entity_0['hits']['hits'][0]['_source']['id']
            id_1 = linked_entity_1['hits']['hits'][0]['_source']['id']
            edge_list_0 = linked_entity_0['hits']['hits'][0]['_source']['edges']
            edge_list_1 = linked_entity_1['hits']['hits'][0]['_source']['edges']
            node_info = {
                # 'entity_0': linked_entity_0,
                # 'entity_1': linked_entity_1,
                'id_0': id_0,
                'id_1': id_1,
                'edge_list_0': edge_list_0,
                'edge_list_1': edge_list_1,
                'hop': 0
            }
            hop = connect_node(node_info)
            item['hops_' + keys] = hop
    return dataset


def find_in(edge_list, total_list):
    for item in total_list:
        if edge_list == item:
            return True
    return False


def sort_dict(input: dict):
    temp = sorted(input.items(), key=lambda x: x[1], reverse=True)
    temp_dict = {}
    try:
        max_score = temp[0][1]
    except IndexError:
        max_score = -1
    else:
        for temp_line in temp:
            temp_dict[temp_line[0]] = temp_line[1]
    return temp_dict, max_score


def find_overlap(dataset: dict, related_column='related_column_dot'):
    for item in tqdm.tqdm(dataset.values(), desc='find overlap'):
        max_score = {}
        for keys, cols in item['related_column_dot'].items():
            entity_overlap = {}
            predicate_overlap = {}
            total_edge = []
            total_predicate = []
            for index, lines in enumerate(item['cell']):
                # print(item['cell'][index][int(keys)])
                linked_entity = item['related_info'][item['cell'][index][int(keys)]]['hits']['hits']
                existed_predicate = []
                if not linked_entity:
                    continue
                edge_list = linked_entity[0]['_source']['edges']
                for edge in edge_list:
                    # if not find_in(edge, total_edge):
                    #   total_edge.append(edge)
                    # if edge[0] not in total_predicate:
                    #   total_predicate.append(edge[0])
                    # else:
                    if edge[1] not in entity_overlap:
                        entity_overlap[edge[1]] = 1 / item['num_rows']
                    else:
                        entity_overlap[edge[1]] += 1 / item['num_rows']
                    # if edge[0] not in total_predicate:
                    if edge[0] not in existed_predicate:
                        existed_predicate.append(edge[0])
                        if edge[0] not in predicate_overlap:
                            predicate_overlap[edge[0]] = 1 / item['num_rows']
                        else:
                            predicate_overlap[edge[0]] += 1 / item['num_rows']

            temp_dict_entity, s = sort_dict(entity_overlap)
            max_score[keys] = s
            temp_dict_predicate, _ = sort_dict(predicate_overlap)
            item['entity_overlap_' + str(keys)] = temp_dict_entity
            item['predicate_overlap_' + str(keys)] = temp_dict_predicate
        '''
        best_col = sorted(max_score.items(), key=lambda x: x[1], reverse=True)
        try:
            if best_col:
                item['best_col'] = best_col[0][0]
            else:
                item['best_col'] = -1
        except IndexError:
            print('Error here')
            print(best_col)
        '''
        item['best_overlap_score'] = max_score
    with jsonlines.open('./test/train_processed_KBlinked_total_linked.jsonl', mode='w') as file:
        for _, value in dataset.items():
            file.write(value)
    return dataset


def build_simple_dsl(field, index):
    dsl = {
        'query': {
            'match': {field: index}
        }
    }
    return dsl


def link_predicate(dataset):
    for item in tqdm.tqdm(dataset.values(), desc='link predicate'):
        for col in item['predicted_column_idx']:
            best_col = item['best_overlap_score'][col]
            predicate = item['predicate_overlap_' + str(best_col)]
            chosen_predicate = []
            predicate_linked_entities = {}
            for index, lines in enumerate(item['cell']):
                cell_value = lines[int(best_col)]
                best_cell_dict = item['related_info'][cell_value]
                try:
                    edges = best_cell_dict['hits']['hits'][0]['_source']['edges']
                except IndexError:
                    # print(best_cell_dict)
                    print('wrong out put at index:{} cell:{}'.format(item['index'], cell_value))
                for edge in edges:
                    if edge[0] in predicate:
                        chosen_predicate.append(edge)
                        candidate_entities = es.search(index=index_name, body=build_simple_dsl('id', edge[1]), size=1)
                        predicate_linked_entities[edge[0] + '_col=' + str(index)] = candidate_entities['hits']['hits']
            item['predicate_linked_entities_' + col] = predicate_linked_entities
    with jsonlines.open('./test/train_processed_KBlinked_total_linkedpredicate.jsonl', mode='w') as file:
        for _, value in dataset.items():
            file.write(value)
    return dataset


def connect_node(node_info: dict):
    node_info['hop'] += 1
    if int(node_info['id_1'][1:]) in node_info['edge_list_0']:
        print('Find in {} hops'.format(node_info['hop']))
        return node_info
    if node_info['hop'] == 5:
        print("Can't find in 5 hop neighbor")
        return 5
    for node in node_info['edge_list_0']:
        dsl = {
            'query': {
                'match': {'id': "Q" + str(node)}
            }
        }
        entity = es.search(index=index_name, doc_type=doc_type_name, body=dsl, size=1)['hits']['hits'][0]['_source']
        next_node_info = {
            'id_0': entity['id'],
            'id_1': node_info['id_1'],
            'edge_list_0': entity['edges'],
            'edge_list_1': node_info['edge_list_1'],
            'hop': node_info['hop']
        }
        connect_node(next_node_info)


def search(entity_id):
    key = 'id'
    dsl = {'query': {'match': {key: entity_id}}}
    candidate_entities = es.search(index=index_name, body=dsl, size=1)
    # print(candidate_entities)
    return candidate_entities['hits']['hits'][0]['_source']['label']


def return_dsl(key, value):
    return {
        "min_score": 1.0,
        "sort": ["_score"],
        'query': {
            'match': {key: value}
        }
    }


def build_path(set_0: list, node0_edge: dict, node_set_1: set):
    one_hop_cells = 0
    two_hop_cells = 0
    for nodes in set_0:
        if 'edges' in nodes:
            for connected_pair in nodes['edges']:
                predicate = connected_pair[0]
                wiki_id = connected_pair[1]

                if 'Q' not in wiki_id:
                    continue
                elif wiki_id in node_set_1:
                    node0_edge['edges'][wiki_id] = [wiki_id, 1]
                    one_hop_cells += 1
                    '''
                    try:
                        result0 = es.search(
                            index=index_name,
                            doc_type=doc_type_name,
                            body=return_dsl('id', wiki_id),
                            size=1)['hits']['hits'][0]['_source']
                    except IndexError:
                        print(wiki_id)
                        continue
                    if result0['id'] in node_set_1:
                        node0_edge['edges'][wiki_id] = [result0['id'], 1]
                        one_hop_cells += 1
                    '''
                else:
                    try:
                        result0 = es.search(
                            index=index_name,
                            doc_type=doc_type_name,
                            body=return_dsl('id', wiki_id),
                            size=1)['hits']['hits'][0]['_source']
                    except IndexError:
                        print(wiki_id)
                        continue
                    if 'edges' in result0:
                        for connected_pair_hop2 in result0['edges']:
                            if 'Q' not in connected_pair_hop2[1]:
                                continue
                            try:
                                result0_hop2 = es.search(index=index_name,
                                                         doc_type=doc_type_name,
                                                         body=return_dsl('id', connected_pair_hop2[1]),
                                                         size=1)['hits']['hits'][0]['_source']
                            except IndexError:
                                print(connected_pair_hop2[1])
                            if result0_hop2['id'] in node_set_1:
                                node0_edge['edges'][wiki_id] = [result0_hop2['id'], 2]
                                two_hop_cells += 1
    return [one_hop_cells, two_hop_cells]


def find_hops(node0, node1):
    one_hop_cells = 0
    two_hop_cells = 0
    set_0 = node0['match']
    set_1 = node1['match']
    node_set_0, node_set_1 = set(), set()  # Contains 30 nodes of set 0 and 1
    for linked_cell0 in set_0:
        if linked_cell0['id'] not in node_set_0:
            node_set_0.add(linked_cell0['id'])
    for linked_cell1 in set_1:
        if linked_cell1['id'] not in node_set_1:
            node_set_1.add(linked_cell1['id'])
            '''
    if node0['value'] == 'W. Garne' and node1['value'] == 'Wellingborough':
        print(node_set_0)
        print(node_set_1)
    '''
    l0 = build_path(set_0, node0, node_set_1)
    l1 = build_path(set_1, node1, node_set_0)
    one_hop_cells += (l0[0] + l1[0])
    two_hop_cells += (l0[1] + l1[1])
    return [one_hop_cells, two_hop_cells]
