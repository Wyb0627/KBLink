import kb
import tqdm
import json
from argparse import ArgumentParser


def generate_feature_vec(dataset: dict, mode='head', filter_size=25):
    total_count = 0
    linked_count = 0
    for table_idx, table in tqdm.tqdm(dataset.items(), desc='process_neighbor'):
        total_count += len(table['label'])
        linked_vec = {}
        count_dict = {}
        if 'linked_cell' in table:
            for row_idx, linked_list_for_row in table['linked_cell'].items():
                for linked_dict in linked_list_for_row:
                    if 'match' not in linked_dict:
                        continue
                    if not linked_dict['match']:
                        continue
                    if (linked_dict['col_idx'] not in table['label'] and
                            str(linked_dict['col_idx']) not in table['label']):
                        continue
                    if linked_dict['col_idx'] not in linked_vec:
                        linked_vec[linked_dict['col_idx']] = []
                        count_dict[linked_dict['col_idx']] = 0
                    if count_dict[linked_dict['col_idx']] == filter_size:
                        continue
                    neighbor = look_up_neighbor(linked_dict['match'][0]["_source"])
                    if neighbor:
                        linked_vec[linked_dict['col_idx']].append(neighbor)
                        count_dict[linked_dict['col_idx']] += 1
                        if mode == 'head':
                            break
                if mode == 'head':
                    break

            linked_count += len(linked_vec)
        table['linked_cell'] = linked_vec

    print('Link count:{}, Total: {}, propotion: {}'.format(linked_count, total_count, linked_count / total_count))
    return dataset


def look_up_neighbor(entity: dict):
    neighbor_list = []
    if entity['types']:
        if isinstance(entity['types'], list):
            entity['types'] = entity['types'][0]
        label_edge = kb.search(entity['types'])
        if label_edge and 'disambiguation page' not in label_edge:
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
            # edge_id, entity_id = edge[0], edge[1]
            label_edge = kb.search(edge_id)
            label_entity = kb.search(entity_id)
            if not label_edge or not label_entity:
                continue
            if 'disambiguation page' in label_entity:
                continue
            neighbor_list.append(' ' + label_edge + ' ' + label_entity)
    if neighbor_list:
        return ' '.join(neighbor_list)
    else:
        return ''


if __name__ == '__main__':
    parser = ArgumentParser()
    # parser.add_argument("--train_csv_dir", help="input csv dir for training", default="./data/ft_cell/train_csv")
    parser.add_argument("--filter", help="row filter", type=str, default=25)
    parser.add_argument("--dataset", help="dataset", type=str, default='iswc')
    args = parser.parse_args()
    dataset_name = args.dataset
    filter_size = args.filter

    print('Start to read: processed_dataset_{}_{}.json'.format(dataset_name, filter_size))
    with open('./data_final/processed_dataset_{}_{}.json'.format(dataset_name, filter_size), 'r') as file:
        dataset = json.load(file)
    ten_dict = {}
    if filter_size == 'all':
        filter_size_input = 99999
    else:
        filter_size_input = int(filter_size)
    dataset = generate_feature_vec(dataset, mode='all', filter_size=filter_size_input)
    for i in range(3):
        ten_dict[i] = dataset[str(i)]
    with open('./data_final/processed_dataset_{}_{}.json'.format(dataset_name, filter_size), mode='w') as file:
        file.write(json.dumps(dataset, indent=4))
