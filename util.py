import pandas as pd
from pathlib import Path
import json
import numpy as np
import os
import torch
import tqdm


class Util:
    @staticmethod
    def to_label_dict_gt(label_path, label: pd.DataFrame):
        gt = pd.read_csv(label_path,
                         header=None,
                         names=['table_id', 'column_id', 'types'],
                         dtype={'column_id': np.int8})
        gt.sort_values(by=['table_id', 'column_id'], inplace=True)
        label_dict = {}
        for _, row in gt.iterrows():
            fid = row['table_id']
            label_dict[fid] = label_dict.get(fid, {})
            label_dict[fid][row['column_id']] = label[label['type'].isin([row['types']])].values.tolist()[0]
        return label_dict

    @staticmethod
    def csvdir_to_jsonl(csv_dir, out_jsonl_path, label_path=None, label=None):
        new_lines = []
        dict_gt = Util.to_label_dict_gt(label_path, label)
        index = 0
        table_size = 300
        for fpath in Path(csv_dir).glob('*.csv'):
            # print('Reading {}'.format(str(fpath)))
            table = pd.read_csv(fpath, header=None, keep_default_na=False, dtype=str,
                                ).replace(np.nan, '', regex=True).values
            table_cutted = table[:, :table_size].tolist()
            if table.size > table_size:
                row = int(table_size / table.shape[1])
                table_cutted = table[:row + 1, :].tolist()
            for l in table_cutted:
                for ind, cell in enumerate(l):
                    if len(cell) > table_size:
                        l[ind] = cell[:table_size]
                        print('cut {} characters per cell at {}. \ntotal {} cells'.format(len(cell) - len(l[ind]),
                                                                                          fpath.stem, table.size))

            # if table.size > max:
            #    max = table.size
            #    max_size=table.shape
            line = {'index': index, 'id': fpath.stem, 'table_data': table_cutted}
            if label_path is not None:
                line['predicted_column_idx'] = dict_gt[line['id']] if line['id'] in dict_gt else []
            if line['predicted_column_idx']:
                new_lines.append(line)
                index += 1
        Util.dump_lines(new_lines, out_jsonl_path + '/train_new.jsonl')

    @staticmethod
    def dump_lines(lines, out_path):
        with open(out_path, 'w') as fout:
            for line in lines:
                fout.write(json.dumps(line) + '\n')

    @staticmethod
    def load_dict(jsonl_path, key, length=-1):
        # print('loading {},,'.format(jsonl_path))
        data = {}
        with open(jsonl_path) as f:
            for idx, line in tqdm.tqdm(enumerate(f), desc='Loading Data'):
                if idx == length:
                    break
                else:
                    line = json.loads(line)
                    data[line[key]] = line
        return data

    @staticmethod
    def fill_token(data):
        max_len = 0
        mask_total = []
        for row in data:
            mask = []
            for cell in row:
                mask.append([1] * len(cell))
                if len(cell) > max_len:
                    max_len = len(cell)
            mask_total.append(mask)
        for i, row in enumerate(data):
            for j, cell in enumerate(row):
                mask_total[i][j].extend([0] * (max_len - len(cell)))
                cell.extend([0] * (max_len - len(cell)))
        '''
        for row in data:
            for cell in row:
                if len(cell.data['input_ids']) > max_len:
                    max_len = len(cell.data['input_ids'])
        for row in data:
            for cell in row:
                cell.data['input_ids'].extend([0] * (max_len - len(cell.data['input_ids'])))
                cell.data['token_type_ids'].extend([0] * (max_len - len(cell.data['token_type_ids'])))
                cell.data['attention_mask'].extend([0] * (max_len - len(cell.data['attention_mask'])))
        '''
        return mask_total

    @staticmethod
    def fill_token_col(data):
        max_len = 0
        mask_total = []
        for row in data:
            mask = [1] * len(row)
            if len(row) > max_len:
                max_len = len(row)
            mask_total.append(mask)
        for i, row in enumerate(data):
            mask_total[i].extend([0] * (max_len - len(row)))
            row.extend([0] * (max_len - len(row)))
        return mask_total

    @staticmethod
    def combine_initial_dims(tensor: torch.Tensor) -> torch.Tensor:
        """
        Given a (possibly higher order) tensor of ids with shape
        (d1, ..., dn, sequence_length)
        Return a view that's (d1 * ... * dn, sequence_length).
        If original tensor is 1-d or 2-d, return it as is.
        """
        if tensor.dim() <= 2:
            return tensor
        else:
            return tensor.view(-1, tensor.size(-1))

    @staticmethod
    def uncombine_initial_dims(tensor: torch.Tensor, original_size: torch.Size) -> torch.Tensor:
        """
        Given a tensor of embeddings with shape
        (d1 * ... * dn, sequence_length, embedding_dim)
        and the original shape
        (d1, ..., dn, sequence_length),
        return the reshaped tensor of embeddings with shape
        (d1, ..., dn, sequence_length, embedding_dim).
        If original size is 1-d or 2-d, return it as is.
        """
        if len(original_size) <= 2:
            return tensor
        else:
            view_args = list(original_size) + [tensor.size(-1)]
            return tensor.view(*view_args)

    @staticmethod
    def get_range_vector(size: int, device: int) -> torch.Tensor:
        """
        Returns a range vector with the desired size, starting at 0. The CUDA implementation
        is meant to avoid copy data from CPU to GPU.
        """
        if device > -1:
            return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
        else:
            return torch.arange(0, size, dtype=torch.long)

    @staticmethod
    def get_device_of(tensor: torch.Tensor) -> int:
        """
        Returns the device of the tensor.
        """
        if not tensor.is_cuda:
            return -1
        else:
            return tensor.get_device()

    @staticmethod
    def add_label(dataset: dict, path: str):
        with open(path, 'r') as file:
            label = json.load(file)
        for table in dataset.values():
            label_dict = {}
            table['label'] = label_dict
            if table['id'] not in label:
                print('{} does not contain label'.format(table['id']))
                continue
            for info in label[table['id']]:
                label_dict[int(info['col_idx'])] = info['label']
            table['label'] = label_dict
        return dataset
