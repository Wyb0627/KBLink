import os
from util import Util
import torch
import numpy as np
import random
import kb


def set_cpu_num(cpu_num):
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    setup_seed(0)
    set_cpu_num(32)
    base_dir = '../data/'
    dataset = Util.load_dict(base_dir + 'viznet_dataset.jsonl', 'index')
    dataset = kb.clean_link(dataset)
    dataset, hop_count_list = kb.link(dataset, 'linked_cell', link_only=True, size1=30, size2=10, hop=1)
    print(hop_count_list)
    print('create_complete')
