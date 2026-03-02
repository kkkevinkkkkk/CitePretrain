from torch.utils.data import Dataset
from typing import Dict
import numpy as np
import torch

# modified from https://github.com/zitongyang/synthetic_continued_pretraining

def _get_bin(task_name: str, split: str):
    assert task_name in ['quality', 'rehearsal', 'instruct']
    bin_data_dir = 'data/dataset/bins'
    implemented_quality_split = {
        'entigraph': f'{bin_data_dir}/quality_all-entigraphgpt-4-turbo.bin',
    }
    implemented_rehearsal_split = {
        'rpj-train': f'{bin_data_dir}/togethercomputer_RedPajama_Data_1T_Sample_train.bin',
        'rpj-test': f'{bin_data_dir}/togethercomputer_RedPajama_Data_1T_Sample_test.bin'
    }
    implemented_instruct_split = {
        'ultrachat-train': f'{bin_data_dir}/ultrachat_train.bin',
        'ultrachat-test': f'{bin_data_dir}/ultrachat_test.bin'
    }
    if task_name == 'quality':
        assert split in implemented_quality_split
        return implemented_quality_split[split]
    elif task_name == 'rehearsal':
        assert split in implemented_rehearsal_split
        return implemented_rehearsal_split[split]
    elif task_name == 'instruct':
        assert split in implemented_instruct_split
        return implemented_instruct_split[split]
    else:
        raise NotImplementedError(f"Task {task_name} is not implemented")


class _MemmapDataset(Dataset):
    def __init__(self, block_size: int, bin_file: str, subsample_ratio: float=1.0):
        self.block_size = block_size
        self.ids = np.memmap(bin_file, dtype=np.int32, mode='r')
        self.ids = self.ids[:int(len(self.ids)*subsample_ratio)]

    def __len__(self):
        return int(len(self.ids)/self.block_size)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        assert i < len(self)
        start_ind = i*self.block_size
        end_ind = (i+1)*self.block_size
        x_id = self.ids[start_ind:end_ind].copy()
        return dict(input_ids=torch.from_numpy(x_id).long(),
                    labels=torch.from_numpy(x_id).long())

class CPTIndexDataset(_MemmapDataset):
    def __init__(self, block_size: int, bin_file: str,
                 subsample_ratio: float=1.0, index_ratio: float=0.0, random_state: int=42,
                 sot_token_id: int=128002, eot_token_id: int=128003):
        super().__init__(block_size, bin_file, subsample_ratio)
        self.index_ratio = index_ratio
        self.sot_token_id = sot_token_id
        self.eot_token_id = eot_token_id
        np.random.seed(random_state)

    def mask_invalid_tokens(self, token_ids):
        """
        1. Ignore any EOT that appears before a matching SOT (left-to-right).
        2. Ignore any SOT that does not have a matching EOT (right-to-left).
        3. Tokens in valid matched [SOT ... EOT] regions (including the SOT/EOT themselves)
           remain. Others get set to -100.
        """
        arr = np.array(token_ids, copy=True)
        unmatched = False
        UNMATCHED = -1

        # ---------------- PASS 1: Ignore EOT before any SOT ----------------
        c_sot = (arr == self.sot_token_id).cumsum()  # left to right
        c_eot = (arr == self.eot_token_id).cumsum()

        # An EOT is unmatched if at index i, #EOT so far > #SOT so far
        unmatched_eot_mask = (arr == self.eot_token_id) & (c_eot == 1) & (c_sot == 0)
        if unmatched_eot_mask.any():
            unmatched = True
            arr[unmatched_eot_mask] = UNMATCHED

        # Cumulative count of how many SOTs and EOTs we've seen so far
        c_sot = (arr == self.sot_token_id).cumsum()
        c_eot = (arr == self.eot_token_id).cumsum()

        # We are "inside" a region if #SOT encountered so far > #EOT encountered so far
        valid_mask = ((c_sot - c_eot) > 0) | (arr == self.sot_token_id) | (arr == self.eot_token_id)

        # Set all invalid tokens to -100
        if unmatched:
            arr = np.array(token_ids, copy=True)
        arr[~valid_mask] = -100

        return arr

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        assert i < len(self)
        start_ind = i * self.block_size
        end_ind = (i + 1) * self.block_size
        x_id = self.ids[start_ind:end_ind].copy()
        # find the tokens between sot and eot, there could be multiple sot and eot
        if np.random.rand() < self.index_ratio:
            labels = self.mask_invalid_tokens(x_id)
        else:
            labels = x_id

        return dict(input_ids=torch.from_numpy(x_id).long(),
                    labels=torch.from_numpy(labels).long())




class ReplayDataset(CPTIndexDataset):
    def __init__(self, block_size: int, bin_file: str, replay_file: str=None,
                 subsample_ratio: float=1.0,
                 replay_ratio: float=0.0, random_state: int=42,
                 index_ratio: float=1.0,
                 sot_token_id: int=128002, eot_token_id: int=128003,
                 ):
        super().__init__(block_size, bin_file, subsample_ratio, index_ratio, random_state, sot_token_id, eot_token_id)
        replay_file = replay_file if replay_file is not None else bin_file
        self.replay_data = _MemmapDataset(block_size, replay_file, 1.0)
        self.replay_ratio = replay_ratio
        self.original_len = int(len(self.ids)/self.block_size)
        total_len = self.original_len + int(self.original_len * self.replay_ratio)
        # set random seed
        np.random.seed(random_state)
        self.order = np.random.permutation(total_len)
        # self.order = np.arange(total_len)


    def __len__(self):
        replay_len = int(self.original_len * self.replay_ratio)
        return self.original_len + replay_len

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        idx = self.order[i]
        if idx < self.original_len:
            return super().__getitem__(idx)
        else:
            idx = idx - self.original_len
            return self.replay_data[idx]







class CPTDataset(_MemmapDataset):
    def __init__ (self, block_size: int, rehearsal_rate: float,
                 subsample_ratio: float):
        assert rehearsal_rate <= 1.0
        self.rehearsal_rate = rehearsal_rate
        self.rehearsal_data = _MemmapDataset(block_size, _get_bin('rehearsal', 'rpj-train'), 1.0)
        super().__init__(block_size,
                            _get_bin('quality', 'entigraph'),
                            subsample_ratio)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        if np.random.rand() < self.rehearsal_rate:
            idx = np.random.randint(len(self.rehearsal_data))
            return self.rehearsal_data[idx]
        else:
            return super().__getitem__(i)

def get_task_data_module(task_name: str,
                         block_size: int,
                         rehearsal_rate: float,
                         subsample_ratio: float,
                         **kwargs):
    if task_name == 'quality':
        train = CPTDataset(block_size, rehearsal_rate, subsample_ratio)
        val = _MemmapDataset(block_size, _get_bin('rehearsal', 'rpj-test'), 1.0)
        return dict(train_dataset=train, eval_dataset=val)
    if task_name == 'instruct':
        train = _MemmapDataset(block_size, _get_bin('instruct', 'ultrachat-train'), 1.0)
        val = _MemmapDataset(block_size, _get_bin('instruct', 'ultrachat-test'), 1.0)
        return dict(train_dataset=train, eval_dataset=val)
    else:
        raise NotImplementedError(f"Task {task_name} is not implemented")


if __name__ == '__main__':
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_fast=True)
    tokenizer.model_max_length = 2**20

    block_size = 2048
    rehearsal_rate = 0.1
    subsample_ratio = 1.0
    task_name = 'quality'
    data_module = get_task_data_module(task_name, block_size,
                                       rehearsal_rate, subsample_ratio)
    for example in data_module['train_dataset']:
        print(tokenizer.decode(example['input_ids']))
        break