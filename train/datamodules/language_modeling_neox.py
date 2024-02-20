# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm.py
from itertools import chain, zip_longest
from pathlib import Path
import pickle
from typing import Any, Dict, List, Union
import subprocess
import mmap
import os
import time

import numpy as np
import torch


from multiprocessing.shared_memory import SharedMemory

import numpy as np

import torch
from torch.utils.data.dataloader import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset

from pytorch_lightning import LightningDataModule

from train.datamodules.datasets.lm_dataset import LMDataset
from train.datamodules.datasets.indexed_dataset import MMapIndexedDataset
from train.datamodules.fault_tolerant_sampler import RandomFaultTolerantSampler
from train.datamodules.fault_tolerant_sampler import FaultTolerantDistributedSampler
from train.datamodules.datasets.detokenizer import DATASET_TOKENIZATION_REGISTRY
from train.utils.utils import get_logger, print_rank_zero

logger = get_logger()



class NeoxLMDataModule(LightningDataModule):
    def __init__(self, 
        max_length: int=1024,
        batch_size: int=32, 
        batch_size_eval: int=None, 
        global_batch_size: int=None,
        num_test_samples: int=1000,
        num_valid_samples: int=1000,
        max_steps: int=None,
        num_workers=1,
        pin_memory=False,
        drop_last: bool=False,
        seed: int = 42,
        **kwargs
    ):
        super().__init__()
        self.batch_size_eval = batch_size if batch_size_eval is None else batch_size_eval
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.max_length = max_length
        self.drop_last = drop_last

        self.num_train_samples = global_batch_size * max_steps
        self.num_test_samples = num_test_samples
        self.num_valid_samples = num_valid_samples
        self.seed = seed
        self.indices = range(10)

    def prepare_data(self, stage=None):
        # SE: `prepare_data` is only run on the main process (unlike setup) -- this is important 
        # because build_datasets calls `_build_index_mappings` which produces the
        # the sample of training examples to use 
        # DDP will then create a clone of this process for each GPU, and each of these 
        # clones will have access to the same datasets
        self.datasets = build_datasets(
            data_paths={
                "train": ["/var/cr06_data/sim_data/pile/pile/pile_text_document"],
                "test": ["/var/cr06_data/sim_data/pile/pile_test/pile_test_text_document"],
                "valid": ["/var/cr06_data/sim_data/pile/pile_validation/pile_validation_text_document"],
            },
            num_samples={
                "train": [self.num_train_samples],
                "test": [self.num_test_samples],
                "valid": [self.num_valid_samples],
            },
            seq_length=self.max_length,
            seed=self.seed,
            skip_warmup=True,
        )
    
    def setup(self, stage=None):
        pass

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """ The train dataloader """
        return self._data_loader(
            self.datasets["train"][0], batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The val dataloader """
        return self._data_loader(
            self.datasets["valid"][0], batch_size=self.batch_size_eval, shuffle=False
        )

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader """
        return self._data_loader(
            self.datasets["test"][0], batch_size=self.batch_size_eval, shuffle=False
        )
    
    def predict_dataloader(self, indices=None, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The predict dataloader """
        print(f"{self.indices=}")
        subset = torch.utils.data.Subset(self.datasets["test"][0], self.indices)
        # subset = self.datasets["test"][0]
        return self._data_loader(
            subset, batch_size=self.batch_size_eval, shuffle=False
        )

    def _data_loader(self, dataset: Dataset, batch_size: int, shuffle: bool = False,
                     sampler=None) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=1,  # Data is already in memory, we don't need many workers
            shuffle=shuffle,
            sampler=sampler,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
            # persistent_workers=True
        )

    def load_state_dict(self, checkpoint):
        if self.fault_tolerant:
            self.fast_forward_epochs = checkpoint['loops']['fit_loop']['epoch_progress']['current']['completed']
            # TD [2022-08-07] ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
            # behind, so we're using the optimizer's progress. This is set correctly in seq.py.
            self.fast_forward_batches = checkpoint['loops']['fit_loop']['epoch_loop.batch_progress']['current']['completed']
        # At this point the train loader hasn't been constructed yet


def build_datasets(
    data_paths: Dict[str, List[str]],
    num_samples: Dict[str, List[int]],
    seq_length: int,
    seed: int,
    skip_warmup: bool, 
    build_index_mappings=True,
):
    # build individual datasets
    datasets = {}
    for split in ["train", "test", "valid"]:
        datasets[split] = []
        for i, path in enumerate(data_paths[split]):
            dataset = build_dataset(
                data_prefix=path,
                name=f"train_{i}",
                num_samples=num_samples[split][i],
                seq_length=seq_length,
                seed=seed,
                skip_warmup=skip_warmup,
                build_index_mappings=build_index_mappings,
                source_index=i
            )
            datasets[split].append(dataset)
            
    return datasets



def build_dataset(
    data_prefix: str,
    name,
    num_samples,
    seq_length: int,
    seed,
    skip_warmup,
    build_index_mappings=True,
    source_index=None
):
    """Build train/valid/test datasets."""

    print(f"{data_prefix=}, {name=}, {num_samples=}, {seq_length=}, {seed=}, {skip_warmup=}")
    indexed_dataset = MMapIndexedDataset(data_prefix, skip_warmup)

    total_num_of_documents = indexed_dataset.sizes.shape[0]
    print_rank_zero("    {}:".format(name))
    print_rank_zero("     no. of documents:{}".format(total_num_of_documents))
    dataset = None
    documents = np.arange(start=0, stop=total_num_of_documents, step=1, dtype=np.int32)
    dataset = GPT2Dataset(
        name,
        data_prefix,
        documents,
        indexed_dataset,
        num_samples,
        seq_length,
        seed,
        build_index_mappings=build_index_mappings,
        source_index=source_index
    )

    return dataset



class GPT2Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        name,
        data_prefix,
        documents,
        indexed_dataset,
        num_samples,
        seq_length,
        seed,
        build_index_mappings=True,
        use_shared_fs=True,
        source_index: int = None,
    ):
        self.name = name
        self.data_prefix = data_prefix
        self.indexed_dataset = indexed_dataset
        self.source_index = source_index

        # Checks
        assert np.min(documents) >= 0
        assert np.max(documents) < indexed_dataset.sizes.shape[0]

        if build_index_mappings:
            # Build index mappings.
            self.doc_idx, self.sample_idx, self.shuffle_idx = _build_index_mappings(
                self.name,
                data_prefix,
                documents,
                self.indexed_dataset.sizes,
                num_samples,
                seq_length,
                seed,
                use_shared_fs=use_shared_fs,
            )
            self.shuffle_idx_len = self.shuffle_idx.shape[0] - 1
            self.sample_idx_len = self.sample_idx.shape[0] - 1

            if self.shuffle_idx_len != self.sample_idx_len:
                print(
                    f"WARNING: shuffle index length ({self.shuffle_idx_len}) is not equal to sample index length ({self.sample_idx_len})"
                )


    def __len__(self):
        return min(self.shuffle_idx_len, self.sample_idx_len)

    def __getitem__(self, idx):
        try:
            # Get the shuffled index.
            idx = self.shuffle_idx[idx]
            # Start and end documents and offsets.
            doc_index_f = self.sample_idx[idx][0]
            doc_index_l = self.sample_idx[idx + 1][0]
            offset_f = self.sample_idx[idx][1]
            offset_l = self.sample_idx[idx + 1][1]
            # If we are within the same document, just extract the chunk.
            if doc_index_f == doc_index_l:
                sample = self.indexed_dataset.get(
                    self.doc_idx[doc_index_f],
                    offset=offset_f,
                    length=offset_l - offset_f + 1,
                )
                doc_index = np.ones_like(sample, dtype=np.int64) * self.doc_idx[doc_index_f]
                doc_offset = np.arange(offset_f, offset_l + 1, dtype=np.int64)
            else:
                # Otherwise, get the rest of the initial document.
                sample_list = [
                    self.indexed_dataset.get(self.doc_idx[doc_index_f], offset=offset_f)
                ]
                # Loop over all in between documents and add the entire document.
                for i in range(doc_index_f + 1, doc_index_l):
                    sample_list.append(self.indexed_dataset.get(self.doc_idx[i]))
                # And finally add the relevant portion of last document.
                sample_list.append(
                    self.indexed_dataset.get(
                        self.doc_idx[doc_index_l], length=offset_l + 1
                    )
                )
                doc_index_list = [
                    np.ones_like(samples) * self.doc_idx[doc_index_f + i]
                    for i, samples in enumerate(sample_list)
                ]
                doc_offset_list = [np.arange(offset_f, sample_list[0].shape[0] + offset_f, dtype=np.int64)] + [
                    np.arange(samples.shape[0], dtype=np.int64)
                    for samples in sample_list[1:-1]
                ] + [np.arange(offset_l + 1, dtype=np.int64)]
                sample = np.concatenate(sample_list)
                doc_index = np.concatenate(doc_index_list).astype(np.int64)
                doc_offset = np.concatenate(doc_offset_list).astype(np.int64)
                assert sample.shape[0] == doc_index.shape[0]
                assert sample.shape[0] == doc_offset.shape[0]

            sample = np.array(sample, dtype=np.int64)
            loss_mask = np.ones_like(sample)
            
            return sample[:-1], sample[1:], {
                "text": sample, 
                "doc_index": doc_index, 
                "doc_offset": doc_offset,
                "loss_mask": loss_mask,
                "sample_index": np.ones_like(doc_index) * idx,
                "source_index": np.ones_like(doc_index) * self.source_index,
            }
        except IndexError:
            new_idx = idx % len(self)
            print(
                f"WARNING: Got index out of bounds error with index {idx} - taking modulo of index instead ({new_idx})"
            )
            return self[new_idx]


def _build_index_mappings(
    name,
    data_prefix,
    documents,
    sizes,
    num_samples,
    seq_length,
    seed,
    use_shared_fs=True,
):
    """Build doc-idx, sample-idx, and shuffle-idx.
    doc-idx: is an array (ordered) of documents to be used in training.
    sample-idx: is the start document index and document offset for each
       training sample.
    shuffle-idx: maps the sample index into a random index into sample-idx.
    """
    # Number of tokens in each epoch and number of required epochs.
    tokens_per_epoch = _num_tokens(documents, sizes)
    num_epochs = _num_epochs(tokens_per_epoch, seq_length, num_samples)
    # rng state
    np_rng = np.random.RandomState(seed=seed)

    # Filename of the index mappings.
    _filename = data_prefix
    _filename += "_{}_indexmap".format(name)
    _filename += "_{}ns".format(num_samples)
    _filename += "_{}sl".format(seq_length)
    _filename += "_{}s".format(seed)
    doc_idx_filename = _filename + "_doc_idx.npy"
    sample_idx_filename = _filename + "_sample_idx.npy"
    shuffle_idx_filename = _filename + "_shuffle_idx.npy"


    # Build the indexed mapping if not exist.
    if (
        (not os.path.isfile(doc_idx_filename))
        or (not os.path.isfile(sample_idx_filename))
        or (not os.path.isfile(shuffle_idx_filename))
    ):
        print_rank_zero(
            " > WARNING: could not find index map files, building "
            "the indices on rank 0 ..."
        )
        # doc-idx.
        start_time = time.time()
        doc_idx = _build_doc_idx(documents, num_epochs, np_rng)
        np.save(doc_idx_filename, doc_idx, allow_pickle=True)
        print_rank_zero(
            " > elapsed time to build and save doc-idx mapping "
            "(seconds): {:4f}".format(time.time() - start_time)
        )
        # sample-idx.
        start_time = time.time()
        # Use C++ implementation for speed.
        from train.datamodules.neox_utils import helpers

        assert doc_idx.dtype == np.int32
        assert sizes.dtype == np.int32

        num_samples = (num_epochs * tokens_per_epoch - 1) / seq_length
        if 2 * (num_samples + 1) < np.iinfo(np.int32).max:
            sample_idx = helpers.build_sample_idx_int32(
                sizes, doc_idx, seq_length, num_epochs, tokens_per_epoch
            )
        else:
            sample_idx = helpers.build_sample_idx_int64(
                sizes, doc_idx, seq_length, num_epochs, tokens_per_epoch
            )
        np.save(sample_idx_filename, sample_idx, allow_pickle=True)
        print_rank_zero(
            " > elapsed time to build and save sample-idx mapping "
            "(seconds): {:4f}".format(time.time() - start_time)
        )
        # shuffle-idx.
        start_time = time.time()
        # -1 is due to data structure used to retrieve the index:
        #    sample i --> [sample_idx[i], sample_idx[i+1])
        shuffle_idx = _build_shuffle_idx(sample_idx.shape[0] - 1, np_rng)
        np.save(shuffle_idx_filename, shuffle_idx, allow_pickle=True)
        print_rank_zero(
            " > elapsed time to build and save shuffle-idx mapping"
            " (seconds): {:4f}".format(time.time() - start_time)
        )


    # Load mappings.
    start_time = time.time()
    print_rank_zero(" > loading doc-idx mapping from {}".format(doc_idx_filename))
    doc_idx = np.load(doc_idx_filename, allow_pickle=True, mmap_mode="r")
    print_rank_zero(" > loading sample-idx mapping from {}".format(sample_idx_filename))
    sample_idx = np.load(sample_idx_filename, allow_pickle=True, mmap_mode="r")
    print_rank_zero(" > loading shuffle-idx mapping from {}".format(shuffle_idx_filename))
    shuffle_idx = np.load(shuffle_idx_filename, allow_pickle=True, mmap_mode="r")
    print_rank_zero(
        "    loaded indexed file in {:3.3f} seconds".format(time.time() - start_time)
    )
    print_rank_zero("    total number of samples: {}".format(sample_idx.shape[0]))
    print_rank_zero("    total number of epochs: {}".format(num_epochs))

    return doc_idx, sample_idx, shuffle_idx


def _num_tokens(documents, sizes):
    """Total number of tokens in the dataset."""
    return np.sum(sizes[documents])


def _num_epochs(tokens_per_epoch, seq_length, num_samples):
    """Based on number of samples and sequence length, calculate how many
    epochs will be needed."""
    num_epochs = 0
    total_tokens = 0
    while True:
        num_epochs += 1
        total_tokens += tokens_per_epoch
        # -1 is because we need to retrieve seq_length + 1 token each time
        # but the last token will overlap with the first token of the next
        # sample except for the last sample.
        if ((total_tokens - 1) // seq_length) >= num_samples:
            return num_epochs


def _build_doc_idx(documents, num_epochs, np_rng):
    """Build an array with length = number-of-epochs * number-of-documents.
    Each index is mapped to a corresponding document."""
    doc_idx = np.mgrid[0:num_epochs, 0 : len(documents)][1]
    doc_idx[:] = documents
    doc_idx = doc_idx.reshape(-1)
    doc_idx = doc_idx.astype(np.int32)
    np_rng.shuffle(doc_idx)
    return doc_idx



def _build_shuffle_idx(size, np_rng):
    """Build the range [0, size) and shuffle."""
    dtype_ = np.uint32
    if size >= (np.iinfo(np.uint32).max - 1):
        dtype_ = np.int64
    shuffle_idx = np.arange(start=0, stop=size, step=1, dtype=dtype_)
    np_rng.shuffle(shuffle_idx)
    return shuffle_idx
