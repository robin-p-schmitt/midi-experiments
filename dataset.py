from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import h5py
import numpy as np
import math
from typing import Callable, Optional, Union

from data_utils import (
  FRAMES_PER_BAR,
  NUM_BARS,
  decompress_played_notes,
  VALID_DATASET_NAMES,
  MAESTRO_HDF_FILE_PATH,
  LAKH_HDF_FILE_PATH,
  visualize_piano_roll,
  NUM_NOTES,
  SILENT_IDX,
)


class PianoRollDataset(Dataset):
  def __init__(self, hdf5_file, indices=None):
    self.hdf5_file = hdf5_file
    self.indices = indices
    self.file = None  # Placeholder for file handle
    self.transform = BarTransform(bars=NUM_BARS, num_notes=NUM_NOTES)

    # If no indices are provided, use the full dataset
    if self.indices is None:
      with h5py.File(hdf5_file, "r") as f:
        self.indices = np.arange(f["seq_lens"].shape[0])

    with h5py.File(hdf5_file, "r") as f:
      seq_lens = f["seq_lens"]

      # taken from https://github.com/yizhouzhao/MusicVAE/blob/master/src/data_utils.py
      self.index_mapper = []
      self.num_samples = 0
      for i in self.indices:
        num_splits = self.transform.get_num_sections(seq_lens[i])
        self.num_samples += num_splits
        for j in range(num_splits):
          self.index_mapper.append((i, j))

  def __len__(self):
    return self.num_samples

  def __getitem__(self, idx):
    if self.file is None:
      self.file = h5py.File(self.hdf5_file, "r")

    hdf_idx, section_idx = self.index_mapper[idx]
    start_idx = sum(self.file["seq_lens"][:hdf_idx])
    seq_len = self.file["seq_lens"][hdf_idx]
    played_notes = self.file["played_notes"][start_idx:start_idx + seq_len]
    piano_roll = decompress_played_notes(played_notes)

    piano_roll = self.transform(piano_roll)[section_idx]
    seq_len = piano_roll.shape[0]
    sampling_freq = self.file["sampling_freq"][hdf_idx]
    # visualize_piano_roll(piano_roll.T, sampling_freq, f"piano_roll_{idx}.png")
    # raise

    piano_roll = torch.tensor(piano_roll, dtype=torch.float32)
    seq_len = torch.tensor(seq_len, dtype=torch.int32)
    sampling_freq = torch.tensor(sampling_freq, dtype=torch.int32)

    return piano_roll, seq_len, sampling_freq

  def __del__(self):
    # Ensure the file handle is closed when the Dataset is destroyed
    if self.file is not None:
      self.file.close()
      self.file = None


class PlayedNotesDataset(PianoRollDataset):
  def __getitem__(self, idx):
    if self.file is None:
      self.file = h5py.File(self.hdf5_file, "r")

    hdf_idx, section_idx = self.index_mapper[idx]
    start_idx = sum(self.file["seq_lens"][:hdf_idx])
    seq_len = self.file["seq_lens"][hdf_idx]
    played_notes = self.file["played_notes"][start_idx:start_idx + seq_len]

    played_notes = self.transform(played_notes)[section_idx]
    seq_len = played_notes.shape[0]
    sampling_freq = self.file["sampling_freq"][hdf_idx]

    played_notes = torch.tensor(played_notes, dtype=torch.int32)
    seq_len = torch.tensor(seq_len, dtype=torch.int32)
    sampling_freq = torch.tensor(sampling_freq, dtype=torch.int32)

    return played_notes, seq_len, sampling_freq


# Collate function for dynamic padding
def collate_fn(batch):
    piano_rolls, seq_lens, sampling_freqs = zip(*batch)
    # Pad sequences to the max time_steps in the batch
    piano_rolls_padded = pad_sequence(piano_rolls, batch_first=True, padding_value=0.0)
    seq_lens = torch.stack(seq_lens)
    sampling_freqs = torch.stack(sampling_freqs)
    return piano_rolls_padded, seq_lens, sampling_freqs


def worker_init_fn(worker_id):
  worker_info = torch.utils.data.get_worker_info()
  dataset = worker_info.dataset
  dataset.file = None  # Ensure the file is opened fresh in each worker


# taken from https://github.com/yizhouzhao/MusicVAE/blob/master/src/data_utils.py
class BarTransform:
  def __init__(self, bars=1, num_notes=88, use_padding=False, sample_is_categorical: bool = False):
    self.split_size = bars * FRAMES_PER_BAR
    self.num_notes = num_notes
    self.use_padding = use_padding
    self.sample_is_categorical = sample_is_categorical

  def get_num_sections(self, sample_length: int):
    return math.ceil(sample_length / self.split_size)

  def __call__(self, sample: np.ndarray):
    """
    Split a sample into n bars
    :param sample: np.array of shape [seq_len, note_count]
    :return:
    """
    sample_length = sample.shape[0]

    # padding is done in the collate_fn -> we don't need to pad here?
    # if self.use_padding:
    #   # Pad the sample with 0's if there's not enough to create equal splits into n bars
    #   leftover = sample_length % self.split_size
    #   if leftover != 0:
    #     padding_size = self.split_size - leftover
    #     if self.sample_is_categorical:
    #       padding = np.full((padding_size,), SILENT_IDX)
    #     else:
    #       padding = np.zeros((padding_size, self.num_notes))
    #     sample = np.append(sample, padding, axis=0)

    sections = self.get_num_sections(sample_length)
    # Split into X equal sections
    split_list = np.array_split(sample, indices_or_sections=sections)

    return split_list


def _get_data_splits_for_training(dataset_size: int, dataset_fraction: float = 1.0):
  # Define the split ratios
  split_ratios = {"train": 0.95, "devtrain": 0.05, "dev": 0.05}

  # Generate shuffled indices
  indices = np.arange(dataset_size)
  np.random.seed(42)  # For reproducibility
  np.random.shuffle(indices)

  # Apply dataset fraction
  dataset_size = int(dataset_size * dataset_fraction)
  indices = indices[:dataset_size]

  # Compute split sizes
  train_end = int(split_ratios["train"] * dataset_size)
  devtrain_end = int(split_ratios["devtrain"] * dataset_size)

  # Create index splits
  train_indices = indices[:train_end]
  devtrain_indices = indices[:devtrain_end]  # devtrain is a subset of train
  dev_indices = indices[train_end:]

  return train_indices, devtrain_indices, dev_indices


def get_data_loaders_for_training(
        hdf_file_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        dataset_fraction: float = 1.0,
        dataset_cls: Callable = PianoRollDataset,
):
  # Fetch dataset size
  with h5py.File(hdf_file_path, "r") as f:
    if "seq_lens" in f:
      dataset_size = f["seq_lens"].shape[0]
    else:
      dataset_size = f["piano_rolls"].shape[0]

  # Get data splits
  train_indices, devtrain_indices, dev_indices = _get_data_splits_for_training(dataset_size, dataset_fraction)

  # Create datasets for each split
  train_dataset = dataset_cls(hdf_file_path, indices=train_indices)
  devtrain_dataset = dataset_cls(hdf_file_path, indices=devtrain_indices)
  dev_dataset = dataset_cls(hdf_file_path, indices=dev_indices)

  # Create DataLoaders
  train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    worker_init_fn=worker_init_fn,
    collate_fn=collate_fn
  )
  devtrain_loader = DataLoader(
    devtrain_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    worker_init_fn=worker_init_fn,
    collate_fn=collate_fn
  )
  dev_loader = DataLoader(
    dev_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    worker_init_fn=worker_init_fn,
    collate_fn=collate_fn
  )

  return train_loader, devtrain_loader, dev_loader
