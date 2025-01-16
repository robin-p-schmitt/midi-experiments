from torch.utils.data import Dataset, DataLoader
import torch
import h5py
import numpy as np


class PianoRollDataset(Dataset):
  def __init__(self, hdf5_file, indices=None, max_time_steps=500):
    self.hdf5_file = hdf5_file
    self.indices = indices
    self.max_time_steps = max_time_steps
    self.file = None  # Placeholder for file handle

    # If no indices are provided, use the full dataset
    if self.indices is None:
      with h5py.File(hdf5_file, "r") as f:
        self.indices = np.arange(f["piano_rolls"].shape[0])

  def __len__(self):
    return len(self.indices)

  def __getitem__(self, idx):
    if self.file is None:
      self.file = h5py.File(self.hdf5_file, "r")

    actual_idx = self.indices[idx]
    piano_roll = self.file["piano_rolls"][actual_idx]

    piano_roll = torch.tensor(piano_roll, dtype=torch.float32)
    piano_roll = piano_roll.permute(1, 0)
    return piano_roll

  def __del__(self):
    # Ensure the file handle is closed when the Dataset is destroyed
    if self.file is not None:
      self.file.close()
      self.file = None


def worker_init_fn(worker_id):
  worker_info = torch.utils.data.get_worker_info()
  dataset = worker_info.dataset
  dataset.file = None  # Ensure the file is opened fresh in each worker


def _get_data_splits_for_training(dataset_size):
  # Define the split ratios
  split_ratios = {"train": 0.9, "devtrain": 0.05, "dev": 0.05}

  # Generate shuffled indices
  indices = np.arange(dataset_size)
  np.random.seed(42)  # For reproducibility
  np.random.shuffle(indices)

  # Compute split sizes
  train_end = int(split_ratios["train"] * dataset_size)
  devtrain_end = train_end + int(split_ratios["devtrain"] * dataset_size)

  # Create index splits
  train_indices = indices[:train_end]
  devtrain_indices = indices[train_end:devtrain_end]
  dev_indices = indices[devtrain_end:]

  return train_indices, devtrain_indices, dev_indices


def get_data_loaders_for_training(
        hdf5_file_path: str,
        batch_size: int = 32,
        num_workers: int = 2
):
  # Fetch dataset size
  with h5py.File(hdf5_file_path, "r") as f:
    dataset_size = f["piano_rolls"].shape[0]

  # Get data splits
  train_indices, devtrain_indices, dev_indices = _get_data_splits_for_training(dataset_size)

  # Create datasets for each split
  train_dataset = PianoRollDataset(hdf5_file_path, indices=train_indices)
  devtrain_dataset = PianoRollDataset(hdf5_file_path, indices=devtrain_indices)
  dev_dataset = PianoRollDataset(hdf5_file_path, indices=dev_indices)

  # Create DataLoaders
  train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    worker_init_fn=worker_init_fn
  )
  devtrain_loader = DataLoader(
    devtrain_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    worker_init_fn=worker_init_fn
  )
  dev_loader = DataLoader(
    dev_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    worker_init_fn=worker_init_fn
  )

  return train_loader, devtrain_loader, dev_loader
