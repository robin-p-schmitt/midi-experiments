import torch
from torch import nn
import torch.nn.functional as F  # noqa
import numpy as np
from typing import Callable, Optional
import os
import math

from dataset import get_data_loaders_for_training
from data_utils import HDF_FILE_PATH
from torch.nn.utils.rnn import pack_padded_sequence

MODEL_CHECKPOINTS_PATH = "model_checkpoints"


class PositionalEncoding(nn.Module):

  def __init__(self, d_model: int, max_len: int = 5000):
    super().__init__()

    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, 1, d_model)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)
    self.register_buffer('pe', pe, persistent=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Arguments:
        x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
    """
    return self.pe[:x.size(0)]


class TransformerDecoderModel(nn.Module):
  def __init__(self, input_dim, embed_dim, ff_dim, num_heads, num_layers, max_time_steps):
    super(TransformerDecoderModel, self).__init__()
    self.embedding = nn.Linear(input_dim, embed_dim)
    self.positional_encoding = PositionalEncoding(
      d_model=embed_dim,
      max_len=max_time_steps,
    )

    decoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
    self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

    self.output_layer = nn.Linear(embed_dim, input_dim)

  def forward(self, x):
    """

    :param x: input of shape [seq_len, batch_size, input_dim]
    :return:
    """
    x = self.embedding(x)
    x += self.positional_encoding(x)
    causal_mask = nn.Transformer.generate_square_subsequent_mask(x.size(0))
    x = self.transformer_decoder(
      x,
      mask=causal_mask,
      is_causal=True,
    )
    x = self.output_layer(x)

    return x


def calc_model_output_and_loss(
        piano_rolls: torch.Tensor,
        seq_lens: torch.Tensor,
        model: TransformerDecoderModel,
        criterion: Callable,
        optimizer: torch.optim.Optimizer,
        train_mode: bool,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        epoch=0,
):
  """

  :param piano_rolls: input of shape [seq_len, batch_size, input_dim]
  :param seq_lens:
  :param model:
  :param criterion:
  :param optimizer:
  :param train_mode:
  :param scheduler:
  :return:
  """
  if train_mode:
    model.train()
  else:
    model.eval()

  # shift right by 1
  piano_rolls_shifted = F.pad(piano_rolls, (0, 0, 0, 0, 1, 0))[:-1]  # [seq_len, batch_size, input_dim]
  outputs = model(piano_rolls_shifted)

  from data_utils import visualize_piano_roll
  visualize_piano_roll(
    torch.sigmoid(outputs).detach().cpu().permute(1, 2, 0)[0].numpy(),
    epoch
  )

  piano_rolls_packed = pack_padded_sequence(
    piano_rolls, seq_lens, batch_first=False, enforce_sorted=False)
  output_packed = pack_padded_sequence(
    outputs, seq_lens, batch_first=False, enforce_sorted=False)

  loss = criterion(output_packed.data, piano_rolls_packed.data)

  if train_mode:
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if scheduler:
      scheduler.step()

  return outputs, loss


def boosted_mse(outputs, ground_truth):
  """
  Boost MSE loss by giving more weight to non-zero values in the ground truth.
  :param outputs:
  :param ground_truth:
  :return:
  """
  error = outputs - ground_truth
  error[ground_truth != 0] *= 10
  loss_boosted = torch.mean(error ** 2)

  return loss_boosted


def train(
        model: TransformerDecoderModel,
        n_epochs: int,
        lr: float,
        batch_size: int,
        alias: str,
        criterion: Callable,
        lr_scheduling: Optional[str] = None,
):
  checkpoint_path = f"{MODEL_CHECKPOINTS_PATH}/{alias}.pt"
  if os.path.exists(checkpoint_path):
    print(f"Checkpoint found at {checkpoint_path}. Skipping training of {alias}.")
    return

  os.makedirs(MODEL_CHECKPOINTS_PATH, exist_ok=True)
  print(f"Training {alias}...")
  print(f"LR: {lr}")
  print(f"LR scheduling: ", lr_scheduling)
  print(f"Criterion: {criterion}")

  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  print(f"Using device: {device}")

  model.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  train_loader, devtrain_loader, dev_loader = get_data_loaders_for_training(
    hdf5_file_path=HDF_FILE_PATH,
    batch_size=batch_size,
  )

  if lr_scheduling:
    if lr_scheduling == "oclr":
      scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.01,
        steps_per_epoch=len(train_loader),
        epochs=n_epochs,
      )
    else:
      raise NotImplementedError
  else:
    scheduler = None

  for epoch in range(n_epochs):
    if scheduler:
      print(f"Starting epoch {epoch} with LR {scheduler.get_lr()}")
    for data_alias, data_loader in [
      ("train", train_loader),
      ("dev", dev_loader),
      ("devtrain", devtrain_loader),
    ]:
      train_mode = data_alias == "train"
      if train_mode:
        model.train()
      else:
        model.eval()
      for batch_idx, (piano_rolls, seq_lens) in enumerate(data_loader):
        piano_rolls = piano_rolls.to(device).transpose(0, 1)  # [seq_len, batch_size, input_dim]
        with torch.set_grad_enabled(train_mode):
          outputs, loss = calc_model_output_and_loss(
            piano_rolls,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            train_mode=train_mode,
            seq_lens=seq_lens,
            epoch=epoch
          )
        print(f"Epoch {epoch + 1}/{n_epochs}, Batch {batch_idx + 1}, {data_alias} Loss: {loss.item():.4f}")

  # Save model checkpoint
  torch.save(model.state_dict(), checkpoint_path)
