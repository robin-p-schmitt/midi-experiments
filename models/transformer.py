import torch
from torch import nn
import torch.nn.functional as F  # noqa
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Callable, Optional
import os
import math
from torch.nn.utils.rnn import pack_padded_sequence

from dataset import get_data_loaders_for_training
from data_utils import visualize_piano_roll, FRAMES_PER_BAR

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
  def __init__(self, input_dim, embed_dim, num_heads, num_layers, max_time_steps):
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

  return outputs, loss, piano_rolls_shifted


def train(
        model: TransformerDecoderModel,
        n_epochs: int,
        lr: float,
        batch_size: int,
        alias: str,
        criterion: Callable,
        lr_scheduling: Optional[str] = None,
        max_lr: float = 0.01,
        dataset_name: str = "maestro",
        dataset_fraction: float = 1.0,
        load_checkpoint: Optional[str] = None,
):
  checkpoint_path = f"{MODEL_CHECKPOINTS_PATH}/{alias}"
  if os.path.exists(checkpoint_path):
    print(f"Checkpoint found at {checkpoint_path}. Skipping training of {alias}.")
    return

  os.makedirs(checkpoint_path, exist_ok=True)
  print(f"Training {alias}...")
  print(f"LR: {lr}")
  print(f"LR scheduling: ", lr_scheduling)
  print(f"Criterion: {criterion}")

  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  print(f"Using device: {device}")

  if load_checkpoint:
    model.load_state_dict(torch.load(load_checkpoint), strict=False)
  model.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  train_loader, devtrain_loader, dev_loader = get_data_loaders_for_training(
    dataset_name=dataset_name,
    batch_size=batch_size,
    dataset_fraction=dataset_fraction,
  )

  if lr_scheduling:
    if lr_scheduling == "oclr":
      scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=len(train_loader),
        epochs=n_epochs,
      )
    else:
      raise NotImplementedError
  else:
    scheduler = None

  # Initialize TensorBoard writer
  writer = SummaryWriter(log_dir=f"./runs/{alias}")

  avg_losses_per_epoch = {"train": [], "dev": [], "devtrain": []}
  best_dev_epoch = 1

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

      running_loss = 0.0
      for batch_idx, (piano_rolls, seq_lens, _) in enumerate(data_loader):
        piano_rolls = piano_rolls.to(device).transpose(0, 1)  # [seq_len, batch_size, input_dim]
        with torch.set_grad_enabled(train_mode):
          outputs, loss, piano_rolls_shifted = calc_model_output_and_loss(
            piano_rolls,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            train_mode=train_mode,
            seq_lens=seq_lens,
          )
          running_loss += loss.item()

          # if train_mode:
          #   from data_utils import visualize_piano_roll
          #   for name, tensor in [("outputs", outputs), ("piano_rolls_shifted", piano_rolls_shifted)]:
          #     visualize_piano_roll(
          #       torch.sigmoid(tensor).detach().cpu().permute(1, 2, 0)[1].numpy(),
          #       8,
          #       f"{name}_{epoch}"
          #     )

        # mem_info = torch.cuda.mem_get_info(device)
        print(
          f"Epoch {epoch + 1}/{n_epochs}, Batch {batch_idx + 1} - "
          f"{data_alias} Loss: {loss.item():.4f} - "
          # f"Free/Total Memory: {(mem / 1024 ** 3 for mem in mem_info)} GB"
        )

      avg_loss = running_loss / len(data_loader)
      avg_losses_per_epoch[data_alias].append(avg_loss)
      writer.add_scalar(f"{data_alias}_loss/batch", avg_loss, epoch)

      if scheduler and data_alias == "train":
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

    # Save model checkpoint
    torch.save(model.state_dict(), f"{checkpoint_path}/model_epoch_{epoch + 1}.pt")

    # remove all checkpoints except for the best and last one
    best_dev_epoch = np.argmin(avg_losses_per_epoch["dev"])
    for epoch_file in os.listdir(checkpoint_path):
      assert epoch_file.startswith("model_epoch_")
      model_epoch = int(epoch_file.split("_")[-1].split(".")[0])
      if model_epoch not in [best_dev_epoch + 1, epoch + 1]:
        os.remove(f"{checkpoint_path}/{epoch_file}")

  print(f"Models saved to {checkpoint_path}")
  writer.close()

  best_dev_checkpoint_path = f"{checkpoint_path}/model_epoch_{best_dev_epoch + 1}.pt"
  test(
    model=model,
    checkpoint_path=best_dev_checkpoint_path,
    alias=alias,
    criterion=criterion,
    dataset_name=dataset_name,
  )


def test(
        model: TransformerDecoderModel,
        checkpoint_path: str,
        alias: str,
        criterion: Callable,
        dataset_name: str = "maestro",
        n_context_bars: int = 4,
        num_samples: int = 5,
):
  def _visualize_piano_roll(piano_rolls_tensor: torch.Tensor, sampling_freq_tensor: torch.Tensor, filename: str):
    """

    :param piano_rolls_tensor: tensor of shape [seq_len, batch_size, num_notes]
    """

    visualize_piano_roll(
      piano_rolls_tensor.permute(1, 2, 0)[0].cpu().numpy(),
      sampling_freq=sampling_freq_tensor[0].item(),
      filename=filename
    )

  model.load_state_dict(torch.load(checkpoint_path), strict=False)

  piano_roll_dir = f"{os.path.dirname(checkpoint_path)}/piano_rolls"
  os.makedirs(piano_roll_dir, exist_ok=True)

  print(f"Testing {alias}...")
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  print(f"Using device: {device}")
  model.to(device)

  n_context_frames = n_context_bars * FRAMES_PER_BAR

  _, _, dev_loader = get_data_loaders_for_training(
    dataset_name=dataset_name,
    batch_size=1,
    dataset_fraction=1.0,
  )

  for data_alias, data_loader in [
    ("dev", dev_loader),
    # ("devtrain", devtrain_loader),
  ]:
    model.eval()

    for batch_idx, (piano_rolls, seq_lens, sampling_freq) in enumerate(data_loader):
      piano_rolls = piano_rolls.to(device).transpose(0, 1)  # [seq_len, batch_size, input_dim]
      _visualize_piano_roll(
        piano_rolls,
        sampling_freq,
        f"{piano_roll_dir}/gt_piano_roll_{batch_idx}"
      )

      piano_rolls_shifted = F.pad(piano_rolls, (0, 0, 0, 0, 1, 0))[:-1]  # [seq_len, batch_size, input_dim]
      with torch.no_grad():
        n_prediction_steps = piano_rolls.size(0) - n_context_frames
        # context frames to start prediction
        piano_rolls_context = piano_rolls_shifted[:n_context_frames].to(device)
        # create tensor to hold the predicted tensor
        predicted_piano_rolls = torch.zeros(
          (piano_rolls.size(0) + 1, *piano_rolls.size()[1:]),
          device=device
        )
        # insert context into tensor while considering initial 0 padding frame
        predicted_piano_rolls[:n_context_frames] = piano_rolls_context

        for step in range(n_prediction_steps + 1):
          logits = model(predicted_piano_rolls[:n_context_frames + step])
          outputs = torch.sigmoid(logits)
          # outputs[outputs >= 0.5] = 1
          # outputs[outputs < 0.5] = 0
          predicted_piano_rolls[n_context_frames + step] = outputs[-1]

        predicted_piano_rolls = predicted_piano_rolls[1:]  # remove initial 0 padding frame
        _visualize_piano_roll(
          predicted_piano_rolls,
          sampling_freq,
          f"{piano_roll_dir}/pred_piano_roll_{batch_idx}"
        )

      if batch_idx == num_samples:
        break

  print(f"Predictions saved to {piano_roll_dir}")
