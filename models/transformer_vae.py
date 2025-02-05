import copy
import json
import torch
from torch import nn
import torch.nn.functional as F  # noqa
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Callable, Optional, Dict
import os
import math
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim.lr_scheduler import OneCycleLR, ExponentialLR
from torch.optim import Adam
import matplotlib.pyplot as plt

from dataset import get_data_loaders_for_training, PianoRollDataset, PlayedNotesDataset
from data_utils import (
  visualize_piano_roll,
  FRAMES_PER_BAR,
  VALID_DATASET_NAMES,
  MAESTRO_MONOPHONIC_HDF_FILE_PATH,
  LAKH_MONOPHONIC_HDF_FILE_PATH,
  SILENT_IDX,
  NUM_BARS
)

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


class VAETransformerEncoder(nn.Module):
  def __init__(self, input_dim, d_model, num_heads, num_layers, max_time_steps):
    super(VAETransformerEncoder, self).__init__()
    self.embedding = nn.Embedding(input_dim, d_model)
    self.positional_encoding = PositionalEncoding(
      d_model=d_model,
      max_len=max_time_steps,
    )

    encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    self.encoder_out = nn.Linear(d_model, 2 * d_model)

  def forward(self, x):
    """

    :param x: input of shape [seq_len + 1, batch_size, input_dim]. The last frame is a special frame which later
    acts as the hidden representation of the input.
    :return:
    """
    x = self.embedding(x)
    x += self.positional_encoding(x)
    x = self.transformer_encoder(x)
    # use the last output as the hidden representation of the input
    enc_out = x[-1]
    enc_out = self.encoder_out(enc_out)

    mu, log_var = torch.chunk(enc_out, 2, dim=-1)
    log_var = F.softplus(log_var)
    sigma = torch.exp(log_var * 2)

    with torch.no_grad():
      batch_size = x.size(1)
      epsilon = torch.randn_like(mu)

    # reparameterization trick to get latent features
    z = mu + epsilon * sigma

    return z, mu, log_var


class VAEHierarchicalTransformerDecoder(nn.Module):
  def __init__(
          self,
          d_model,
          output_dim,
          num_heads,
          conductor_num_layers,
          decoder_num_layers,
          max_time_steps,
          conductor_in_act=None,
          transform_conductor_out=False,
  ):
    super(VAEHierarchicalTransformerDecoder, self).__init__()
    self.positional_encoding = PositionalEncoding(
      d_model=d_model,
      max_len=max_time_steps,
    )
    self.embedding = nn.Embedding(output_dim, d_model)

    decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
    self.conductor = nn.TransformerEncoder(decoder_layer, num_layers=conductor_num_layers)
    self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_num_layers)

    self.conductor_in = nn.Linear(d_model, d_model)
    if conductor_in_act:
      self.conductor_in_act = conductor_in_act
    else:
      self.conductor_in_act = lambda x: x

    if transform_conductor_out:
      self.conductor_out = nn.Linear(d_model, d_model)
      self.conductor_out_act = torch.tanh
    else:
      self.conductor_out = lambda x: x
      self.conductor_out_act = lambda x: x

    self.output_layer = nn.Linear(d_model, output_dim)

  def forward(self, z: torch.Tensor, output_seq_len: int, num_sub_seqs: int, x: Optional[torch.Tensor]):
    """

    :param z: hidden representation shape [batch_size, d_model]
    :param output_seq_len: number of frames of the output (= number of frames of the encoder input)
    :param num_sub_seqs: chunk the output into num_sub_seqs sub sequences
    :param x: optional input of shape [output_seq_len, batch_size, input_dim]
    :return:
    """
    z = self.conductor_in(z)
    z = self.conductor_in_act(z)

    # create zero tensor to hold the predicted tensor
    conductor_seq = torch.zeros((num_sub_seqs + 1, *z.size()), device=z.device)
    # the first frame is the given hidden representation
    conductor_seq[0] = z
    for step in range(1, num_sub_seqs + 1):
      # add positional encoding to the input
      conductor_seq_w_pos_enc = conductor_seq[:step] + self.positional_encoding(conductor_seq[:step])
      # next frame is last output of the transformer
      conductor_seq[step] = self.conductor(conductor_seq_w_pos_enc)[-1]

    conductor_seq = conductor_seq[1:]  # [num_sub_seqs, batch_size, d_model]
    conductor_seq = conductor_seq.unsqueeze(0)  # [1, num_sub_seqs, batch_size, d_model]
    conductor_seq = self.conductor_out(conductor_seq)
    conductor_seq = self.conductor_out_act(conductor_seq)

    if x is not None:
      x = self.embedding(x)  # [output_seq_len, batch_size, d_model]
      # chunk the output into num_sub_seqs sub sequences which are processed independently
      sub_seq_len = output_seq_len // num_sub_seqs
      x = torch.reshape(x, [sub_seq_len, num_sub_seqs, *x.size()[1:]])  # [sub_seq_len, num_sub_seqs, batch_size, d_model]

      # for each sub sequence, get the last note from the previous sub sequence
      # for the first sub sequence, use a zero vector
      context_notes =x[-1]  # [num_sub_seqs, batch_size, d_model]
      context_notes = torch.cat(
        (torch.zeros(1, *context_notes.size()[1:], device=context_notes.device), context_notes),
        dim=0
      )[:-1]  # [num_sub_seqs, batch_size, d_model]
      context_notes = context_notes.unsqueeze(0)  # [1, num_sub_seqs, batch_size, d_model]

      # concatenate the conductor sequence with the input
      x_ext = torch.cat((conductor_seq, context_notes, x), dim=0)  # [2 + sub_seq_len, num_sub_seqs, batch_size, d_model]
      # remove the last frame of the input (shift right)
      x_ext = x_ext[:-1]  # [1 + sub_seq_len, num_sub_seqs, batch_size, d_model]
      # combine the batch dimension with the num_sub_seqs dimension
      x_ext = torch.reshape(x_ext, [1 + sub_seq_len, num_sub_seqs * x.size(2), x.size(3)])  # [1 + sub_seq_len, num_sub_seqs * batch_size, d_model]

      x_ext += self.positional_encoding(x_ext)
      causal_mask = nn.Transformer.generate_square_subsequent_mask(x_ext.size(0))
      outputs = self.decoder(
        x_ext,
        mask=causal_mask,
        is_causal=True,
      )
      outputs = outputs[1:]  # [sub_seq_len, num_sub_seqs * batch_size, d_model]
      outputs = torch.reshape(
        outputs,
        [sub_seq_len, num_sub_seqs, x.size(2), x.size(3)]
      )  # [sub_seq_len, num_sub_seqs, batch_size, d_model]
      outputs = torch.reshape(outputs, [output_seq_len, x.size(2), x.size(3)])  # [output_seq_len, batch_size, d_model]
    else:
      raise NotImplementedError

    return self.output_layer(outputs)


class VAETransformerDecoder(nn.Module):
  def __init__(
          self,
          d_model,
          output_dim,
          num_heads,
          decoder_num_layers,
          max_time_steps,
  ):
    super(VAETransformerDecoder, self).__init__()
    self.positional_encoding = PositionalEncoding(
      d_model=d_model,
      max_len=max_time_steps,
    )
    self.embedding = nn.Embedding(output_dim, d_model)

    decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
    self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_num_layers)

    self.output_layer = nn.Linear(d_model, output_dim)

  def forward(self, z: torch.Tensor, x: Optional[torch.Tensor]):
    """

    :param z: hidden representation shape [batch_size, d_model]
    :param x: optional input of shape [output_seq_len, batch_size, input_dim]
    :return:
    """

    if x is not None:
      x = self.embedding(x)  # [output_seq_len, batch_size, d_model]

      # concatenate the conductor sequence with the input
      x_ext = torch.cat((z.unsqueeze(0), x), dim=0)  # [1 + output_seq_len, batch_size, d_model]
      # remove the last frame of the input (shift right)
      x_ext = x_ext[:-1]  # [output_seq_len, batch_size, d_model]

      x_ext += self.positional_encoding(x_ext)
      causal_mask = nn.Transformer.generate_square_subsequent_mask(x_ext.size(0))
      outputs = self.decoder(
        x_ext,
        mask=causal_mask,
        is_causal=True,
      )
    else:
      raise NotImplementedError

    return self.output_layer(outputs)


class VAEModel(nn.Module):
  def __init__(
          self,
          input_dim,
          output_dim,
          max_time_steps,
          encoder_opts: Dict,
          decoder_opts: Dict,
          decoder_cls=VAEHierarchicalTransformerDecoder,
  ):
    super(VAEModel, self).__init__()

    self.encoder = VAETransformerEncoder(
      input_dim=input_dim,
      max_time_steps=max_time_steps,
      **encoder_opts,
    )

    self.decoder = decoder_cls(
      output_dim=output_dim,
      max_time_steps=max_time_steps,
      **decoder_opts,
    )

  def encode(self, x):
    return self.encoder(x)

  def decode(self, z, output_seq_len, num_sub_seqs, x=None):
    return self.decoder(z, output_seq_len, num_sub_seqs, x)


class CyclicAnnealingScheduler:
  def __init__(
          self,
          M: int = 4,
          R: float = 0.5,
          f: Callable = lambda x: 2 * x,
          max_value: float = 1.0,
  ):
    self.num_steps_ = None
    self.ratio = None
    self.ratio_ceil = None

    self.M = M
    self.R = R
    self.f = f
    self.max_value = max_value

  @property
  def num_steps(self):
    assert self.num_steps_ is not None, "num_steps not set"
    return self.num_steps_

  @num_steps.setter
  def num_steps(self, num_steps: int):
    self.num_steps_ = num_steps
    self._set_ratios()

  def _set_ratios(self):
    self.ratio = self.num_steps / self.M
    self.ratio_ceil = math.ceil(self.ratio)

  def __call__(self, n: int):
    """

    :param n: integer in range(1, self.num_steps + 1)
    :return:
    """

    tau = (n - 1) % self.ratio_ceil
    tau /= self.ratio
    beta = self.f(tau) if tau <= self.R else 1
    beta *= self.max_value

    return beta


def calc_model_output_and_loss(
        played_notes: torch.Tensor,
        seq_lens: torch.Tensor,
        model: VAEModel,
        criterion: Callable,
        optimizer: torch.optim.Optimizer,
        train_mode: bool,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        beta: Optional[float] = 0.01,
        min_lr: Optional[float] = None,
        beta_scheduler: Optional[CyclicAnnealingScheduler] = None,
):
  """

  :param played_notes:
  :param seq_lens:
  :param model:
  :param criterion:
  :param optimizer:
  :param train_mode:
  :param scheduler:
  :param beta:
  :param min_lr:
  :param beta_scheduler:
  :return:
  """
  if train_mode:
    model.train()
  else:
    model.eval()

  # append special idx. in this frame, the transformer encoder should store the hidden representation z
  # from which the decoder should reconstruct the original input sequence
  played_notes_ext = torch.cat(
    (played_notes, played_notes[-1].unsqueeze(0)),
  )
  played_notes_ext = played_notes_ext.scatter(0, index=seq_lens.to(played_notes.device).to(torch.int64).unsqueeze(0), value=SILENT_IDX + 1)

  z, mu, log_var = model.encode(x=played_notes_ext)
  outputs = model.decode(z=z, output_seq_len=played_notes.size(0), num_sub_seqs=NUM_BARS, x=played_notes)

  played_notes_packed = pack_padded_sequence(
    played_notes, seq_lens, batch_first=False, enforce_sorted=False)
  output_packed = pack_padded_sequence(
    outputs, seq_lens, batch_first=False, enforce_sorted=False)

  recon_loss = criterion(output_packed.data, played_notes_packed.data)
  kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
  kl_loss = torch.mean(kl_loss)
  kl_loss *= beta  # kl weight

  loss = recon_loss + kl_loss

  if train_mode:
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if scheduler and (not min_lr or scheduler.get_last_lr()[0] > min_lr):
      scheduler.step()

  return outputs, loss, recon_loss, kl_loss


def train(
        model: VAEModel,
        n_epochs: int,
        batch_size: int,
        alias: str,
        criterion: Callable,
        optimizer_opts: Dict,
        lr_scheduling_opts: Optional[Dict] = None,
        dataset_name: str = "maestro",
        dataset_fraction: float = 1.0,
        load_checkpoint: Optional[str] = None,
        beta: float = 0.01,
        beta_scheduler_opts: Optional[Dict] = None,
):
  checkpoint_path = f"{MODEL_CHECKPOINTS_PATH}/{alias}"
  if os.path.exists(checkpoint_path):
    print(f"Checkpoint found at {checkpoint_path}. Skipping training of {alias}.")
    return

  training_stats_file_path = f"{checkpoint_path}/training_stats.txt"
  training_stats = {}

  os.makedirs(checkpoint_path, exist_ok=True)
  print(f"Training {alias}...")
  print(f"Optimizer: {optimizer_opts}")
  print(f"LR scheduling: {lr_scheduling_opts}")
  print(f"Criterion: {criterion}")
  print(f"KL loss scale: {beta if beta_scheduler_opts is None else beta_scheduler_opts}")

  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  print(f"Using device: {device}")

  if load_checkpoint:
    model.load_state_dict(torch.load(load_checkpoint), strict=False)
  model.to(device)

  optimizer_cls = eval(optimizer_opts.pop("cls"))
  optimizer = optimizer_cls(model.parameters(), **optimizer_opts)

  dataset_name = dataset_name.upper()
  assert dataset_name in VALID_DATASET_NAMES, f"Invalid dataset name: {dataset_name}"
  hdf_file_path = eval(f"{dataset_name}_MONOPHONIC_HDF_FILE_PATH")

  train_loader, devtrain_loader, dev_loader = get_data_loaders_for_training(
    hdf_file_path=hdf_file_path,
    batch_size=batch_size,
    dataset_fraction=dataset_fraction,
    dataset_cls=PlayedNotesDataset,
  )

  if lr_scheduling_opts:
    lr_scheduling_opts = copy.deepcopy(lr_scheduling_opts)
    scheduler_cls = eval(lr_scheduling_opts.pop("cls"))
    min_lr = lr_scheduling_opts.pop("min_lr", None)
    scheduler = scheduler_cls(optimizer, **lr_scheduling_opts)
  else:
    scheduler = None
    min_lr = None

  if beta_scheduler_opts:
    beta_scheduler_opts = copy.deepcopy(beta_scheduler_opts)
    beta_scheduler_cls = eval(beta_scheduler_opts.pop("cls"))
    assert beta_scheduler_cls == CyclicAnnealingScheduler, "Only CyclicAnnealingScheduler is supported"
    beta_scheduler = beta_scheduler_cls(**beta_scheduler_opts)
    beta_scheduler.num_steps = len(train_loader) * n_epochs
  else:
    beta_scheduler = None
  betas = []

  # Initialize TensorBoard writer
  writer = SummaryWriter(log_dir=f"./runs/{alias}")

  best_dev_epoch = 1

  for epoch in range(1, n_epochs + 1):
    training_stats[epoch] = {"losses": {}}

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

      running_rec_loss = 0.0
      running_kl_loss = 0.0
      num_batches = len(data_loader)
      for batch_idx, (played_notes, seq_lens, _) in enumerate(data_loader, start=1):
        if beta_scheduler:
          if train_mode:
            beta = beta_scheduler(batch_idx + (epoch - 1) * num_batches)
            betas.append(beta)
          else:
            beta = 1.0

        played_notes = played_notes.to(device).transpose(0, 1)  # [seq_len, batch_size]
        with torch.set_grad_enabled(train_mode):
          outputs, loss, recon_loss, kl_loss = calc_model_output_and_loss(
            played_notes,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            train_mode=train_mode,
            seq_lens=seq_lens,
            beta=beta,
            min_lr=min_lr,
            beta_scheduler=beta_scheduler,
          )
          running_kl_loss += kl_loss.item()
          running_rec_loss += recon_loss.item()

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
          f"Epoch {epoch}/{n_epochs}, Batch {batch_idx} - "
          f"{data_alias} Total loss: {loss.item():.4f}, "
          f"Recon loss: {recon_loss.item():.4f}, "
          f"KL loss: {kl_loss.item():.4f} - KL weight: {beta:.4f}"
        )

      avg_kl_loss = running_kl_loss / num_batches
      avg_rec_loss = running_rec_loss / num_batches
      avg_total_loss = avg_rec_loss + avg_kl_loss
      writer.add_scalar(f"{data_alias}_kl-loss/epoch", avg_kl_loss, epoch)
      writer.add_scalar(f"{data_alias}_recon-loss/epoch", avg_rec_loss, epoch)
      writer.add_scalar(f"{data_alias}_total-loss/epoch", avg_total_loss, epoch)
      training_stats[epoch]["losses"][data_alias] = {
        "total": avg_total_loss,
        "recon": avg_rec_loss,
        "kl": avg_kl_loss,
      }

      if scheduler and data_alias == "train":
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

    # Save model checkpoint
    torch.save(model.state_dict(), f"{checkpoint_path}/model_epoch_{epoch}.pt")

    # remove all checkpoints except for the best and last one
    dev_losses = list(map(lambda x: x["losses"]["dev"]["total"], training_stats.values()))
    best_dev_epoch = np.argmin(dev_losses) + 1
    for epoch_file in os.listdir(checkpoint_path):
      assert epoch_file.startswith("model_epoch_")
      model_epoch = int(epoch_file.split("_")[-1].split(".")[0])
      if model_epoch not in [best_dev_epoch, epoch]:
        os.remove(f"{checkpoint_path}/{epoch_file}")

    training_stats[epoch]["last_lr"] = scheduler.get_last_lr()[0] if scheduler else lr

  print(f"Models saved to {checkpoint_path}")
  writer.close()

  with open(training_stats_file_path, "w") as f:
    json.dump(training_stats, f, indent=2)

  if len(betas) != 0:
    plt.plot(betas)
    plt.savefig(f"{checkpoint_path}/betas.png")
    plt.close()

  best_dev_checkpoint_path = f"{checkpoint_path}/model_epoch_{best_dev_epoch}.pt"
  # test(
  #   model=model,
  #   checkpoint_path=best_dev_checkpoint_path,
  #   alias=alias,
  #   criterion=criterion,
  #   dataset_name=dataset_name,
  # )


def test(
        model: VAEModel,
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
