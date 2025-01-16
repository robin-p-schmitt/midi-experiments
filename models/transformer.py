import torch
from torch import nn
import torch.nn.functional as F  # noqa
import numpy as np

from dataset import get_data_loaders_for_training
from data_utils import HDF_FILE_PATH


class TransformerDecoderModel(nn.Module):
  def __init__(self, input_dim, embed_dim, ff_dim, num_heads, num_layers, max_time_steps):
    super(TransformerDecoderModel, self).__init__()
    self.embedding = nn.Linear(input_dim, embed_dim)
    self.positional_encoding = nn.Parameter(torch.zeros(1, max_time_steps, embed_dim))

    decoder_layer = nn.TransformerDecoderLayer(
        d_model=embed_dim,
        nhead=num_heads,
        dim_feedforward=ff_dim,
        batch_first=True
    )
    self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    self.output_layer = nn.Linear(embed_dim, input_dim)

  def forward(self, x):
    x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
    x = self.transformer_decoder(
        x,
        memory=x,
        tgt_mask=nn.Transformer.generate_square_subsequent_mask(x.size(1)),
        tgt_is_causal=True,
    )
    return self.output_layer(x)


def calc_model_output_and_loss(
        piano_rolls: torch.Tensor,
        model: TransformerDecoderModel,
        criterion: nn.MSELoss,
        optimizer: torch.optim.Optimizer,
        train_mode: bool,
):
  piano_rolls_shifted = F.pad(piano_rolls, (0, 0, 1, 0))[:, :-1]
  outputs = model(piano_rolls_shifted)
  loss = criterion(outputs, piano_rolls)  # Example target as input

  if train_mode:
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  return outputs, loss


def train(
        model: TransformerDecoderModel,
        n_epochs: int,
        lr: float,
        batch_size: int,
        alias: str,
):
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  model.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  criterion = nn.MSELoss()  # Assuming regression task

  train_loader, devtrain_loader, dev_loader = get_data_loaders_for_training(
    hdf5_file_path=HDF_FILE_PATH,
    batch_size=batch_size,
  )

  for epoch in range(n_epochs):
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
      for batch_idx, piano_rolls in enumerate(data_loader):
        piano_rolls = piano_rolls.to(device)
        with torch.set_grad_enabled(train_mode):
          outputs, loss = calc_model_output_and_loss(
            piano_rolls,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_mode=train_mode,
          )
        print(f"Epoch {epoch + 1}/{n_epochs}, Batch {batch_idx + 1}, {data_alias} Loss: {loss.item():.4f}")

    # Save model checkpoint
    torch.save(model.state_dict(), f"{alias}.pt")
