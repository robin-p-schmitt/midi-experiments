from torch.nn import BCEWithLogitsLoss

from data_utils import (
  prepare_data
)
from models.configs import transformer_v1_config, transformer_vae_v1_config
from models.transformer import TransformerDecoderModel, train
from models.transformer_vae import VAETransformerEncoder
from utils.dict_update import dict_update_deep

prepare_data(dataset_name="maestro", num_workers=6)
# prepare_data(dataset_name="lakh", num_workers=6)

configs = []

# configs.append(dict(
#   alias="lakh_transformer_v1_lr-1e-3_bce_5-epochs_batch-size-256",
#   model_opts=transformer_v1_config['model_opts'],
#   train_opts=dict_update_deep(
#     transformer_v1_config['train_opts'],
#     {
#       "lr": 1e-3,
#       "criterion": BCEWithLogitsLoss(),
#       "n_epochs": 10,
#       "batch_size": 512,
#       "dataset_name": "lakh",
#       "dataset_fraction": 1.0,
#     },
#   )
# ))

configs.append(dict(
  alias="maestro_transformer_vae_v1_lr-1e-3_20-epochs_batch-size-128",
  model_opts=transformer_vae_v1_config['model_opts'],
  train_opts=dict_update_deep(
    transformer_vae_v1_config['train_opts'],
    {
      "lr": 1e-3,
      "n_epochs": 20,
      "batch_size": 128,
      "dataset_name": "maestro",
      "dataset_fraction": 1.0,
    },
  )
))

for config in configs:
  model_cls = config["model_opts"].pop("cls")
  model = model_cls(**config["model_opts"])

  train_func = config["train_opts"].pop("train_func")
  train_func(
    model,
    alias=config["alias"],
    **config["train_opts"],
  )
