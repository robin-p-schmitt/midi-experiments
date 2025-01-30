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

# configs.append(dict(
#   alias="maestro_transformer_vae_v1_lr-1e-3_20-epochs_batch-size-128",
#   model_opts=transformer_vae_v1_config['model_opts'],
#   train_opts=dict_update_deep(
#     transformer_vae_v1_config['train_opts'],
#     {
#       "lr": 1e-3,
#       "n_epochs": 20,
#       "batch_size": 128,
#       "dataset_name": "maestro",
#       "dataset_fraction": 1.0,
#       "kl_loss_scale": 0.005,
#     },
#   )
# ))

# for single training sample: train Total loss: 0.0925, Recon loss: 0.0059, KL loss: 0.0866
configs.append(dict(
  alias="maestro_transformer_vae_v1_lr-5e-4_oclr_max-lr-5e-3_20-epochs_batch-size-256_kl-scale-0.01",
  model_opts=transformer_vae_v1_config['model_opts'],
  train_opts=dict_update_deep(
    transformer_vae_v1_config['train_opts'],
    {
      "lr": 1e-4,
      "lr_scheduling": "oclr",
      "max_lr": 5e-4,
      "n_epochs": 200,
      "batch_size": 256,
      "dataset_name": "maestro",
      "dataset_fraction": 1.0,
      "kl_loss_scale": 0.05,
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
