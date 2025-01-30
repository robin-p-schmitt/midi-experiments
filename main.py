import copy

from torch.nn import BCEWithLogitsLoss

from data_utils import (
  prepare_data
)
from models.configs import transformer_v1_config, transformer_vae_v1_config, transformer_vae_v2_config
from models.transformer import TransformerDecoderModel, train
from models.transformer_vae import VAETransformerEncoder
from utils.dict_update import dict_update_deep

prepare_data(dataset_name="maestro", num_workers=6)
# prepare_data(dataset_name="lakh", num_workers=6)

configs = []


# "dev": {
#   "total": 3.5579125839285553,
#   "recon": 3.5474765300750732,
#   "kl": 0.010436053853482008
# },
configs.append(dict(
  alias="maestro_transformer_vae_v1_50-epochs_0.1-data-fraction",
  model_opts=transformer_vae_v1_config['model_opts'],
  train_opts=dict_update_deep(
    transformer_vae_v1_config['train_opts'],
    {
      "n_epochs": 50,
      "batch_size": 256,
      "dataset_name": "maestro",
      "dataset_fraction": 0.1,
      "kl_loss_scale": 1.0,
    },
  )
))

# "dev": {
#   "total": 3.5121482904069126,
#   "recon": 3.509843945503235,
#   "kl": 0.002304344903677702
# },
configs.append(dict(
  alias="maestro_transformer_vae_v1_50-epochs_0.1-data-fraction_wd-1e-6",
  model_opts=transformer_vae_v1_config['model_opts'],
  train_opts=dict_update_deep(
    transformer_vae_v1_config['train_opts'],
    {
      "n_epochs": 50,
      "batch_size": 256,
      "dataset_name": "maestro",
      "dataset_fraction": 0.1,
      "kl_loss_scale": 1.0,
      "optimizer_opts.weight_decay": 1e-6,
    },
  )
))

# "dev": {
#   "total": 3.072379293269478,
#   "recon": 3.0698957443237305,
#   "kl": 0.002483548945747316
# },
configs.append(dict(
  alias="maestro_transformer_vae_v1_50-epochs_0.2-data-fraction",
  model_opts=transformer_vae_v1_config['model_opts'],
  train_opts=dict_update_deep(
    transformer_vae_v1_config['train_opts'],
    {
      "n_epochs": 50,
      "batch_size": 256,
      "dataset_name": "maestro",
      "dataset_fraction": 0.2,
      "kl_loss_scale": 1.0,
    },
  )
))

configs.append(dict(
  alias="maestro_transformer_vae_v2_50-epochs_0.1-data-fraction",
  model_opts=transformer_vae_v2_config['model_opts'],
  train_opts=dict_update_deep(
    transformer_vae_v2_config['train_opts'],
    {
      "n_epochs": 50,
      "batch_size": 256,
      "dataset_name": "maestro",
      "dataset_fraction": 0.1,
      "kl_loss_scale": 1.0,
    },
  )
))

for config in configs:
  config = copy.deepcopy(config)
  model_cls = config["model_opts"].pop("cls")
  model = model_cls(**config["model_opts"])

  train_func = config["train_opts"].pop("train_func")
  train_func(
    model,
    alias=config["alias"],
    **config["train_opts"],
  )
