import copy
from torch.nn import MSELoss, BCEWithLogitsLoss

from data_utils import NUM_NOTES, FRAMES_PER_BAR, NUM_BARS
from models.transformer import TransformerDecoderModel, train as train_transformer
from models.transformer_vae import VAETransformerEncoder, train as train_vae_transformer


transformer_v1_config = dict(
  model_opts=dict(
    cls=TransformerDecoderModel,
    input_dim=NUM_NOTES,
    embed_dim=256,
    num_heads=8,
    num_layers=4,
    max_time_steps=NUM_BARS * FRAMES_PER_BAR,
  ),
  train_opts=dict(
    train_func=train_transformer,
    batch_size=32,
    n_epochs=10,
    lr=0.001,
    criterion=BCEWithLogitsLoss(),
  )
)

transformer_vae_v1_config = dict(
  model_opts=dict(
    cls=VAETransformerEncoder,
    input_dim=NUM_NOTES + 2,  # add 2 for the silent idx and the special hidden representation index
    embed_dim=256,
    num_heads=8,
    num_layers=4,
    max_time_steps=NUM_BARS * FRAMES_PER_BAR + 1,  # add 1 for the special hidden representation index
  ),
  train_opts=dict(
    train_func=train_vae_transformer,
    batch_size=32,
    n_epochs=10,
    lr=0.001,
    criterion=BCEWithLogitsLoss(),
  )
)
