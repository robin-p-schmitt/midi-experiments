import copy
from torch.nn import MSELoss, BCEWithLogitsLoss

from data_utils import NUM_NOTES, FRAMES_PER_BAR, NUM_BARS


transformer_v1_config = dict(
  model_opts=dict(
    input_dim=NUM_NOTES,
    embed_dim=256,
    num_heads=8,
    num_layers=4,
    max_time_steps=NUM_BARS * FRAMES_PER_BAR,
  ),
  train_opts=dict(
    batch_size=32,
    n_epochs=10,
    lr=0.001,
    criterion=BCEWithLogitsLoss(),
  )
)
