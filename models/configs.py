from data_utils import NUM_NOTES, MAX_NUM_TIME_STEPS


transformer_v1_config = dict(
  model_opts=dict(
    input_dim=NUM_NOTES,
    embed_dim=256,
    ff_dim=512,
    num_heads=8,
    num_layers=4,
    max_time_steps=MAX_NUM_TIME_STEPS,
  ),
  train_opts=dict(
    batch_size=32,
    n_epochs=10,
    lr=0.001,
  )
)
