from data_utils import download_dataset, unpack_dataset, build_hdf_file
from models.configs import transformer_v1_config
from models.transformer import TransformerDecoderModel, train

download_dataset()
unpack_dataset()
build_hdf_file()

model = TransformerDecoderModel(**transformer_v1_config['model_opts'])
train(
  model,
  alias='transformer_v1',
  **transformer_v1_config['train_opts'],
)
