from data_utils import (
  download_dataset, unpack_dataset, build_hdf_file, build_hdf_file_multi, HDF_FILE_PATH, DATA_UNPACKED_PATH
)
from models.configs import transformer_v1_config
from models.transformer import TransformerDecoderModel, train, boosted_mse
from utils.dict_update import dict_update_deep

download_dataset()
unpack_dataset()
# build hdf with single core
# build_hdf_file(
#   hdf_file_path=HDF_FILE_PATH,
#   subdir=DATA_UNPACKED_PATH,
#   queue=None,
# )

build_hdf_file_multi(num_cores=6)

# model = TransformerDecoderModel(**transformer_v1_config['model_opts'])
# train(
#   model,
#   alias='transformer_v1',
#   **transformer_v1_config['train_opts'],
# )
#
# train(
#   model,
#   alias='transformer_v1_boosted-mse',
#   **dict_update_deep(
#     transformer_v1_config['train_opts'],
#     {"criterion": boosted_mse},
#   ),
# )
