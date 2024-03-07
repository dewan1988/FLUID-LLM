"""
Testing
"""

import sys
import argparse
import logging
from torch.utils.data import DataLoader

from utils import set_seed, load_params_from_file
from data_utils import generate_dummy_ts_dataset
from models.model import MultivariateTimeLLM

logging.basicConfig(level=logging.INFO,
                    format=f'[{__name__}:%(levelname)s] %(message)s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path',
                        default="configs/training1.json",
                        required=True,
                        help='Path to the json config for training')

    args = parser.parse_args(sys.argv[1:])

    set_seed()
    training_params = load_params_from_file(args.config_path)
    logging.info(f"Parameters for training: {training_params}")

    # Test dummy data
    B, T, N, M, PATCH_DIM = 4, 64, 10, 5, 16
    dummy_univariate_dataset = generate_dummy_ts_dataset(multivariate=False,
                                                         batch_size=B,
                                                         horizon=T,
                                                         n=N,
                                                         in_dim=PATCH_DIM)

    print(f'Dummy univariate dataset shape: {dummy_univariate_dataset.shape}')

    dummy_multivariate_dataset = generate_dummy_ts_dataset(multivariate=True,
                                                           batch_size=B,
                                                           horizon=T,
                                                           n=N,
                                                           m=M,
                                                           in_dim=PATCH_DIM)

    print(f'Dummy multivariate dataset shape: {dummy_multivariate_dataset.shape}')

    # Test model forward pass
    model = MultivariateTimeLLM(training_params, N=N, M=M, patch_dim=PATCH_DIM)
    model_output = model(dummy_multivariate_dataset).last_hidden_state
    print("Out")
    print(model_output.shape)
