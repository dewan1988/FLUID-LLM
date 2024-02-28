"""
Testing
"""

import torch

from utils import set_seed, get_huggingface_model
from models import MultivariateTimeLLM

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

    model = MultivariateTimeLLM(training_params)




