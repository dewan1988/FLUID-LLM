CONFIG_PATH="configs/inference1.json"
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
accelerate launch --mixed_precision=bf16 src/inference.py --config_path=$CONFIG_PATH