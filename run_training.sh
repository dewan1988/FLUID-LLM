CONFIG_PATH="configs/training1.json"
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
accelerate launch --mixed_precision fp16 src/main.py --config_path=$CONFIG_PATH