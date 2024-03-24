CONFIG_PATH="configs/inference1.json"
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python src/inference.py --config_path=$CONFIG_PATH