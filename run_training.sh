CONFIG_PATH="configs/training1.yaml"
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
# Down here, mixed_precision can be [bf16, fp16, fp32, no]
# --use_deepspeed
accelerate launch --mixed_precision=bf16 --gradient_accumulation_steps=4 src/main.py --config_path=$CONFIG_PATH