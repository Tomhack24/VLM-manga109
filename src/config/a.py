from config_loader import load_config, get_model_config, get_bnb_config, get_lora_config

config = load_config()
bnb_config = get_bnb_config(config)
print(f"Loaded BitsAndBytesConfig: {bnb_config}")