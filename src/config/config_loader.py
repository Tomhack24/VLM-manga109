import yaml
import os
import torch
from pathlib import Path

def load_config(config_path="./model_config.yaml"):

    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config

def get_model_config(config):
    """モデル設定を取得"""
    return config.get("model", {})

def get_bnb_config(config):
    """BitsAndBytesConfig用の設定を取得し、torch型を変換"""
    bnb_config = config.get("bnb", {}).copy()
    
    # torch.bfloat16などの文字列を実際のオブジェクトに変換
    if "bnb_4bit_compute_type" in bnb_config:
        compute_type_str = bnb_config["bnb_4bit_compute_type"]
        if compute_type_str == "torch.bfloat16":
            bnb_config["bnb_4bit_compute_type"] = torch.bfloat16
        elif compute_type_str == "torch.float16":
            bnb_config["bnb_4bit_compute_type"] = torch.float16
    
    return bnb_config

def get_lora_config(config):
    """LoRA設定を取得"""
    return config.get("lora", {})