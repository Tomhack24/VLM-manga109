import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info 
from peft import LoraConfig, get_peft_model
import torch.nn as nn
import json
from ..src.config.config_loader import load_config, get_model_config, get_bnb_config, get_lora_config

# 設定を読み込み
config = load_config()
model_config = get_model_config(config)
bnb_config_dict = get_bnb_config(config)
lora_config_dict = get_lora_config(config)

video_paths = []

for i in range(2, 3):
    path = f"/home/yoshioka/workdir/VLM-manga109/src/data/Manga109_released_2023_12_07/images/AisazuNihaIrarenai/{i:03d}.jpg"
    video_paths.append(path)

print(f"Video paths: {video_paths}")

# 設定からの値を使用
USE_QLORA = model_config.get("use_qlora", True)
MODEL_ID = model_config.get("id", "Qwen/Qwen2.5-VL-3B-Instruct")
device = model_config.get("device", "cuda:1" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
print(f"Model ID: {MODEL_ID}")
print(f"Use QLoRA: {USE_QLORA}")

if USE_QLORA:
    # YAMLから設定を読み込んでBitsAndBytesConfigを作成
    bnb_config = BitsAndBytesConfig(**bnb_config_dict)


model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype="auto", 
    device_map=device,
    quantization_config=bnb_config if USE_QLORA else None
)

#モデルが量子化できているのか確認
model_config_dict = model.config.to_dict()


def get_all_linear_layer_names(model):
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            target_modules.append(name)
    return target_modules

# target_modulesの決定：設定で指定されている場合はそれを使用、そうでなければ全Linearレイヤーを取得
if lora_config_dict.get("target_modules"):
    target_modules = lora_config_dict["target_modules"]
    print(f"Using configured target modules: {target_modules}")
else:
    target_modules = get_all_linear_layer_names(model)
    print(f"Using all linear layers as target modules: {len(target_modules)} modules found")

# YAMLから設定を読み込んでLoraConfigを作成
lora_config = LoraConfig(
    lora_alpha=lora_config_dict.get("lora_alpha", 16),
    lora_dropout=lora_config_dict.get("lora_dropout", 0.05),
    r=lora_config_dict.get("r", 8),
    bias=lora_config_dict.get("bias", "none"),
    target_modules=target_modules,
    task_type=lora_config_dict.get("task_type", "CAUSAL_LM"),
)


model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

processor = AutoProcessor.from_pretrained(MODEL_ID)
messages = [
    {"role": "user", "content": [
        {"type": "video", "video": video_paths},
        {"type": "text", "text": "男は炊飯器のコンセントを，上と下のどちらに挿したましたか？"},
    ]},
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)


inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
)       

inputs = {k: v.to(device) for k, v in inputs.items()}

model.eval() 


with torch.no_grad(): # 推論時は勾配計算を無効化し、メモリ使用量を削減
    generated_ids = model.generate(**inputs, max_new_tokens=128)

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print("======Output Text:=======")
print(output_text)

