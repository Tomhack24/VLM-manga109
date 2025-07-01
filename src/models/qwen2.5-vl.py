import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info 
from peft import LoraConfig, get_peft_model
import torch.nn as nn
import json

video_paths = []

for i in range(2, 3):
    path = f"/home/yoshioka/workdir/VLM-manga109/src/data/Manga109_released_2023_12_07/images/AisazuNihaIrarenai/{i:03d}.jpg"
    video_paths.append(path)

print(f"Video paths: {video_paths}")
USE_QLORA = True


if USE_QLORA:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_type=torch.bfloat16
    )

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

if torch.cuda.is_available():

    device = "cuda:1"
else:
    device = "cpu"

print(f"Using device: {device}")


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

target_modules = get_all_linear_layer_names(model)

lora_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias="none",   # "none"は、LoRAのバイアスを使用しないことを意味します 
    target_modules=target_modules,
    task_type="CAUSAL_LM", #次のトークン予測タスク
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

