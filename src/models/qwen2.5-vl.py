import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info # 仮定: このユーティリティは適切に定義されている
from peft import LoraConfig, get_peft_model

USE_QLORA = True
lora_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias="none",
    target_modules=[
        "visual.patch_embed.proj",
        "visual.blocks.0.attn.qkv",
        "visual.blocks.0.attn.proj",
        "visual.blocks.0.mlp.gate_proj",
        "visual.blocks.0.mlp.up_proj",
        "visual.blocks.0.mlp.down_proj",
        "visual.merger.mlp.0",
        "model.embed_tokens",
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
        "model.layers.0.mlp.down_proj",
        "lm_head",
    ],
    task_type="CAUSAL_LM",
)

if USE_QLORA:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_type=torch.bfloat16
    )

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# **変更点1: deviceの決定方法をより明確に**
# 複数GPU環境で特定のGPUを使用したい場合
# 例えば、常にcuda:0を使いたい場合は、直接 "cuda:0" を指定
# それ以外は、利用可能なCUDAデバイスがあれば "cuda" (デフォルトデバイス) を使う
if torch.cuda.is_available():
    # 複数GPUがある場合でも、明示的に1つのデバイスを指定する方がトラブルが少ない
    # `device_map="auto"` を使う場合は、後続で必ず `model.to(device)` を徹底
    # ここでは、もし複数GPUがあってもcuda:0を使うことを想定
    device = "cuda:1"
else:
    device = "cpu"

print(f"Using device: {device}")


model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype="auto", # この設定は維持
    # **変更点2: device_mapの扱いの見直し**
    # もしモデルが大きすぎて単一GPUに乗り切らない場合は "auto" が必要ですが、
    # そうでない場合、または単一GPUで動かしたい場合は明示的にNoneにするか、
    # 特定のデバイスを指定することで、自動分散による意図しない挙動を防げます。
    # ここでは、`model.to(device)` で明示的に移動することを前提に `device_map=None` を推奨
    # モデルが単一のGPUに収まる場合:
    device_map=None, # Noneに設定し、後でmodel.to(device)で手動で配置
    # モデルが単一のGPUに収まらない場合（ただし今回は異なるデバイスのエラーなので、収まる前提で対処）:
    # device_map="auto", # この場合でも、get_peft_model後にmodel.to(device)が重要
    quantization_config=bnb_config if USE_QLORA else None
)

model = get_peft_model(model, lora_config)

# **変更点3: モデルを明示的に指定したデバイスに移動**
# `device_map=None` の場合、モデルはCPUにロードされるため、ここでGPUに移動します。
# `device_map="auto"` の場合でも、PEFTモデルがラップされた後、
# 念のためすべての層がターゲットデバイスにあることを保証するためにこの行は重要です。
model.to(device)

model.print_trainable_parameters()

processor = AutoProcessor.from_pretrained(MODEL_ID)
messages = [
    {"role": "user", "content": [
        {"type": "video", "video": ["src/data/Manga109_released_2023_12_07/images/AisazuNihaIrarenai/002.jpg", "src/data/Manga109_released_2023_12_07/images/AisazuNihaIrarenai/003.jpg"],},
        {"type": "text", "text": "Describe this animation."},
    ]},
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
# **変更点4: process_vision_infoから返されるテンソルのデバイス確認と移動**
# `process_vision_info` 関数が内部でテンソルを生成する場合、
# それらがデフォルトでCPUに生成されるか、またはシステム内の別のGPUに生成される可能性があります。
# そのため、明示的にターゲットデバイスに移動させます。
image_inputs, video_inputs = process_vision_info(messages)

# image_inputsとvideo_inputsがテンソルのリストや辞書である場合を考慮
# 実際には、processorに渡す前にこれらのテンソルが正しいデバイスにあることを確認する必要があります
# ここでは、processorに渡す際に`.to(device)`を適用するようになっていますが、
# もし`process_vision_info`が既にテンソルを返している場合は、そこで `.to(device)` を適用する必要があります。
# しかし、QwenのProcessorはパスを受け取るので、ここでは変更なしで大丈夫なはずです。

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
    torch_dtype=torch.bfloat16
)       

# **変更点5: 入力テンソルを明示的に指定したデバイスに移動**
# ここが最も重要です。processorが生成したすべてのテンソルを、モデルと同じデバイスに移動させます。
inputs = {k: v.to(device) for k, v in inputs.items()}

# **変更点6: モデルの推論モード設定**
# 推論時には `model.eval()` を呼び出すことで、ドロップアウトなどが無効になり、
# 安定した動作と省メモリ化が期待できます。
model.eval() 

# QLoRAの場合、`.to(device)`の後に`model.generate`を呼び出す前に、
# QuantizationConfigが正しく適用されているか最終確認
# これは情報提供のためですが、デバッグに役立ちます
# for name, param in model.named_parameters():
#     if hasattr(param, 'quant_state'):
#         print(f"Layer {name} quant_state: {param.quant_state}")


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
