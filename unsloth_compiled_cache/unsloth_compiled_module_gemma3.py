"""
2025.6.6
2025.6.8
4.53.0
0.19.0
__UNSLOTH_VERSIONING__
"""

# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import importlib.util
if importlib.util.find_spec("unsloth_studio") is None:
    UNSLOTH_STUDIO_ENABLED = False
else:
    UNSLOTH_STUDIO_ENABLED = os.environ.get("UNSLOTH_STUDIO_DISABLED", "0") == "0"
pass
from typing import List, Dict, Tuple, Optional, Any, Callable
import math


import os
import torch
from unsloth_zoo.loss_utils import (
    fused_linear_cross_entropy,
    compiled_ce_loss_function,
)

if UNSLOTH_STUDIO_ENABLED:
    from unsloth_zoo.loss_utils import fast_linear_cross_entropy

scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention
@torch.compiler.disable(recursive = False)
def disable_compile_scaled_dot_product_attention(*args, **kwargs):
    return scaled_dot_product_attention(*args, **kwargs)
pass


from transformers.modeling_flash_attention_utils import is_flash_attn_available

if is_flash_attn_available():
    try:
        from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask
    except:
        flash_attn_supports_top_left_mask = None
    try:
        from transformers.modeling_flash_attention_utils import _flash_attention_forward
    except:
        _flash_attention_forward = None
    try:
        from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
    except:
        FlashAttentionKwargs = None
    try:
        from transformers.modeling_flash_attention_utils import flash_attn_varlen_func
    except:
        flash_attn_varlen_func = None
else:
    flash_attn_supports_top_left_mask = None
    _flash_attention_forward = None
    FlashAttentionKwargs = None
    flash_attn_varlen_func = None
pass


torch_compile_options = {'epilogue_fusion': True, 'max_autotune': False, 'shape_padding': True, 'trace.enabled': False, 'triton.cudagraphs': False, 'debug': False, 'dce': True, 'memory_planning': True, 'coordinate_descent_tuning': False, 'trace.graph_diagram': False, 'compile_threads': 24, 'combo_kernels': False, 'group_fusion': True, 'disable_progress': True, 'verbose_progress': False, 'triton.multi_kernel': False, 'triton.use_block_ptr': False, 'triton.enable_persistent_tma_matmul': True, 'triton.autotune_at_compile_time': True}

from torch.nn import CrossEntropyLoss

@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def normal_cross_entropy_loss(self, hidden_states, labels):
    logits = self.lm_head(hidden_states)
    logits = logits.float()
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, self.config.vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    return loss, logits
pass

# We need an empty logits flag to warn people logits will not be returned anymore unless asked ie
# os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
LOGITS_ERROR_STRING = \
    "Unsloth: Logits are empty from 2024.11 onwards. To get raw logits again, please "\
    'set the environment variable `UNSLOTH_RETURN_LOGITS` to `"1" BEFORE starting to train ie before `trainer.train()`. For example:\n'\
    "```\nimport os\n"\
    "os.environ['UNSLOTH_RETURN_LOGITS'] = '1'\n"\
    "trainer.train()\n```\n"\
    "No need to restart your console - just add `os.environ['UNSLOTH_RETURN_LOGITS'] = '1'` before trainer.train() and re-run the cell!"

def raise_logits_error(*args, **kwargs): raise NotImplementedError(LOGITS_ERROR_STRING)
def return_none(*args, **kwargs): return None
class EmptyLogits:
    def __init__(self): return
    def raise_getattr_error(self, attr): return return_none if attr == "to" else raise_logits_error
    __getitem__ = raise_logits_error
    __getattr__ = raise_getattr_error
    def __repr__(self): return LOGITS_ERROR_STRING
    def __str__ (self): return LOGITS_ERROR_STRING
pass
EMPTY_LOGITS = EmptyLogits()
functions = dir(torch.Tensor)
for j, function in enumerate(functions):
    if function.startswith("__") and function.endswith("__"):
        exec(f"def raise_{j}(*args, **kwargs): print('{function}')", globals(), locals())
        try: exec(f"EMPTY_LOGITS.{function} = raise_{j}", globals(), locals())
        except: continue
pass


def mask_attention_mask_out(labels = None, attention_mask = None):
    if labels is not None and attention_mask is not None:
        attention_mask = attention_mask.to(device = labels.device)
        labels[attention_mask == 0] = -100
    return labels
pass


from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.models.gemma3.modeling_gemma3 import (Callable, Optional, Union, torch, nn, ACT2FN, Cache, PretrainedConfig, GenerationMixin, create_causal_mask, create_sliding_window_causal_mask, BaseModelOutputWithPast, ModelOutput, CausalLMOutputWithPast, ROPE_INIT_FUNCTIONS, dynamic_rope_update, PreTrainedModel, can_return_tuple, AutoModel, Gemma3Config, Gemma3TextConfig, logger, __name__, Gemma3Model, Gemma3CausalLMOutputWithPast, Gemma3PreTrainedModel, Gemma3TextModel, Gemma3ForCausalLM, Gemma3ForConditionalGeneration)

@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def create_masks_for_generate(
    config: PretrainedConfig,
    input_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values: Optional[Cache],
    or_mask_function: Optional[Callable] = None,
    and_mask_function: Optional[Callable] = None,
    **kwargs,
):
    """
    This function mimics how we create the masks in the `modeling_xxx.py` files, and is used in `generate` in order
    to easily create the masks in advance, when we compile the forwards with Static caches.

    Args:
        config (`PretrainedConfig`):
            The model config.
        input_embeds (`torch.Tensor`):
            The input embeddings of shape (batch_size, query_length, hidden_dim). This is used only to infer the
            batch size, query length and dtype.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length).
            It can also be an already prepared 4D mask, in which case it is returned as-is.
        cache_position (`torch.Tensor`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        past_key_values (`Cache`, optional):
            The past key values, if we use a cache.
        or_mask_function (`Callable`, optional):
            An optional mask function to combine with the other mask function (by doing the union of both). This is
            useful to easily overlay another mask on top of the causal one, for example for image tokens handling.
        and_mask_function (`Callable`, optional):
            An optional mask function to combine with the other mask function (by doing the intersection of both). This is
            useful to easily overlay another mask on top of the causal one, for example for image tokens handling.
    """
    # The attribute reside in the text config for composite models
    effective_config = config.get_text_config()
    # Prepare the mask args
    mask_kwargs = {
        "config": effective_config,
        "input_embeds": input_embeds,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "past_key_values": past_key_values,
        "or_mask_function": or_mask_function,
        "and_mask_function": and_mask_function,
    }

    # If the attribute exist, we need several masks
    if hasattr(effective_config, "layer_types"):
        causal_masks = {}
        for layer_pattern in set(effective_config.layer_types):
            causal_masks[layer_pattern] = LAYER_PATTERN_TO_MASK_FUNCTION_MAPPING[layer_pattern](**mask_kwargs)
        return causal_masks
    # In this case, all layers are sliding
    elif getattr(effective_config, "sliding_window", None) is not None:
        return create_sliding_window_causal_mask(**mask_kwargs)
    # In this case, all layers are chunked
    elif getattr(effective_config, "attention_chunk_size", None) is not None:
        return create_chunked_causal_mask(**mask_kwargs)
    # All layers use standard causal attention
    return create_causal_mask(**mask_kwargs)


@torch.compile(fullgraph = False, dynamic = True, options = torch_compile_options)
def Gemma3MLP_forward(self, x):
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj

class Gemma3MLP(nn.Module):
    def __init__(self, config: Gemma3TextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, x):
        return Gemma3MLP_forward(self, x)


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def Gemma3RMSNorm_forward(self, x):
    output = self._norm(x.float())
    # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
    # See https://github.com/huggingface/transformers/pull/29402
    output = output * (1.0 + self.weight.float())
    return output.type_as(x)

class Gemma3RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return Gemma3RMSNorm_forward(self, x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
@torch.no_grad()
@dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
def Gemma3RotaryEmbedding_forward(self, x, position_ids):
    inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
    position_ids_expanded = position_ids[:, None, :].float()

    device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):  # Force float32
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class Gemma3RotaryEmbedding(nn.Module):
    def __init__(self, config: Gemma3TextConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq


    def forward(self, x, position_ids):
        return Gemma3RotaryEmbedding_forward(self, x, position_ids)


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def apply_rotary_pos_emb(q, k, cos, sin,  unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scaling is None:
        scaling = module.head_dim**-0.5

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap
    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def Gemma3MultiModalProjector_forward(self, vision_outputs: torch.Tensor):
    batch_size, _, seq_length = vision_outputs.shape

    reshaped_vision_outputs = vision_outputs.transpose(1, 2)
    reshaped_vision_outputs = reshaped_vision_outputs.reshape(
        batch_size, seq_length, self.patches_per_image, self.patches_per_image
    )
    reshaped_vision_outputs = reshaped_vision_outputs.contiguous()

    pooled_vision_outputs = self.avg_pool(reshaped_vision_outputs)
    pooled_vision_outputs = pooled_vision_outputs.flatten(2)
    pooled_vision_outputs = pooled_vision_outputs.transpose(1, 2)

    normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs)

    projected_vision_outputs = torch.matmul(normed_vision_outputs, self.mm_input_projection_weight)
    return projected_vision_outputs.type_as(vision_outputs)

class Gemma3MultiModalProjector(nn.Module):
    def __init__(self, config: Gemma3Config):
        super().__init__()

        self.mm_input_projection_weight = nn.Parameter(
            torch.zeros(config.vision_config.hidden_size, config.text_config.hidden_size)
        )

        self.mm_soft_emb_norm = Gemma3RMSNorm(
            config.vision_config.hidden_size, eps=config.vision_config.layer_norm_eps
        )

        self.patches_per_image = int(config.vision_config.image_size // config.vision_config.patch_size)
        self.tokens_per_side = int(config.mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.avg_pool = nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.kernel_size)

    def forward(self, vision_outputs: torch.Tensor):
        return Gemma3MultiModalProjector_forward(self, vision_outputs)


@torch.compiler.disable(recursive = False)
@can_return_tuple
def Gemma3ForCausalLM_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **loss_kwargs,
) -> CausalLMOutputWithPast:
    r"""
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
        config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
        (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Example:

    ```python
    >>> from transformers import AutoTokenizer, Gemma3ForCausalLM

    >>> model = Gemma3ForCausalLM.from_pretrained("google/gemma-2-9b")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")

    >>> prompt = "What is your favorite condiment?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "What is your favorite condiment?"
    ```"""

    if self.training and self.config._attn_implementation != "eager":
        logger.warning_once(
            "It is strongly recommended to train Gemma3 models with the `eager` attention implementation "
            f"instead of `{self.config._attn_implementation}`. Use `eager` with `AutoModelForCausalLM.from_pretrained('<path-to-checkpoint>', attn_implementation='eager')`."
        )
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs: BaseModelOutputWithPast = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        cache_position=cache_position,
        **loss_kwargs,
    )

    hidden_states = outputs.last_hidden_state
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = EMPTY_LOGITS
    loss = None
    NOT_RETURN_LOGITS = os.environ.get('UNSLOTH_RETURN_LOGITS', '0') == '0'
    RETURN_HIDDEN_STATES = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1"
    
    n_items = None
    if (loss_kwargs) != () and type(loss_kwargs) is dict:
        n_items = (loss_kwargs).get("num_items_in_batch", None) or (loss_kwargs).get("n_items", None)
    if n_items is None:
        all_locals = locals()
        if 'loss_kwargs' in all_locals:
            __kwargs = all_locals['loss_kwargs']
            if type(__kwargs) is dict:
                n_items = __kwargs.get("num_items_in_batch", None) or __kwargs.get("n_items", None)
        if n_items is None and 'kwargs' in all_locals:
            __kwargs = all_locals['kwargs']
            if type(__kwargs) is dict:
                n_items = __kwargs.get("num_items_in_batch", None) or __kwargs.get("n_items", None)
        if n_items is None:
            all_locals = all_locals.values()
            for __kwargs in all_locals:
                if type(__kwargs) is dict:
                    n_items = __kwargs.get("num_items_in_batch", None) or __kwargs.get("n_items", None)
                    break
    pass
    
    requires_grad_ = self.lm_head.weight.requires_grad
    requires_grad_ = requires_grad_ or self.lm_head.weight.dtype == torch.float32
    
    if RETURN_HIDDEN_STATES:
        logits = hidden_states[:, slice_indices, :]
    elif labels is None:
        logits = self.lm_head(hidden_states[:, slice_indices, :])
    elif (UNSLOTH_STUDIO_ENABLED and NOT_RETURN_LOGITS and labels is not None) and not requires_grad_:
        loss = fast_linear_cross_entropy(
            hidden_states        = hidden_states[:, slice_indices, :],
            lm_head              = self.lm_head,
            labels               = labels,
            num_items_in_batch   = n_items,
            logit_softcapping    = None if (self.config.final_logit_softcapping) == () else (self.config.final_logit_softcapping),
            logit_scale_multiply = None if () == () else (),
            logit_scale_divide   = None if () == () else (),
        )
    elif (() == () and () == ()) and NOT_RETURN_LOGITS and self.loss_function.__name__.endswith("ForCausalLMLoss") and labels is not None and not requires_grad_:
        loss = fused_linear_cross_entropy(
            hidden_states      = hidden_states[:, slice_indices, :],
            lm_weight          = self.lm_head.weight,
            labels             = labels.to(self.lm_head.weight.device),
            num_items_in_batch = n_items,
            logit_softcapping  = None if (self.config.final_logit_softcapping) == () else (self.config.final_logit_softcapping),
        )
    elif self.loss_function.__name__.endswith("ForCausalLMLoss") and labels is not None:
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        torch._dynamo.mark_dynamic(logits, 1)
        torch._dynamo.mark_dynamic(labels, 1)
        loss = compiled_ce_loss_function(
            output_logits        = logits,
            output_labels        = labels,
            logit_scale_multiply = () if () != () else 0,
            logit_scale_divide   = () if () != () else 0,
            logit_softcapping    = (self.config.final_logit_softcapping) if (self.config.final_logit_softcapping) not in (None, (),) else 0,
            vocab_size           = (self.vocab_size),
            n_items              = n_items if n_items is not None else 0,
            requires_grad_       = requires_grad_,
        )
    else:
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if () != ():
            logits = logits * ()
        if () != ():
            logits = logits / ()
        if (self.config.final_logit_softcapping) not in (None, (),):
            logits = logits / (self.config.final_logit_softcapping)
            logits = torch.tanh(logits)
            logits = logits * (self.config.final_logit_softcapping)
        loss = self.loss_function(logits, labels.to(self.lm_head.weight.device), self.vocab_size, **loss_kwargs)


    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

class Gemma3ForCausalLM(Gemma3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    config_class = Gemma3TextConfig
    base_model_prefix = "language_model"

    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.model = Gemma3TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **loss_kwargs,
    ) -> CausalLMOutputWithPast:
        return Gemma3ForCausalLM_forward(self, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, cache_position, logits_to_keep, **loss_kwargs)


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def token_type_ids_mask_function(token_type_ids: Optional[torch.Tensor], tokens_per_image: int) -> Optional[Callable]:
    """
    This function adds the correct offsets to the `q_idx` and `kv_idx` as the torch API can only accept lengths,
    not start and end indices.
    """
    # Do not return an additional mask in this case
    if token_type_ids is None:
        return None

    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        # If the difference is less than image size, both are part of the same image block
        same_image_block = torch.abs(kv_idx - q_idx) <= tokens_per_image
        # If it's 1 for both query and key/value, we are in an image block
        is_image_block = (token_type_ids[batch_idx, q_idx] == 1) & (token_type_ids[batch_idx, kv_idx] == 1)

        # This is bidirectional attention whenever we are dealing with image tokens
        return is_image_block & same_image_block

    return inner_mask


@torch.compiler.disable(recursive = False)
def Gemma3ForConditionalGeneration_forward(
    self,
    input_ids: torch.LongTensor = None,
    pixel_values: torch.FloatTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[list[torch.FloatTensor], Cache]] = None,
    token_type_ids: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **lm_kwargs,
) -> Union[tuple, Gemma3CausalLMOutputWithPast]:
    r"""
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
        config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
        (masked), the loss is only computed for the tokens with labels in `[0, ..., config.text_config.vocab_size]`.

    Example:

    ```python
    >>> from PIL import Image
    >>> import requests
    >>> from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    >>> model = Gemma3ForConditionalGeneration.from_pretrained("google/gemma-3-4b-it")
    >>> processor = AutoProcessor.from_pretrained("google/gemma-3-4b-it")

    >>> messages = [
    ...     {
    ...         "role": "system",
    ...         "content": [
    ...             {"type": "text", "text": "You are a helpful assistant."}
    ...         ]
    ...     },
    ...     {
    ...         "role": "user", "content": [
    ...             {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
    ...             {"type": "text", "text": "Where is the cat standing?"},
    ...         ]
    ...     },
    ... ]

    >>> inputs = processor.apply_chat_template(
    ...     messages,
    ...     tokenize=True,
    ...     return_dict=True,
    ...     return_tensors="pt",
    ...     add_generation_prompt=True
    ... )
    >>> # Generate
    >>> generate_ids = model.generate(**inputs)
    >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "user\nYou are a helpful assistant.\n\n\n\n\n\nWhere is the cat standing?\nmodel\nBased on the image, the cat is standing in a snowy area, likely outdoors. It appears to"
    ```
    """

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        token_type_ids=token_type_ids,
        attention_mask=mask_attention_mask_out(labels = labels, attention_mask = attention_mask),
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        labels=labels,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        **lm_kwargs,
    )

    hidden_states = outputs[0]
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = EMPTY_LOGITS
    loss = None
    NOT_RETURN_LOGITS = os.environ.get('UNSLOTH_RETURN_LOGITS', '0') == '0'
    RETURN_HIDDEN_STATES = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1"
    
    all_locals = locals()
    n_items = None
    if 'loss_kwargs' in all_locals:
        __kwargs = all_locals['loss_kwargs']
        if type(__kwargs) is dict:
            n_items = __kwargs.get("num_items_in_batch", None) or __kwargs.get("n_items", None)
    if n_items is None and 'kwargs' in all_locals:
        __kwargs = all_locals['kwargs']
        if type(__kwargs) is dict:
            n_items = __kwargs.get("num_items_in_batch", None) or __kwargs.get("n_items", None)
    if n_items is None:
        all_locals = all_locals.values()
        for __kwargs in all_locals:
            if type(__kwargs) is dict:
                n_items = __kwargs.get("num_items_in_batch", None) or __kwargs.get("n_items", None)
                break
    pass
    
    requires_grad_ = self.lm_head.weight.requires_grad
    requires_grad_ = requires_grad_ or self.lm_head.weight.dtype == torch.float32
    
    if RETURN_HIDDEN_STATES:
        logits = hidden_states[:, slice_indices, :]
    elif labels is None:
        logits = self.lm_head(hidden_states[:, slice_indices, :])
    else:
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        torch._dynamo.mark_dynamic(logits, 1)
        torch._dynamo.mark_dynamic(labels, 1)
        if attention_mask is not None:
            torch._dynamo.mark_dynamic(attention_mask, 1)
        loss = compiled_ce_loss_function(
            output_logits        = logits,
            output_labels        = labels,
            logit_scale_multiply = () if () != () else 0,
            logit_scale_divide   = () if () != () else 0,
            logit_softcapping    = () if () not in (None, (),) else 0,
            vocab_size           = (self.config.text_config.vocab_size),
            n_items              = n_items if n_items is not None else 0,
            mask                 = attention_mask,
            requires_grad_       = requires_grad_,
        )


    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return Gemma3CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=outputs.image_hidden_states,
    )


class Gemma3ForConditionalGeneration(Gemma3PreTrainedModel, GenerationMixin):
    _checkpoint_conversion_mapping = {
        "^language_model.model": "model.language_model",
        "^vision_tower": "model.vision_tower",
        "^multi_modal_projector": "model.multi_modal_projector",
        "^language_model.lm_head": "lm_head",
    }
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Gemma3Config):
        super().__init__(config)
        self.model = Gemma3Model(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_image_features(self, pixel_values):
        return self.model.get_image_features(pixel_values)

    # Make modules available throught conditional class for BC
    @property
    def language_model(self):
        return self.model.language_model

    @property
    def vision_tower(self):
        return self.model.vision_tower

    @property
    def multi_modal_projector(self):
        return self.model.multi_modal_projector


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[list[torch.FloatTensor], Cache]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **lm_kwargs,
    ) -> Union[tuple, Gemma3CausalLMOutputWithPast]:
        return Gemma3ForConditionalGeneration_forward(self, input_ids, pixel_values, attention_mask, position_ids, past_key_values, token_type_ids, cache_position, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, return_dict, logits_to_keep, **lm_kwargs)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        pixel_values=None,
        attention_mask=None,
        token_type_ids=None,
        use_cache=True,
        logits_to_keep=None,
        labels=None,
        **kwargs,
    ):
        # Overwritten -- custom `position_ids` and `pixel_values` handling
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            token_type_ids=token_type_ids,
            **kwargs,
        )

        # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
        # Otherwise we need pixel values to be passed to model. NOTE: use_cache=False needs pixel_values always
        if cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values

        return model_inputs

    @staticmethod
    def create_masks_for_generate(
        config: PretrainedConfig,
        input_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        cache_position: torch.Tensor,
        past_key_values: Optional[Cache],
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        # Prepare mask arguments
        mask_kwargs = {
            "config": config.get_text_config(),
            "input_embeds": input_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
        }
        # Add the token type ids mask for generate as well
        if token_type_ids is not None and input_embeds.shape[1] != 1:
            # We need to pass an additional mask function to account for token type ids, and it needs to be an `or`
            mask_kwargs["or_mask_function"] = token_type_ids_mask_function(
                token_type_ids.to(cache_position.device), config.mm_tokens_per_image
            )

        return create_masks_for_generate(**mask_kwargs)
