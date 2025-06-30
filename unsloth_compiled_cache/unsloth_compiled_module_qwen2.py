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
from transformers.models.qwen2.modeling_qwen2 import (Optional, Union, torch, nn, ACT2FN, Cache, GenerationMixin, use_kernel_forward_from_hub, BaseModelOutputWithPast, CausalLMOutputWithPast, ROPE_INIT_FUNCTIONS, dynamic_rope_update, PreTrainedModel, Unpack, can_return_tuple, Qwen2Config, __name__, Qwen2PreTrainedModel, Qwen2Model, KwargsForCausalLM, Qwen2ForCausalLM)

@torch.compile(fullgraph = False, dynamic = True, options = torch_compile_options)
def Qwen2MLP_forward(self, x):
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj

class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return Qwen2MLP_forward(self, x)


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
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
def Qwen2RMSNorm_forward(self, hidden_states):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)

@use_kernel_forward_from_hub("RMSNorm")
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        return Qwen2RMSNorm_forward(self, hidden_states)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


@torch.compile(fullgraph = True, dynamic = True, options = torch_compile_options)
@torch.no_grad()
@dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
def Qwen2RotaryEmbedding_forward(self, x, position_ids):
    inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
    position_ids_expanded = position_ids[:, None, :].float()

    device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):  # Force float32
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen2Config, device=None):
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
        return Qwen2RotaryEmbedding_forward(self, x, position_ids)


@torch.compiler.disable(recursive = False)
@can_return_tuple
def Qwen2ForCausalLM_forward(
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
    **kwargs: Unpack[KwargsForCausalLM],
) -> CausalLMOutputWithPast:
    r"""
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
        config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
        (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Example:

    ```python
    >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

    >>> model = Qwen2ForCausalLM.from_pretrained("meta-qwen2/Qwen2-2-7b-hf")
    >>> tokenizer = AutoTokenizer.from_pretrained("meta-qwen2/Qwen2-2-7b-hf")

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""
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
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = EMPTY_LOGITS
    loss = None
    NOT_RETURN_LOGITS = os.environ.get('UNSLOTH_RETURN_LOGITS', '0') == '0'
    RETURN_HIDDEN_STATES = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1"
    
    n_items = None
    if (kwargs) != () and type(kwargs) is dict:
        n_items = (kwargs).get("num_items_in_batch", None) or (kwargs).get("n_items", None)
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
            logit_softcapping    = None if () == () else (),
            logit_scale_multiply = None if () == () else (),
            logit_scale_divide   = None if () == () else (),
        )
    elif (() == () and () == ()) and NOT_RETURN_LOGITS and self.loss_function.__name__.endswith("ForCausalLMLoss") and labels is not None and not requires_grad_:
        loss = fused_linear_cross_entropy(
            hidden_states      = hidden_states[:, slice_indices, :],
            lm_weight          = self.lm_head.weight,
            labels             = labels.to(self.lm_head.weight.device),
            num_items_in_batch = n_items,
            logit_softcapping  = None if () == () else (),
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
            logit_softcapping    = () if () not in (None, (),) else 0,
            vocab_size           = (self.config.vocab_size),
            n_items              = n_items if n_items is not None else 0,
            requires_grad_       = requires_grad_,
        )
    else:
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if () != ():
            logits = logits * ()
        if () != ():
            logits = logits / ()
        if () not in (None, (),):
            logits = logits / ()
            logits = torch.tanh(logits)
            logits = logits * ()
        loss = self.loss_function(logits=logits, labels=labels.to(self.lm_head.weight.device), self.config.vocab_size, **kwargs)


    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

class Qwen2ForCausalLM(Qwen2PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2Model(config)
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
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        return Qwen2ForCausalLM_forward(self, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, cache_position, logits_to_keep, **kwargs)
