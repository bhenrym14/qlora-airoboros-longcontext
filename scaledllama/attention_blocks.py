import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers.models.llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb
import transformers
from xformers.ops import LowerTriangularMask, memory_efficient_attention

class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper

   Modifications: xformers memory_efficient_attention mechanism replaces native pytorch. Causal LowerTriangularMask is
   hardcoded so custom attention masks won't work as implemented.

    """

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)#.to(torch.bfloat16)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)#.to(torch.bfloat16)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)#.to(torch.bfloat16)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        scale = 1/math.sqrt(self.head_dim)
        attn_output = memory_efficient_attention(query_states.transpose(1,2),
                                                  key_states.transpose(1,2),
                                                  value_states.transpose(1,2),
                                                  scale = scale,
                                                  attn_bias = LowerTriangularMask()).transpose(1,2)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

def patch_in_memory_efficient_attention():
    print("patching in xformers memory efficient attention")
    transformers.models.llama.modeling_llama.LlamaAttention = (
        LlamaAttention
    )