# Adapted from  https://colab.research.google.com/drive/1VI2nhlyKvd5cw4-zHvAIk00cAVj2lCCC#scrollTo=e431d2cd

import torch
import transformers

old_init = transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.__init__
def ntk_scaled_init(self, dim, max_position_embeddings=2048, base=10000, device=None):
    # comment this out if you don't want to be inundated when the model instantiates
    print("using nltk scaling for RoPE")

    #The method is just these three lines
    max_position_embeddings = 16384
    a = 8 #Alpha value
    base = base * a ** (dim / (dim-2)) #Base change formula

    old_init(self, dim, max_position_embeddings, base, device)


# apply by placing the following in relevant scripts
# transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.__init__ = ntk_scaled_init