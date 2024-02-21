# Example of transformer -------------------->
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from flash_attn.models.gpt import GPTLMHeadModel

def checker_attr(model_layers, attr):
    has_self_attention = False
    for layer in model_layers:
        if hasattr(layer.mixer, attr):
            has_self_attention = True
            break
    return has_self_attention


# example of gpt model 
seqlen = 2048
hidden_dim = 2048
nheads = 16
n_layer = 24
rotary_emb_fraction = 0.5
config = GPT2Config(vocab_size=50257, n_positions=seqlen, n_embd=hidden_dim,
                    n_layer=n_layer, n_head=nheads, 
                    scale_attn_by_inverse_layer_idx=True, 
                    rotary_emb_fraction=rotary_emb_fraction,
                    use_flash_attn=True, fused_mlp=True,
                    fused_bias_fc=True, fused_dropout_add_ln=True, 
                    pad_vocab_size_multiple=8)
model = GPTLMHeadModel(config)
# Getting layer of self flash attention
model.transformer.layers[0].mixer.inner_attn
model.transformer.layers[0].mixer.inner_cross_attn



# Example of flash attention -------------------->
from flash_attn import flash_attn_func
import torch
# Создаем тензоры q, k, v с более высокими размерностями
batch_size = 2
seqlen_q = 4
num_heads = 3
head_size_og = 5

# Создаем тензор для запросов с правильной формой
q = torch.ones((batch_size, seqlen_q, num_heads, head_size_og), dtype=torch.bfloat16, device='cuda:0')
# Создаем тензоры для ключей и значений с аналогичной формой
k = torch.ones((batch_size, seqlen_q, num_heads, head_size_og), dtype=torch.bfloat16, device='cuda:0')
v = torch.ones((batch_size, seqlen_q, num_heads, head_size_og), dtype=torch.bfloat16, device='cuda:0')



# Создаем тензор для запросов с правильной формой
q = torch.ones((1,1, 10), dtype=torch.bfloat16, device='cuda:0')
# Создаем тензоры для ключей и значений с аналогичной формой
k = torch.ones((1, 1, 10), dtype=torch.bfloat16, device='cuda:0')
v = torch.ones((1, 1, 10), dtype=torch.bfloat16, device='cuda:0')

# Вызываем функцию flash_attn_func с этими тензорами
output = flash_attn_func(q, k, v, 0.5, softmax_scale=1, causal=True)


import transformers