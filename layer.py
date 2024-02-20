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
