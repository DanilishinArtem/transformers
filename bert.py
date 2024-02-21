import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
import flash_attn_2_cuda as flash_attn_cuda
from transformers.models.bert.modeling_bert import BertSelfAttention
from typing import List, Optional, Tuple, Union

def _flash_attn_forward(
    q, k, v, dropout_p, softmax_scale, causal, window_size, alibi_slopes, return_softmax
):
    maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = flash_attn_cuda.fwd(
        q,
        k,
        v,
        None,
        alibi_slopes,
        dropout_p,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        return_softmax,
        None,
    )
    return out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state

def _flash_attn_backward(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    dq,
    dk,
    dv,
    dropout_p,
    softmax_scale,
    causal,
    window_size,
    alibi_slopes,
    deterministic,
    rng_state=None,
):
    maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
    # dq, dk, dv are allocated by us so they should already be contiguous
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    dq, dk, dv, softmax_d, = flash_attn_cuda.bwd(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        alibi_slopes,
        dropout_p,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        deterministic,
        None,
        rng_state,
    )
    return dq, dk, dv, softmax_d


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        out, q, k, v, out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
        )
        ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, out_padded, softmax_lse, rng_state = ctx.saved_tensors
        q, k, v, out_padded, softmax_lse = map(lambda x: x.detach(), (q, k, v, out_padded, softmax_lse))
        grad_q, grad_k, grad_v = _flash_attn_backward(
            grad_out.contiguous(),
            q,
            k,
            v,
            out_padded,
            softmax_lse,
            rng_state,
        )
        return grad_q, grad_k, grad_v, None, None, None, None, None, None, None, None

class FlashAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FlashAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    # def forward(self, input_ids):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ):
        num_heads = 16
        batch_size = 1
        seqlen_q = 5
        head_size_og = 48
        q = torch.ones((batch_size, seqlen_q, num_heads, head_size_og), dtype=torch.bfloat16, device='cuda')
        q = input_ids[0,:].unsqueeze(0).unsqueeze(-2).expand(1, seqlen_q, num_heads, -1).to(dtype=torch.bfloat16)
        k = torch.ones((batch_size, seqlen_q, num_heads, head_size_og), dtype=torch.bfloat16, device='cuda')
        k = input_ids[1,:].unsqueeze(0).unsqueeze(-2).expand(1, seqlen_q, num_heads, -1).to(dtype=torch.bfloat16)
        v = torch.ones((batch_size, seqlen_q, num_heads, head_size_og), dtype=torch.bfloat16, device='cuda')
        v = input_ids[2,:].unsqueeze(0).unsqueeze(-2).expand(1, seqlen_q, num_heads, -1).to(dtype=torch.bfloat16)
        # You can define your logic here for using FlashAttnFunc
        dropout_p = 0.1
        softmax_scale = None
        causal = True
        window_size = (-1, -1)
        alibi_slopes = None
        deterministic = False
        return_softmax = False

        return FlashAttnFunc.apply(
            q, k, v, dropout_p, softmax_scale, causal, window_size, alibi_slopes, deterministic, return_softmax
        )

class BertWithFlashAttention(nn.Module):
    def __init__(self, num_classes, flash_attention_dim=512):
        super(BertWithFlashAttention, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.flash_attention = FlashAttention(input_dim=self.bert.config.hidden_size,output_dim=num_classes)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_output = outputs.last_hidden_state[:, 0, :].to(dtype=torch.bfloat16)  # Use [CLS] token representation
        # flash_attention_output = self.flash_attention(q=bert_output, k=bert_output, v=bert_output)
        logits = self.classifier(bert_output)
        return logits, flash_attention_output

# Example usage
model = BertWithFlashAttention(num_classes=2).to('cuda')
# input_ids = torch.tensor([[1, 2, 3, 4, 5]]).to('cuda')
attention_mask = torch.tensor([[1, 1, 1, 1, 1]]).to('cuda')
input_ids = model.bert.dummy_inputs['input_ids'].to('cuda')

logits, flash_attention_output = model(input_ids, attention_mask)
print("Logits:", logits)
print("Flash Attention Output:", flash_attention_output)
 

bert = BertModel.from_pretrained('bert-base-uncased')
bert = bert.to('cuda')
bert(input_ids=input_ids, attention_mask=attention_mask)
# bert.encoder.layer[0].attention = FlashAttention(input_dim=bert.encoder.layer[0].attention.self.query.in_features,output_dim=bert.encoder.layer[0].attention.output.dense.out_features)
bert.encoder.layer[0] = FlashAttention(input_dim=bert.encoder.layer[0].attention.self.query.in_features,output_dim=bert.encoder.layer[0].attention.output.dense.out_features)

# BertSelfAttention


# out for embeddings
out_embeddings = bert.embeddings(input_ids=input_ids)

# out for attention
out_attenrion = bert.encoder.layer[0].attention.self(out_embeddings)[0]

attention 

num_heads = 16
batch_size = 1
seqlen_q = 5
head_size_og = 48

q = torch.ones((batch_size, seqlen_q, num_heads, head_size_og), dtype=torch.bfloat16, device='cuda:0')
q = out_embeddings[0,:].unsqueeze(0).unsqueeze(-2).expand(1, seqlen_q, num_heads, -1)
k = torch.ones((batch_size, seqlen_q, num_heads, head_size_og), dtype=torch.bfloat16, device='cuda:0')
q = out_embeddings[1,:].unsqueeze(0).unsqueeze(-2).expand(1, seqlen_q, num_heads, -1)
v = torch.ones((batch_size, seqlen_q, num_heads, head_size_og), dtype=torch.bfloat16, device='cuda:0')
q = out_embeddings[2,:].unsqueeze(0).unsqueeze(-2).expand(1, seqlen_q, num_heads, -1)


