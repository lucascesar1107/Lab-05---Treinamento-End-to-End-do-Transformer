import math
import torch
import torch.nn as nn
import torch.nn.functional as F



# 1. FUNÇÕES E BLOCOS BASE


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output, attn_weights


class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


class AddNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, sublayer_output):
        return self.norm(x + sublayer_output)



# 2. EMBEDDING + POSITIONAL ENCODING


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]



# 3. ENCODER


class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.ffn      = PositionWiseFFN(d_model, d_ff)
        self.addnorm1 = AddNorm(d_model)
        self.addnorm2 = AddNorm(d_model)

    def forward(self, x, src_mask=None):
        attn_output, _ = scaled_dot_product_attention(x, x, x, mask=src_mask)
        x = self.addnorm1(x, attn_output)
        ffn_output = self.ffn(x)
        x = self.addnorm2(x, ffn_output)
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, num_layers, max_len=512):
        super().__init__()
        self.embedding           = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, d_ff) for _ in range(num_layers)
        ])

    def forward(self, x, src_mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x



# 4. DECODER


def generate_causal_mask(seq_len, device):
    return torch.tril(torch.ones(seq_len, seq_len, device=device))


class DecoderBlock(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.ffn      = PositionWiseFFN(d_model, d_ff)
        self.addnorm1 = AddNorm(d_model)
        self.addnorm2 = AddNorm(d_model)
        self.addnorm3 = AddNorm(d_model)

    def forward(self, y, Z):
        seq_len = y.size(1)
        causal_mask = generate_causal_mask(seq_len, y.device)
        masked_out, _ = scaled_dot_product_attention(y, y, y, mask=causal_mask)
        y = self.addnorm1(y, masked_out)

        cross_out, _ = scaled_dot_product_attention(y, Z, Z, mask=None)
        y = self.addnorm2(y, cross_out)

        ffn_out = self.ffn(y)
        y = self.addnorm3(y, ffn_out)
        return y


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, num_layers, max_len=512):
        super().__init__()
        self.embedding           = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, d_ff) for _ in range(num_layers)
        ])
        self.output_linear = nn.Linear(d_model, vocab_size)

    def forward(self, y, Z):
        y = self.embedding(y)
        y = self.positional_encoding(y)
        for layer in self.layers:
            y = layer(y, Z)
        logits = self.output_linear(y)  # retorna logits (sem softmax)
        return logits



# 5. TRANSFORMER COMPLETO


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=128, d_ff=256, num_layers=2, max_len=512):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, d_ff, num_layers, max_len)
        self.decoder = Decoder(tgt_vocab_size, d_model, d_ff, num_layers, max_len)

    def forward(self, encoder_input, decoder_input):
        Z      = self.encoder(encoder_input)
        logits = self.decoder(decoder_input, Z)
        return logits



# 6. INFERÊNCIA AUTO-REGRESSIVA (Lab 04)


def greedy_decode(model, encoder_input, start_token_id, eos_token_id, max_steps=50):
    model.eval()
    with torch.no_grad():
        Z = model.encoder(encoder_input)

        dec_input        = torch.tensor([[start_token_id]], dtype=torch.long,
                                        device=encoder_input.device)
        generated_tokens = [start_token_id]

        for _ in range(max_steps):
            logits     = model.decoder(dec_input, Z)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
            generated_tokens.append(next_token)

            if next_token == eos_token_id:
                break

            dec_input = torch.cat(
                [dec_input,
                 torch.tensor([[next_token]], dtype=torch.long,
                               device=encoder_input.device)],
                dim=1
            )

    return generated_tokens