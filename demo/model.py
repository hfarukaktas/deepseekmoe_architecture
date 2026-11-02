"""
model.py
========

The two models from DeepseekMoE.ipynb, for inference. A pre-norm Transformer
whose feed-forward layer is, with moe=True, a Mixture-of-Experts in every block
but the first: 1 shared expert plus 8 routed experts, of which each token uses
the top 2. With moe=False every block keeps a dense FFN (the notebook's GPT).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, embedding_dim: int, expert_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, expert_dim),
            nn.GELU(),
            nn.Linear(expert_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepSeekMoE(nn.Module):
    def __init__(self, embedding_dim: int = 384, expert_dim: int = 512, num_shared: int = 1,
                 num_routed: int = 8, top_k: int = 2, dropout: float = 0.1):
        super().__init__()
        self.num_routed = num_routed
        self.top_k = top_k
        self.shared_experts = nn.ModuleList(
            [Expert(embedding_dim, expert_dim, dropout) for _ in range(num_shared)])
        self.routed_experts = nn.ModuleList(
            [Expert(embedding_dim, expert_dim, dropout) for _ in range(num_routed)])
        self.router = nn.Linear(embedding_dim, num_routed, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x_flat = x.view(-1, C)

        shared_out = torch.zeros_like(x_flat)
        for expert in self.shared_experts:
            shared_out += expert(x_flat)

        routing_weights = F.softmax(self.router(x_flat), dim=-1)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        routed_out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.routed_experts):
            mask = topk_indices == i
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero(as_tuple=True)[0]
                weight = (topk_weights[token_idx] * mask[token_idx]).sum(dim=-1, keepdim=True)
                routed_out[token_idx] += weight * expert(x_flat[token_idx])

        return (shared_out + routed_out).view(B, T, C)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float, is_moe: bool):
        super().__init__()
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout,
                                               batch_first=True)
        self.ln2 = nn.LayerNorm(embedding_dim)
        if is_moe:
            self.ffn = DeepSeekMoE(embedding_dim=embedding_dim, dropout=dropout)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim * 4),
                nn.GELU(),
                nn.Linear(embedding_dim * 4, embedding_dim),
                nn.Dropout(dropout),
            )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        x_norm = self.ln1(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm, attn_mask=causal_mask,
                                     need_weights=False)
        x = x + attn_out
        return x + self.ffn(self.ln2(x))


class GPTwithMoE(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 384, num_heads: int = 6,
                 num_layers: int = 6, dropout: float = 0.1, block_size: int = 256,
                 moe: bool = True):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, dropout, is_moe=moe and i > 0)
            for i in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(embedding_dim)
        self.output_proj = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.output_proj.weight = self.token_embedding.weight   # weight tying

        causal_mask = torch.triu(torch.ones(block_size, block_size, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal_mask)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor | None = None):
        _, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device)
        x = self.dropout(self.token_embedding(input_ids) + self.position_embedding(positions))
        mask = self.causal_mask[:T, :T]
        for block in self.blocks:
            x = block(x, mask)
        logits = self.output_proj(self.ln_final(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
