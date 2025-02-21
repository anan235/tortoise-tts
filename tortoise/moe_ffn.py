# moe_ffn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoEFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, num_experts=4, k=1, dropout=0.1):
        """
        Sparse Mixture-of-Experts (MoE) FeedForward network.
        
        Args:
            d_model (int): Model dimension.
            d_ff (int): Hidden dimension for each expert.
            num_experts (int): Total number of experts.
            k (int): Number of experts to activate per token.
            dropout (float): Dropout probability.
        """
        super(MoEFeedForward, self).__init__()
        self.num_experts = num_experts
        self.k = k  # For simplicity, we use k=1 (top expert selection)
        self.d_model = d_model
        self.d_ff = d_ff

        # Define experts: each expert is a small feed-forward network.
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])

        # Gating network to select which expert(s) to use.
        self.gate = nn.Linear(d_model, num_experts)
        # Dynamic bias for each expert (non-trainable, updated externally if desired)
        self.register_buffer('expert_bias', torch.zeros(num_experts))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq, d_model)
        Returns:
            Output tensor after MoE FFN.
        """
        batch, seq, _ = x.shape
        # Compute gating logits and add bias.
        gate_logits = self.gate(x) + self.expert_bias  # (batch, seq, num_experts)
        gate_probs = F.softmax(gate_logits, dim=-1)  # (batch, seq, num_experts)
        # For simplicity, select the top expert (k=1) per token.
        top_expert_indices = torch.argmax(gate_probs, dim=-1)  # (batch, seq)

        output = torch.zeros_like(x)
        # Process each expert separately (a loop over experts; vectorize for efficiency in production).
        for expert_idx in range(self.num_experts):
            mask = (top_expert_indices == expert_idx)  # (batch, seq) boolean mask
            if mask.sum() == 0:
                continue
            x_expert = x[mask]  # (N, d_model) where N is the number of tokens for this expert.
            expert_output = self.experts[expert_idx](x_expert)
            output[mask] = expert_output
        output = self.dropout(output)
        return output
