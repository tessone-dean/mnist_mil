import torch, torch.nn as nn, torch.nn.functional as F

class MILAttention(nn.Module):
    """Un‑gated attention:  \tilde a_k = w^T tanh(V h_k)"""
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.V = nn.Linear(feature_dim, hidden_dim, bias=True)
        self.w = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h):                      # h: (B, K, d)
        a_tilde = self.w(torch.tanh(self.V(h)))   # (B, K, 1)
        return a_tilde.squeeze(-1)                # (B, K)


class GatedAttention(nn.Module):
    """Gated attention:  w^T[ tanh(V h) ⊙ sigmoid(U h) ]"""
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.V = nn.Linear(feature_dim, hidden_dim, bias=True)
        self.U = nn.Linear(feature_dim, hidden_dim, bias=True)
        self.w = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h):                       # h: (B, K, d)
        v = torch.tanh(self.V(h))
        u = torch.sigmoid(self.U(h))
        gated = v * u                           # (B, K, L)
        a_tilde = self.w(gated)                 # (B, K, 1)
        return a_tilde.squeeze(-1)              # (B, K)


class MILClassifier(nn.Module):
    def __init__(self, feature_dim, hidden_dim, gated=False):
        super().__init__()
        Attn = GatedAttention if gated else MILAttention
        self.attn = Attn(feature_dim, hidden_dim)
        self.clf  = nn.Linear(feature_dim, 1)

    def forward(self, H):
        a_tilde = self.attn(H)                 # (B, K)

        a_k = torch.softmax(a_tilde, dim=1)    # (B, K)
        z   = (a_k.unsqueeze(-1) * H).sum(dim=1)   # (B, d)

        logits = self.clf(z).squeeze(-1)       # (B,)
        return torch.sigmoid(logits), a_k

class MILClassifierMasked(MILClassifier):
    """
    Same as MILClassifier, but add a Boolean mask so padded
    instances don’t participate in attention or aggregation.
    """
    def forward(self, H, mask):
        """
        H    : (B, K, d)  padded batch of bags
        mask : (B, K)     1 for real instance, 0 for padding
        """
         
        a_tilde = self.attn(H)                  # (B,K)
        a_tilde = a_tilde.masked_fill(~mask, -1e9)  # kill pads
        a = torch.softmax(a_tilde, 1)           # (B,K)
        z = (a.unsqueeze(-1) * H).sum(1)        # (B,d)
        logits = self.clf(z).squeeze(-1)        # (B,)
        return logits, a                        # return raw logits
