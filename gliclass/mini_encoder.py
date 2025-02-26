import torch.nn as nn
import torch


class TextCompressionLayer(nn.Module):
    def __init__(self, hidden_size, n_pathes=128, num_heads=4, dropout=0.1, alpha=0.5):
        super().__init__()

        self.attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.query_tokens = nn.Parameter(torch.randn(n_pathes, hidden_size))
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, text_embeddings: torch.Tensor):
        B, L, D = text_embeddings.size()

        pooled = text_embeddings.mean(dim=1, keepdim=True)  # (B, 1, D)

        q = self.query_tokens.unsqueeze(0) + self.alpha * pooled  # (B, output_len, D)

        attn_output, _ = self.attn(q, text_embeddings, text_embeddings)  # (B, output_len, D)

        q = self.norm1(q + attn_output)

        ffn_output = self.ffn(q)

        compressed_text = self.norm2(q + ffn_output)

        return compressed_text  # (B, output_len, D)

class ScorerSelfAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super().__init__()

        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor):
        attn_output, _ = self.multihead_attn(x, x, x)  # (B, L, D)

        x = self.norm1(x + attn_output)

        ffn_output = self.ffn(x)

        x = self.norm2(x + ffn_output)

        return x  # (B, L, D)

class ScorerCrossAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super().__init__()

        self.attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.proj_q = nn.Linear(hidden_size, hidden_size)
        self.proj_kv = nn.Linear(hidden_size, hidden_size)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x_1, x_2):
        """
        x_1: (B, L_1, D) - embeddings that will be updated (Query)
        x_2: (B, L_2, D) - embeddings from which information is extracted (Key, Value)

        return: (B, L_1, D) - updated x_1
        """
        q = self.proj_q(x_1)  # (B, L_1, D) - Query
        kv = self.proj_kv(x_2)  # (B, L_2, D) - Key/Value

        attn_output, _ = self.attn(q, kv, kv)  # (B, L_1, D)

        updated_x1 = self.norm1(x_1 + attn_output)

        ffn_output = self.ffn(updated_x1)
        final_x1 = self.norm2(updated_x1 + ffn_output)

        return final_x1  # (B, L_1, D)


class ScorerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super().__init__()

        self.text_cross_attn = ScorerCrossAttentionLayer(hidden_size, num_heads=num_heads, dropout=dropout)
        self.label_self_attn = ScorerSelfAttentionLayer(hidden_size, num_heads=num_heads, dropout=dropout)
        self.label_cross_attn = ScorerCrossAttentionLayer(hidden_size, num_heads=num_heads, dropout=dropout)
    
    def forward(self, compressed_text_rep, text_rep, label_rep):
        compressed_text_rep = self.text_cross_attn(compressed_text_rep, text_rep)
        label_rep = self.label_self_attn(label_rep)
        label_rep = self.label_cross_attn(label_rep, compressed_text_rep)
        compressed_text_rep = self.text_cross_attn(compressed_text_rep, label_rep)
        return compressed_text_rep, label_rep


class MiniEncoderScorer(nn.Module):
    def __init__(self, hidden_size=768, reduced_hidden_size=384, num_heads=4, dropout=0.1, num_blocks=3, alpha=0.5, n_pathes=64):
        super().__init__()
        self.projection = nn.Linear(hidden_size, reduced_hidden_size)
        self.input_compression = TextCompressionLayer(reduced_hidden_size, num_heads=num_heads, dropout=dropout, alpha=alpha, n_pathes=n_pathes)
        self.scorer_blocks = nn.ModuleList([ScorerBlock(reduced_hidden_size, num_heads=num_heads, dropout=dropout) for _ in range(num_blocks)])
        #self.output_compression = TextCompressionLayer(reduced_hidden_size, num_heads=num_heads, dropout=dropout, alpha=alpha, n_pathes=1)

    def forward(self, text_rep, label_rep, *args):
        text_rep = self.projection(text_rep)  # (B, L, 768) → (B, L, 384)
        label_rep = self.projection(label_rep) # (B, L, 768) → (B, L, 384)
        compressed_text_rep = self.input_compression(text_rep)

        for block in self.scorer_blocks:
            compressed_text_rep, label_rep = block(compressed_text_rep, text_rep, label_rep)
        
        #text_rep = self.output_compression(compressed_text_rep)
        text_rep = compressed_text_rep[:, 0, :]
        scores = torch.einsum('BD,BCD->BC', text_rep, label_rep)
        return scores
