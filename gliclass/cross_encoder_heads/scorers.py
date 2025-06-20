import torch
from torch import nn

class MLPScorer(nn.Module):
    def __init__(self, hidden_size, mlp_hidden_size=256):
        super().__init__()
        # Define the MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, mlp_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size // 2, 1)
        )

    def forward(self, cls_rep: torch.Tensor, cls_mask: torch.Tensor) -> torch.Tensor:

        scores = self.mlp(cls_rep).squeeze(-1)  # (batch_size, num_classes)
        scores = scores * cls_mask  # Apply mask to scores
        
        return scores

SCORER2OBJECT = {
    "mlp": MLPScorer,
}