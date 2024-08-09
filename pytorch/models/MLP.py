import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, hidden_dim_sizes: list[int], use_bn: bool = True, **kwargs):
        super().__init__()

        # Define the model
        """
        self.mlp = nn.Sequential(
            nn.Flatten(),

            -----------------------------------------
            nn.Linear(28 * 28, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            -----------------------------------------
            
            nn.Linear(hidden_dim, 10),
            nn.Softmax(dim=1),
        )
        """

        self.layers = []
        self.layers.append(nn.Flatten())
        current_dim = 28 * 28
        for i in range(len(hidden_dim_sizes)):
            self.layers.append(nn.Linear(current_dim, hidden_dim_sizes[i]))
            if use_bn:
                self.layers.append(nn.BatchNorm1d(hidden_dim_sizes[i]))
            self.layers.append(nn.ReLU(inplace=True))
            current_dim = hidden_dim_sizes[i]
        self.layers.append(nn.Linear(current_dim, 10))
        # self.layers.append(nn.Softmax(dim=1))
        # We don't need to use this because nn.CrossEntropyLoss() already includes softmax
        # Also, BCEWithLogitsLoss = Sigmoid + BCELoss
        self.mlp = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.mlp(x)

        return y


if __name__ == "__main__":
    model = MLP(hidden_dim_sizes=[128, 128], use_bn=True)
    print(model)

    x = torch.randn(32, 1, 28, 28)
    y = model(x)
    print(y.shape)
