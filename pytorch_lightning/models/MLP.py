import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        assert (
            num_layers >= 1
        ), "We should have at least one layer because the output layer is counted."

        # Define the model
        """
        self.mlp = nn.Sequential(
            nn.Flatten(),

            -----------------------------------------
            nn.Linear(28 * 28, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            -----------------------------------------
            
            nn.Linear(128, 10),
            nn.Softmax(dim=1),
        )
        """

        self.layers = []
        self.layers.append(nn.Flatten())
        current_dim = 28 * 28
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(current_dim, 128))
            self.layers.append(nn.BatchNorm1d(128))
            self.layers.append(nn.ReLU(inplace=True))
            current_dim = 128
        self.layers.append(nn.Linear(current_dim, 10))
        # self.layers.append(nn.Softmax(dim=1))
        # We don't need to use this because nn.CrossEntropyLoss() already includes softmax
        # Also, BCEWithLogitsLoss = Sigmoid + BCELoss
        self.mlp = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.mlp(x)

        return y


if __name__ == "__main__":
    model = MLP(num_layers=3)
    print(model)

    x = torch.randn(32, 1, 28, 28)
    y = model(x)
    print(y.shape)
