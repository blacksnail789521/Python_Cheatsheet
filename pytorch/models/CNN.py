import torch
import torch.nn as nn


import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(
        self,
        num_conv_layers: int = 2,
        use_bn: bool = True,
    ) -> None:
        super().__init__()

        assert num_conv_layers >= 1, "We should have at least one convolutional layer."
        assert num_conv_layers <= 3, (
            "We don't want to have too many convolutional layers (<= 3) \n"
            "because we double the number of channels each time."
        )

        # Define the model
        """
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            -----------------------------------------
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            -----------------------------------------
            
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 10),
            nn.Softmax(dim=1),
        )
        """

        self.layers = []

        # First convolutional layer
        self.layers.append(nn.Conv2d(1, 32, kernel_size=3, padding=1))
        if use_bn:
            self.layers.append(nn.BatchNorm2d(32))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # Additional convolutional layers
        current_channels = 32
        for _ in range(num_conv_layers - 1):
            self.layers.append(
                nn.Conv2d(
                    current_channels, current_channels * 2, kernel_size=3, padding=1
                )
            )
            if use_bn:
                self.layers.append(nn.BatchNorm2d(current_channels * 2))
            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            current_channels *= 2

        self.layers.append(nn.Flatten())

        # Fully connected layer
        self.layers.append(
            nn.Linear(current_channels * (28 // (2**num_conv_layers)) ** 2, 10)
        )
        # self.layers.append(nn.Softmax(dim=1))  # Not needed for nn.CrossEntropyLoss()

        self.cnn = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cnn(x)
        return y


if __name__ == "__main__":
    model = CNN(num_conv_layers=3)
    print(model)

    x = torch.randn(128, 1, 28, 28)
    y = model(x)
    print(y.shape)
