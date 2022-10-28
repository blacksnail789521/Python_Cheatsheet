import torch
import torch.nn as nn


class DNN(nn.Module):
    
    def __init__(self, dimensions) -> None:
        
        super().__init__()
        
        '''
        self.dnn = nn.Sequential(
            nn.Linear(10, 1000, bias = True),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace = True),
            
            nn.Linear(1000, 1000, bias = True),
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace = True),
            
            nn.Linear(1000, 1, bias = True),
        )
        '''
        
        self.layers = []
        for i in range(len(dimensions) - 1):
            self.layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
            if i < len(dimensions) - 2:
                self.layers.append(nn.BatchNorm1d(dimensions[i + 1]))
                self.layers.append(nn.ReLU(inplace = True))
        
        self.dnn = nn.Sequential(*self.layers)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        y = self.dnn(x)
        
        return y


if __name__ == '__main__':
    
    model = DNN(dimensions = [10, 1000, 1000, 1]) # include input_dim and output_dim
    print(model)
    '''
    DNN(
      (dnn): Sequential(
        (0): Linear(in_features=10, out_features=1000, bias=True)
        (1): BatchNorm1d(1000, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
        (3): Linear(in_features=1000, out_features=1000, bias=True)
        (4): BatchNorm1d(1000, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): ReLU(inplace=True)
        (6): Linear(in_features=1000, out_features=1, bias=True)
      )
    )
    '''