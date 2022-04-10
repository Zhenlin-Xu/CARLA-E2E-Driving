import torch
import torch.nn as nn

class CI_Net(nn.Module):
    ''' The network architecture (command input) of Conditional Imitation Learning Paper. '''
    def __init__(self):
        super(CI_Net, self).__init__()

        # conv attributes
        io = [3, 32, 32, 64, 64, 128, 128, 256, 256,]   # channels
        ks = [5, 3, 3, 3, 3, 3, 3, 3,]                  # kernel size
        st = [2, 1, 2, 1, 2, 1, 1, 1,]                  # stride

        conv_model = []
        for i in range(len(io)-1):
            conv_model += [
                nn.Conv2d(in_channels=io[i], out_channels=io[i+1], kernel_size=ks[i], stride=st[i], padding=0,),
                nn.BatchNorm2d(num_features=io[i+1]),
                nn.Dropout2d(p=0.2),
                nn.ReLU(),
            ]

        unit = [12288+9, 512, 32]

        mlp_model = []
        for i in range(len(unit)-1):
            mlp_model += [
                nn.Linear(in_features=unit[i], out_features=unit[i+1]),
                # nn.BatchNorm1d(num_features=unit[i+1]),
                nn.Dropout(p=0.5),
                nn.ReLU()
            ]
        mlp_model += [nn.Linear(in_features=unit[2], out_features=2)]

        self.conv = nn.Sequential(*conv_model)
        self.flat = nn.Flatten()
        self.mlp  = nn.Sequential(*mlp_model) 

    def forward(self, x, act):
        x = self.conv(x)
        x = self.flat(x)
        x = torch.cat((x, act), dim=1)
        x = self.mlp(x)

        return x

if __name__ == "__main__":
    net = CI_Net()
    print(net)

    inp = torch.randn((1,3,100,200))
    out = net(inp, torch.randn(1,9))
    print(out.shape)


