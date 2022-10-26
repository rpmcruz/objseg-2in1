import torch

class UNet(torch.nn.Module):
    def __init__(self, in_channels, nlayers=4):
        super().__init__()
        self.conv = torch.nn.ModuleList()
        self.deconv = torch.nn.ModuleList()
        prev, next = in_channels, 64
        for _ in range(nlayers):
            self.conv.append(torch.nn.Conv2d(prev, next, 3, 2, 1))
            prev, next = next, next*2
        self.bottleneck = torch.nn.Conv2d(prev, next, 3, 1, 1)
        prev, next = next, next//2
        for i in range(nlayers):
            self.deconv.append(torch.nn.ConvTranspose2d(prev+prev//2, next, 3, 2, 1, 1))
            prev, next = next, next//2
        self.final = torch.nn.Conv2d(prev, 1, 3, 1, 1)

    def forward(self, x, out2=None):
        skip = []
        for layer in self.conv:
            x = torch.nn.functional.relu(layer(x))
            skip.append(x)
        x = self.bottleneck(x)
        for i, layer in enumerate(self.deconv):
            y = torch.cat((x, skip[-i-1]), 1)
            x = pre_x = torch.nn.functional.relu(layer(y))
        if out2 is not None:
            #x = torch.cat((x, out2), 1)
            x += out2
        x = self.final(x)
        return x, pre_x
