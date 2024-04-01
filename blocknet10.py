import torch
import torch.nn as nn
import torch.nn.functional as F

class IntermediateBlock(nn.Module):
    def __init__(self, in_channels, num_conv_layers, conv_params):
        super(IntermediateBlock, self).__init__()
        self.conv_layers = nn.ModuleList([nn.Conv2d(in_channels, *conv_params) for _ in range(num_conv_layers)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm2d(conv_params[0]) for _ in range(num_conv_layers)])
        out_channels = conv_params[0]
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        batch_size = x.size(0)
        channel_means = x.mean(dim=[2, 3])
        a = self.fc(channel_means)
        x_out = torch.stack([F.leaky_relu(conv(x)) for conv in self.conv_layers], dim=-1).sum(dim=-1)
        x_out = torch.stack([bn(x_out) for bn in self.batch_norms], dim=-1).sum(dim=-1)
        return x_out * F.leaky_relu(a.view(batch_size, -1, 1, 1))

class OutputBlock(nn.Module):
    def __init__(self, in_channels, num_classes, hidden_sizes=[]):
        super(OutputBlock, self).__init__()
        self.fc_layers = nn.ModuleList([nn.Linear(in_channels, hidden_sizes[0])] + [nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes)-1)] + [nn.Linear(hidden_sizes[-1], num_classes)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(size) for size in hidden_sizes])

    def forward(self, x):
        channel_means = x.mean(dim=[2, 3])
        out = F.leaky_relu(channel_means)
        for fc, bn in zip(self.fc_layers, self.batch_norms):
            out = F.leaky_relu(bn(fc(out)))
        return out

class CustomCIFAR10Net(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomCIFAR10Net, self).__init__()
        self.intermediate_blocks = nn.ModuleList([
            IntermediateBlock(3, 3, [64, 3, 3, 1, 1]),
            IntermediateBlock(64, 3, [128, 3, 3, 1, 1]),
            IntermediateBlock(128, 3, [256, 3, 3, 1, 1]),
            IntermediateBlock(256, 3, [512, 3, 3, 1, 1]),
            IntermediateBlock(512, 3, [1024, 3, 3, 1, 1])
        ])
        self.output_block = OutputBlock(1024, num_classes, [512, 256])
        self.dropout = nn.Dropout(0.5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')


    def forward(self, x):
        for block in self.intermediate_blocks:
            x = block(x)
            x = self.dropout(x)
        x = self.output_block(x)
        return x