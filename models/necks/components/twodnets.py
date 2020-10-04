import torch.nn as nn
from functools import partial

class BEVNet(nn.Module):
    def __init__(self, in_features, num_filters=256):
        super(BEVNet, self).__init__()
        BatchNorm2d = change_default_args(
            eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
        # Conv2d = change_default_args(bias=False)(nn.Conv2d)
        Conv2d = partial(nn.Con2d, bias=False)
        BatchNorm2d = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        self.conv0 = Conv2d(in_features, num_filters, 3, padding=1)
        self.bn0 = BatchNorm2d(num_filters)

        self.conv1 = Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn1 = BatchNorm2d(num_filters)

        self.conv2 = Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn2 = BatchNorm2d(num_filters)

        self.conv3 = Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn3 = BatchNorm2d(num_filters)

        self.conv4 = Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn4 = BatchNorm2d(num_filters)

        self.conv5 = Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn5 = BatchNorm2d(num_filters)

        self.conv6 = Conv2d(num_filters, num_filters, 3, padding=1)
        self.bn6 = BatchNorm2d(num_filters)

        self.conv7 = Conv2d(num_filters, num_filters, 1)
        self.bn7 = BatchNorm2d(num_filters)

    def forward(self, x):
        x = self.conv0(x)
        x = F.relu(self.bn0(x), inplace=True)
        x = self.conv1(x)
        x = F.relu(self.bn1(x), inplace=True)
        x = self.conv2(x)
        x = F.relu(self.bn2(x), inplace=True)
        x = self.conv3(x)
        x = F.relu(self.bn3(x), inplace=True)
        x = self.conv4(x)
        x = F.relu(self.bn4(x), inplace=True)
        x = self.conv5(x)
        x = F.relu(self.bn5(x), inplace=True)
        x = self.conv6(x)
        x = F.relu(self.bn6(x), inplace=True)
        conv6 = x.clone()
        x = self.conv7(x)
        x = F.relu(self.bn7(x), inplace=True)
        return x, conv6