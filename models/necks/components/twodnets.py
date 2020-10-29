import torch.nn as nn
from  torch.nn  import Sequential
from functools import partial
from models.utils import change_default_args, Sequential
import torch

__all__ = ['BEVNet', 'PCDetBEVNet']
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
        conv6 = x
        x = self.conv7(x)
        x = F.relu(self.bn7(x), inplace=True)
        return x, conv6
    
class PCDetBEVNet(nn.Module):
    def __init__(self, args):
        super(PCDetBEVNet, self).__init__()
        self._concat_input = args['concat_input']
        assert len(args['layer_strides']) == len(args['layer_nums'])
        assert len(args['num_filters']) == len(args['layer_nums'])
        assert len(args['num_upsample_filters']) == len(args['layer_nums'])
        
        # if args['use_norm']:
        BatchNorm2d = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        Conv2d = partial(nn.Conv2d, bias=False)
        ConvTranspose2d = partial(nn.ConvTranspose2d, bias=False)

        in_filters = [args['num_input_features'], *args['num_filters'][:-1]]
        blocks = []
        deblocks = []

        for i, layer_num in enumerate(args['layer_nums']):
            block = Sequential(
                nn.ZeroPad2d(1),
                Conv2d(in_filters[i], args['num_filters'][i], 3, stride=args['layer_strides'][i]),
                BatchNorm2d(args['num_filters'][i]),
                nn.ReLU(),
            )
            for j in range(layer_num):
                block.add(Conv2d(args['num_filters'][i], args['num_filters'][i], 3, padding=1))
                block.add(BatchNorm2d(args['num_filters'][i]))
                block.add(nn.ReLU())
            blocks.append(block)
            deblock = Sequential(
                ConvTranspose2d(
                    args['num_filters'][i], args['num_upsample_filters'][i], args['upsample_strides'][i],
                    stride=args['upsample_strides'][i]
                ),
                BatchNorm2d(args['num_upsample_filters'][i]),
                nn.ReLU(),
            )
            deblocks.append(deblock)

        c_in = sum(args['num_upsample_filters'])
        if self._concat_input:
            c_in += args['num_input_features']

        if len(args['upsample_strides']) > len(args['num_filters']):
            deblock = Sequential(
                ConvTranspose2d(c_in, c_in, args['upsample_strides'][-1], stride=args['upsample_strides'][-1]),
                BatchNorm2d(c_in),
                nn.ReLU(),
            )
            deblocks.append(deblock)
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)
        
        self.conv_out = Sequential(
                # nn.ZeroPad2d(1),
                Conv2d(c_in, args['num_output_features'], 1),
                BatchNorm2d(args['num_output_features']),
                nn.ReLU(),
            )
        
    def forward(self, x_in):
        ups = []
        x = x_in
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            ups.append(self.deblocks[i](x))

        if self._concat_input:
            ups.append(x_in)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        else:
            x = ups[0]
        
        if len(self.deblocks)>len(self.blocks):
            x = self.deblocks[-1](x)        
        
        conv_ps = ups[0]
        
        x = self.conv_out(x)
        
        return x, conv_ps 