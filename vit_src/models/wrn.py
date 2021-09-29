import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from .stoch_depth import conv1x1, conv3x3, DropPath
from .resnet import init_weight


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1, keep_prob=1.):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride=stride)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = conv1x1(in_planes, planes, stride=stride)

        self.drop_path = nn.Identity() if keep_prob >= 1. else DropPath(keep_prob=keep_prob)

    def forward(self, x):
        out = self.dropout(self.conv1(self.relu(self.bn1(x))))
        out = self.conv2(self.relu(self.bn2(out)))
        out = self.drop_path(out)
        out += self.shortcut(x)

        return out


class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, keep_prob=1.):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert (depth - 4) % 6 == 0, 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) // 6
        k = widen_factor

        # print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.prob_now = 1.
        self.prob_step = (self.prob_now - keep_prob) / (n * 3 - 1)
        self.ce = nn.CrossEntropyLoss()

        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3])
        self.linear = nn.Linear(nStages[3], num_classes)

        init_weight(self)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride, keep_prob=self.prob_now))
            self.in_planes = planes
            self.prob_now -= self.prob_step

        return nn.Sequential(*layers)

    def forward(self, x, labels=None):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8, 1, 0)
        z = out.view(out.size(0), -1)
        out = self.linear(z)

        if labels is not None:
            loss = self.ce(out, labels)
            return loss
        else:
            return out, z


def wrn_28_10(dropout_rate=0., keep_prob=1., num_classes=100):
    return Wide_ResNet(depth=28,
                       widen_factor=10,
                       dropout_rate=dropout_rate,
                       num_classes=num_classes,
                       keep_prob=keep_prob)


if __name__ == '__main__':
    net = Wide_ResNet(28, 10, 0.3, 10)
    y = net(Variable(torch.randn(1,3,32,32)))

    print(y.size())
