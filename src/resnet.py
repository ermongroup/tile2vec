'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, no_relu=False,
        activation='relu'):
        super(BasicBlock, self).__init__()
        self.no_relu = no_relu
        self.activation = activation

        # Choose activation function
        if self.activation == 'relu':
            self.activation_fn = F.relu
        elif self.activation == 'leaky_relu':
            self.activation_fn = F.leaky_relu

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # no_relu layer
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes)
        # no_relu layer

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        out = self.activation_fn(self.bn1(self.conv1(x)))
        if self.no_relu:
            out = self.bn3(self.conv3(out))
            return out
        else:
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            # out = F.relu(out)
            out = self.activation_fn(out)
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, n_classes=10, in_channels=4,
        z_dim=512, supervised=False, no_relu=False, loss_type='triplet',
        tile_size=50, activation='relu'):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.in_channels = in_channels
        self.z_dim = z_dim
        self.supervised = supervised
        self.no_relu = no_relu
        self.loss_type = loss_type
        self.tile_size = tile_size
        self.activation = activation

        # Choose activation function
        if self.activation == 'relu':
            self.activation_fn = F.relu
        elif self.activation == 'leaky_relu':
            self.activation_fn = F.leaky_relu

        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.layer5 = self._make_layer(block, self.z_dim, num_blocks[4],
            stride=2, no_relu=self.no_relu)
        self.linear = nn.Linear(self.z_dim*block.expansion, n_classes)

    def _make_layer(self, block, planes, num_blocks, stride, no_relu=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride=stride,
                no_relu=no_relu, activation=self.activation))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def encode(self, x, verbose=False):
        # x = F.relu(self.bn1(self.conv1(x)))
        x = self.activation_fn(self.bn1(self.conv1(x)))
        if verbose: print(x.shape)
        x = self.layer1(x)
        if verbose: print(x.shape)
        x = self.layer2(x)
        if verbose: print(x.shape)
        x = self.layer3(x)
        if verbose: print(x.shape)
        x = self.layer4(x)
        if verbose: print(x.shape)
        x = self.layer5(x)
        if verbose: print(x.shape)
        if self.tile_size == 50:
            x = F.avg_pool2d(x, 4)
        elif self.tile_size == 25:
            x = F.avg_pool2d(x, 2)
        elif self.tile_size == 75:
            x = F.avg_pool2d(x, 5)
        elif self.tile_size == 100:
            x = F.avg_pool2d(x, 7)
        if verbose: print('Pooling:', x.shape)
        z = x.view(x.size(0), -1)
        if verbose: print('View:', z.shape)
        return z

    def forward(self, x):
        if self.supervised:
            z = self.encode(x)
            return self.linear(z)
        else:
            return self.encode(x)

    def triplet_loss(self, z_p, z_n, z_d, margin=0.1, l2=0):
        l_n = torch.sqrt(((z_p - z_n) ** 2).sum(dim=1))
        l_d = - torch.sqrt(((z_p - z_d) ** 2).sum(dim=1))
        l_nd = l_n + l_d
        loss = F.relu(l_n + l_d + margin)
        l_n = torch.mean(l_n)
        l_d = torch.mean(l_d)
        l_nd = torch.mean(l_n + l_d)
        loss = torch.mean(loss)
        if l2 != 0:
            loss += l2 * (torch.norm(z_p) + torch.norm(z_n) + torch.norm(z_d))
        return loss, l_n, l_d, l_nd

    def loss(self, patch, neighbor, distant, margin=0.1, l2=0):
        """
        Computes loss for each batch.
        """
        z_p, z_n, z_d = (self.encode(patch), self.encode(neighbor),
            self.encode(distant))
        if self.loss_type == 'triplet':
            return self.triplet_loss(z_p, z_n, z_d, margin=margin, l2=l2)
        elif self.loss_type == 'cosine':
            return self.cosine_loss(z_p, z_n, z_d)


def ResNet18(n_classes=10, in_channels=4, z_dim=512, supervised=False,
    no_relu=False, loss_type='triplet', tile_size=50, activation='relu'):
    return ResNet(BasicBlock, [2,2,2,2,2], n_classes=n_classes,
        in_channels=in_channels, z_dim=z_dim, supervised=supervised,
        no_relu=no_relu, loss_type=loss_type, tile_size=tile_size, 
        activation=activation)

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    # Test
    net = ResNet18()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

