import jittor as jt
import jittor.nn as nn
from .common_modules import BasicBlock, Norm2d, conv3x3

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = Norm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = Norm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = Norm2d(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv):
                jt.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm):
                jt.init.constant_(m.weight, 1)
                jt.init.constant_(m.bias, 0)

    def execute(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = Norm2d(64)
        self.relu = nn.ReLU()
        # Jittor's nn.Pool replaces MaxPool2d, AvgPool2d. op='maximum' for MaxPool
        self.maxpool = nn.Pool(kernel_size=3, stride=2, padding=1, op='maximum') 
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # op='mean' for AvgPool
        self.avgpool = nn.Pool(kernel_size=7, stride=1, op='mean', global_pool=False) # Match PyTorch default for AvgPool2d(7,stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv):
                jt.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm):
                jt.init.constant_(m.weight, 1)
                jt.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                Norm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def execute(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # For feature extraction, these might not be used by ISNet directly
        # x_features = x # Output before avgpool and fc

        x = self.avgpool(x)
        # Jittor: x.reshape(batch_size, -1) or x.reshape(x.shape[0], -1)
        x = x.reshape(x.shape[0], -1) 
        x = self.fc(x)

        return x
    
    # Helper to extract features if needed, similar to how torchvision does it.
    # ISNet accesses layers directly, so this might not be strictly necessary for ISNet.
    def extract_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = self.maxpool(x) # after maxpool
        x1 = self.layer1(x0) # layer1 output
        x2 = self.layer2(x1) # layer2 output
        x3 = self.layer3(x2) # layer3 output
        x4 = self.layer4(x3) # layer4 output
        return x0, x1, x2, x3, x4

def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    Args:
        **kwargs: Extra arguments passed to the ResNet constructor.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def resnet34(**kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def resnet50(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def resnet101(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

def resnet152(**kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model
