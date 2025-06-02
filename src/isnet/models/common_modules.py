import jittor as jt
import jittor.nn as nn

def Norm2d(in_channels):
    """Custom Norm Function for Jittor.
    For now, maps to jt.nn.BatchNorm. 
    Can be extended if different BN types (e.g., SyncBN) are needed.
    """
    # TODO: Make this configurable via a global cfg if syncbn is needed
    return nn.BatchNorm(in_channels) # jt.nn.BatchNorm is for 2D by default

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = Norm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = Norm2d(planes)
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

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BasicBlock1(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock1, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = Norm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = Norm2d(planes)
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

        out1_conv = self.conv1(x)
        out1_bn = self.bn1(out1_conv)
        out1_relu = self.relu(out1_bn)

        out1_conv2 = self.conv2(out1_relu)
        out1_after_bn2 = self.bn2(out1_conv2)

        if self.downsample is not None:
            residual = self.downsample(x)

        # Corresponds to PyTorch: out = residual + out1 (where out1 is after bn2 but before its relu)
        out_sum = residual + out1_after_bn2 
        
        # Corresponds to PyTorch: out1 = self.relu(out1) (where out1 is after bn2)
        final_out1 = self.relu(out1_after_bn2)
        
        # Corresponds to PyTorch: out = self.relu(out)
        final_out = self.relu(out_sum)

        return final_out, final_out1

class GatedSpatialConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        super(GatedSpatialConv2d, self).__init__()

        # Parameters for the main convolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        self._gate_conv = nn.Sequential(
            Norm2d(in_channels + 1),
            nn.Conv(in_channels + 1, in_channels + 1, kernel_size=1),
            nn.ReLU(), 
            nn.Conv(in_channels + 1, 1, kernel_size=1),
            Norm2d(1),
            nn.Sigmoid()
        )
        
        self.main_conv = nn.Conv(in_channels, out_channels, kernel_size=kernel_size, 
                                 stride=stride, padding=padding, dilation=dilation, 
                                 groups=groups, bias=bias)
        
        self.reset_parameters()

    def execute(self, input_features, gating_features):
        """
        :param input_features:  [NxCxHxW] features from one branch.
        :param gating_features: [Nx1xHxW] features from another branch (1 channel).
        :return:
        """
        # Jittor uses NCHW format by default
        alphas = self._gate_conv(jt.concat([input_features, gating_features], dim=1))
        
        # (alphas + 1) ensures the multiplier is between 1 and 2
        input_features_gated = (input_features * (alphas + 1))
        
        return self.main_conv(input_features_gated)
  
    def reset_parameters(self):
        jt.init.xavier_normal_(self.main_conv.weight)
        if self.main_conv.bias is not None:
            jt.init.zeros_(self.main_conv.bias)
        
        # Initialize gate_conv parameters (optional, but good practice)
        for m in self._gate_conv.modules():
            if isinstance(m, nn.Conv):
                jt.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    jt.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm):
                jt.init.constant_(m.weight, 1)
                jt.init.constant_(m.bias, 0)


class ISNetResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample): # downsample is a boolean
        super(ISNetResidualBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            Norm2d(out_channels),
            nn.ReLU(),
            nn.Conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            Norm2d(out_channels),
        )
        if downsample:
            self.downsample_layer = nn.Sequential(
                nn.Conv(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                Norm2d(out_channels),
            )
        else:
            self.downsample_layer = nn.Sequential()

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
        out = self.body(x)
        
        residual = self.downsample_layer(residual)

        out = nn.relu(out + residual)
        return out

class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv(in_channels, inter_channels, kernel_size=3, stride=1, padding=1, bias=False),
            Norm2d(inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv(inter_channels, out_channels, kernel_size=1, stride=1, padding=0)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv):
                jt.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    jt.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm):
                jt.init.constant_(m.weight, 1)
                jt.init.constant_(m.bias, 0)
                
    def execute(self, x):
        return self.block(x)

class TFD(nn.Module):
    def __init__(self, inch, outch):
        super(TFD, self).__init__()
        self.res1 = BasicBlock1(inch, outch, stride=1, downsample=None)
        self.res2 = BasicBlock1(inch, outch, stride=1, downsample=None)
        self.gate = GatedSpatialConv2d(inch, outch)

    def execute(self,x,f_x):
        u_0 = x
        u_1, delta_u_0 = self.res1(u_0) 
        _, u_2 = self.res2(u_1) 
        
        u_3_pre = self.gate(u_2, f_x)
        u_3 = 3 * delta_u_0 + u_2 + u_3_pre
        return u_3

class SALayer(nn.Module):
    def __init__(self, channel, groups=64):
        super(SALayer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        
        self.cweight = jt.zeros((1, channel // (2 * groups), 1, 1))
        self.cbias = jt.ones((1, channel // (2 * groups), 1, 1))
        self.sweight = jt.zeros((1, channel // (2 * groups), 1, 1))
        self.sbias = jt.ones((1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(num_groups=channel // (2 * groups), num_channels=channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, -1, h, w)
        return x

    def execute(self, x):
        b, c, h, w = x.shape
        x_reshaped = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = jt.chunk(x_reshaped, 2, dim=1)

        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        out_cat = jt.concat([xn, xs], dim=1)
        out_reshaped_back = out_cat.reshape(b, -1, h, w)
        out_shuffled = SALayer.channel_shuffle(out_reshaped_back, 2)
        return out_shuffled
