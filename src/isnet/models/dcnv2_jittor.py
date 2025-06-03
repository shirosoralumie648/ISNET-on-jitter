import jittor as jt
import jittor.nn as nn
import math

class DCNv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=1, dilation=1, deformable_groups=1, bias=True):
        super(DCNv2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.deformable_groups = deformable_groups
        self.with_bias = bias

        self.weight = jt.zeros((out_channels, in_channels, *self.kernel_size))
        
        if self.with_bias:
            self.bias = jt.zeros(out_channels)
        else:
            self.bias = None

        self.conv_offset_mask = nn.Conv2d(
            in_channels, 
            deformable_groups * 3 * self.kernel_size[0] * self.kernel_size[1],
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True
        )
        
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        jt.init.uniform_(self.weight, -stdv, stdv)
        if self.bias is not None:
            jt.init.uniform_(self.bias, -stdv, stdv)
        
        jt.init.constant_(self.conv_offset_mask.weight, 0.)
        jt.init.constant_(self.conv_offset_mask.bias, 0.)
        
    def execute(self, x):
        offset_mask = self.conv_offset_mask(x)
        
        kh, kw = self.kernel_size
        dg = self.deformable_groups
        offset_channels = dg * 2 * kh * kw

        o1 = offset_mask[:, :dg * kh * kw, :, :]
        o2 = offset_mask[:, dg * kh * kw : offset_channels, :, :]
        mask = offset_mask[:, offset_channels:, :, :]
        
        offset = jt.contrib.concat([o1, o2], dim=1)
        mask = jt.sigmoid(mask)

        output = jt.ops.deform_conv2d(
            x,
            offset,
            mask,
            self.weight,
            self.bias if self.with_bias else None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=1, 
            deformable_groups=self.deformable_groups
        )
        return output

class TTOA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, deformable_groups=1):
        super(TTOA, self).__init__()
        self.dcn = DCNv2(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            deformable_groups=deformable_groups,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def execute(self, x):
        return self.relu(self.bn(self.dcn(x)))
