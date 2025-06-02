import jittor as jt
import jittor.nn as nn
from .common_modules import (
    Norm2d, BasicBlock, ISNetResidualBlock, 
    FCNHead, TFD, SALayer, GatedSpatialConv2d
)

class TTOA_stub(nn.Module):
    """Stub for TTOA module. Replaces DCNv2-based TTOA for initial porting."""
    def __init__(self, low_channels, high_channels):
        super(TTOA_stub, self).__init__()
        # This stub will simply add the two features. 
        # Add a 1x1 conv if channel alignment is needed, but original TTOA implies alignment.
        # For ISNet, TTOA is called with same low_channels and high_channels.
        # print(f"Warning: Using TTOA_stub for channels {low_channels}. Actual TTOA functionality is not implemented.")

    def execute(self, x_l, x_h):
        # A simple element-wise sum as a placeholder for the DCN-based aggregation.
        # This assumes x_l and x_h have compatible shapes for addition.
        return x_l + x_h

class ISNet(nn.Module):
    def __init__(self, layer_blocks, channels):
        super(ISNet, self).__init__()

        stem_width = int(channels[0]) # e.g., 16
        self.stem = nn.Sequential(
            Norm2d(3), # BatchNorm before first conv
            nn.Conv(3, stem_width, kernel_size=3, stride=2, padding=1, bias=False),
            Norm2d(stem_width),
            nn.ReLU(),
            nn.Conv(stem_width, stem_width, kernel_size=3, stride=1, padding=1, bias=False),
            Norm2d(stem_width),
            nn.ReLU(),
            nn.Conv(stem_width, 2 * stem_width, kernel_size=3, stride=1, padding=1, bias=False),
            Norm2d(2 * stem_width),
            nn.ReLU(),
            nn.Pool(kernel_size=3, stride=2, padding=1, op='maximum') # MaxPool2d
        )
        
        # Use TTOA_stub instead of the DCNv2-based TTOA
        self.TTOA_low = TTOA_stub(channels[2], channels[2]) # e.g., TTOA_stub(64, 64)
        self.TTOA_high = TTOA_stub(channels[1], channels[1]) # e.g., TTOA_stub(32, 32)
        
        # Main backbone layers using ISNetResidualBlock
        self.layer1 = self._make_layer(block=ISNetResidualBlock, block_num=layer_blocks[0],
                                       in_channels=channels[1], out_channels=channels[1], stride=1)
        self.layer2 = self._make_layer(block=ISNetResidualBlock, block_num=layer_blocks[1],
                                       in_channels=channels[1], out_channels=channels[2], stride=2)
        self.layer3 = self._make_layer(block=ISNetResidualBlock, block_num=layer_blocks[2],
                                       in_channels=channels[2], out_channels=channels[3], stride=2)

        self.deconv2 = nn.ConvTranspose(channels[3], channels[2], kernel_size=4, stride=2, padding=1)
        self.uplayer2 = self._make_layer(block=ISNetResidualBlock, block_num=layer_blocks[1],
                                         in_channels=channels[2], out_channels=channels[2], stride=1)

        self.deconv1 = nn.ConvTranspose(channels[2], channels[1], kernel_size=4, stride=2, padding=1)
        self.uplayer1 = self._make_layer(block=ISNetResidualBlock, block_num=layer_blocks[0],
                                         in_channels=channels[1], out_channels=channels[1], stride=1)

        self.head = FCNHead(channels[1], 1) # Output 1 channel for main segmentation
        
        # Edge branch components
        self.dsn1 = nn.Conv(channels[3], 1, kernel_size=1) # from c3 (channels[3])
        self.dsn2 = nn.Conv(channels[2], 1, kernel_size=1) # from upc2 (channels[2])
        self.dsn3 = nn.Conv(channels[1], 1, kernel_size=1) # from upc1 (channels[1])

        # These Resnet.BasicBlock are from common_modules (originally from Resnet.py)
        self.res1 = BasicBlock(64, 64, stride=1, downsample=None)
        self.d1 = nn.Conv(64, 32, kernel_size=1)
        self.res2 = BasicBlock(32, 32, stride=1, downsample=None)
        self.d2 = nn.Conv(32, 16, kernel_size=1)
        self.res3 = BasicBlock(16, 16, stride=1, downsample=None)
        self.d3 = nn.Conv(16, 8, kernel_size=1) # This was not in original ISNet.py, but makes sense for symmetry
                                               # Original ISNet.py d3 was Conv2d(16,8,1). Added for clarity.

        self.fuse = nn.Conv(64, 1, kernel_size=1, padding=0, bias=False)
        self.cw = nn.Conv(4, 1, kernel_size=1, padding=0, bias=False) # For commented-out SA part

        self.gate1 = GatedSpatialConv2d(32, 32)
        self.gate2 = GatedSpatialConv2d(16, 16)
        self.gate3 = GatedSpatialConv2d(8, 8) # Not used in current forward, but defined in PyTorch version

        self.sigmoid = nn.Sigmoid()
        self.SA = SALayer(4, 2) # For commented-out SA part, channel=4
        self.SA_att = SALayer(17,2) # Not used in current forward, but defined
        
        self.dsup = nn.Conv(3, 64, kernel_size=1) # For x_grad processing
        self.head2 = FCNHead(channels[1], 3) # Not used in current forward, but defined
        self.conv2_1 = nn.Conv(3,1,kernel_size=1) # Not used
        self.conv16 = nn.Conv(3,16,kernel_size=1) # Not used

        self.myb1 = TFD(64,64)
        self.myb2 = TFD(64,64)
        self.myb3 = TFD(64,64)
        
        self._initialize_weights()

    def _make_layer(self, block, block_num, in_channels, out_channels, stride):
        layers = []
        downsample = (in_channels != out_channels) or (stride != 1)
        layers.append(block(in_channels, out_channels, stride, downsample))
        for _ in range(block_num - 1):
            layers.append(block(out_channels, out_channels, 1, False))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv) or isinstance(m, nn.ConvTranspose):
                jt.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    jt.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm): # Covers Norm2d if it's an alias for BatchNorm
                jt.init.constant_(m.weight, 1)
                jt.init.constant_(m.bias, 0)
            # BasicBlock, ISNetResidualBlock, FCNHead, TFD, GatedSpatialConv2d have their own init

    def execute(self, x, x_grad=None):
        # 基于原始 PyTorch 版本的实现
        # 如果没有提供 x_grad，使用简单替代方案
        if x_grad is None:
            x_grad = x
        
        # 获取输入尺寸
        hei, wid = x.shape[2], x.shape[3]

        # Stem 和主干特征提取
        c0_s = self.stem(x)
        c1 = self.layer1(c0_s)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)

        # 解码器路径与 TTOA 实现
        deconc2 = self.deconv2(c3)
        fusec2 = self.TTOA_low(deconc2, c2) 
        upc2 = self.uplayer2(fusec2)

        deconc1 = self.deconv1(upc2)
        fusec1 = self.TTOA_high(deconc1, c1)
        upc1_out = self.uplayer1(fusec1)

        # 边缘分支处理 - 与原始 PyTorch 版本一致
        # DSN (深度监督网络) 边缘输出
        s1 = nn.interpolate(self.dsn1(c3), size=[hei, wid], mode='bilinear', align_corners=True)
        s2 = nn.interpolate(self.dsn2(upc2), size=[hei, wid], mode='bilinear', align_corners=True)
        s3 = nn.interpolate(self.dsn3(upc1_out), size=[hei, wid], mode='bilinear', align_corners=True)

        # 处理 x_grad (边缘检测输入)
        m1f = nn.interpolate(x_grad, size=[hei, wid], mode='bilinear', align_corners=True)
        m1f = self.dsup(m1f)

        # TFD 模块处理 - 与原始实现一致
        cs1 = self.myb1(m1f, s1)
        cs2 = self.myb2(cs1, s2)
        cs = self.myb3(cs2, s3)
        
        # 边缘输出处理
        cs = self.fuse(cs)
        cs = nn.interpolate(cs, size=[hei, wid], mode='bilinear', align_corners=True)
        edge_out = self.sigmoid(cs)

        # 与原始 PyTorch 版本一致的处理方式 - 此处遵循原始注释的 SA 块
        # 原始代码中这部分是被注释的，计算梯度时也不会经过这些层
        # cat = jt.concat([edge_out, x_grad], dim=1)
        # cat = self.SA(cat)
        # acts = self.cw(cat)
        # acts = self.sigmoid(acts)
        
        # 特征融合 - 完全遵循原始实现
        upc1_interpolated = nn.interpolate(upc1_out, size=[hei, wid], mode='bilinear', align_corners=True)
        # 原始实现: fuse = edge_out * upc1 + upc1
        fused_features = edge_out * upc1_interpolated + upc1_interpolated

        # 最终预测头
        pred = self.head(fused_features)
        main_out = nn.interpolate(pred, size=[hei, wid], mode='bilinear', align_corners=True)
        
        return main_out, edge_out
