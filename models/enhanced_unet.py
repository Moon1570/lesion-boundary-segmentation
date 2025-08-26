"""
Enhanced U-Net architectures for better segmentation performance
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """Attention Gate for U-Net to focus on relevant features"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class ResidualBlock(nn.Module):
    """Residual block for deeper networks"""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class AttentionUNet(nn.Module):
    """U-Net with Attention Gates for improved segmentation"""
    def __init__(self, n_channels=3, n_classes=1, channels=[64, 128, 256, 512, 1024]):
        super(AttentionUNet, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.channels = channels
        
        # Encoder
        self.enc1 = self._make_encoder_block(n_channels, channels[0])
        self.enc2 = self._make_encoder_block(channels[0], channels[1])
        self.enc3 = self._make_encoder_block(channels[1], channels[2])
        self.enc4 = self._make_encoder_block(channels[2], channels[3])
        
        # Bottleneck
        self.bottleneck = self._make_encoder_block(channels[3], channels[4])
        
        # Attention Gates
        self.att4 = AttentionGate(channels[4], channels[3], channels[3]//2)
        self.att3 = AttentionGate(channels[3], channels[2], channels[2]//2)
        self.att2 = AttentionGate(channels[2], channels[1], channels[1]//2)
        self.att1 = AttentionGate(channels[1], channels[0], channels[0]//2)
        
        # Decoder
        self.dec4 = self._make_decoder_block(channels[4] + channels[3], channels[3])
        self.dec3 = self._make_decoder_block(channels[3] + channels[2], channels[2])
        self.dec2 = self._make_decoder_block(channels[2] + channels[1], channels[1])
        self.dec1 = self._make_decoder_block(channels[1] + channels[0], channels[0])
        
        # Final classifier
        self.final = nn.Conv2d(channels[0], n_classes, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels),
            ResidualBlock(out_channels, out_channels)
        )
    
    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        
        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))
        
        # Decoder with attention
        d4 = F.interpolate(b, scale_factor=2, mode='bilinear', align_corners=True)
        e4_att = self.att4(d4, e4)
        d4 = torch.cat([d4, e4_att], dim=1)
        d4 = self.dec4(d4)
        
        d3 = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True)
        e3_att = self.att3(d3, e3)
        d3 = torch.cat([d3, e3_att], dim=1)
        d3 = self.dec3(d3)
        
        d2 = F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True)
        e2_att = self.att2(d2, e2)
        d2 = torch.cat([d2, e2_att], dim=1)
        d2 = self.dec2(d2)
        
        d1 = F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True)
        e1_att = self.att1(d1, e1)
        d1 = torch.cat([d1, e1_att], dim=1)
        d1 = self.dec1(d1)
        
        # Final output
        out = self.final(d1)
        return out


class UNetPlusPlus(nn.Module):
    """U-Net++ (Nested U-Net) for improved feature propagation"""
    def __init__(self, n_channels=3, n_classes=1, deep_supervision=True):
        super(UNetPlusPlus, self).__init__()
        
        self.deep_supervision = deep_supervision
        
        # Encoder blocks
        self.conv0_0 = self._conv_block(n_channels, 32)
        self.conv1_0 = self._conv_block(32, 64)
        self.conv2_0 = self._conv_block(64, 128)
        self.conv3_0 = self._conv_block(128, 256)
        self.conv4_0 = self._conv_block(256, 512)
        
        # Nested blocks
        self.conv0_1 = self._conv_block(32 + 64, 32)
        self.conv1_1 = self._conv_block(64 + 128, 64)
        self.conv2_1 = self._conv_block(128 + 256, 128)
        self.conv3_1 = self._conv_block(256 + 512, 256)
        
        self.conv0_2 = self._conv_block(32*2 + 64, 32)
        self.conv1_2 = self._conv_block(64*2 + 128, 64)
        self.conv2_2 = self._conv_block(128*2 + 256, 128)
        
        self.conv0_3 = self._conv_block(32*3 + 64, 32)
        self.conv1_3 = self._conv_block(64*3 + 128, 64)
        
        self.conv0_4 = self._conv_block(32*4 + 64, 32)
        
        # Final layers
        if self.deep_supervision:
            self.final1 = nn.Conv2d(32, n_classes, 1)
            self.final2 = nn.Conv2d(32, n_classes, 1)
            self.final3 = nn.Conv2d(32, n_classes, 1)
            self.final4 = nn.Conv2d(32, n_classes, 1)
        else:
            self.final = nn.Conv2d(32, n_classes, 1)
    
    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(F.max_pool2d(x0_0, 2))
        x0_1 = self.conv0_1(torch.cat([x0_0, F.interpolate(x1_0, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        
        x2_0 = self.conv2_0(F.max_pool2d(x1_0, 2))
        x1_1 = self.conv1_1(torch.cat([x1_0, F.interpolate(x2_0, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, F.interpolate(x1_1, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        
        x3_0 = self.conv3_0(F.max_pool2d(x2_0, 2))
        x2_1 = self.conv2_1(torch.cat([x2_0, F.interpolate(x3_0, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, F.interpolate(x2_1, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, F.interpolate(x1_2, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        
        x4_0 = self.conv4_0(F.max_pool2d(x3_0, 2))
        x3_1 = self.conv3_1(torch.cat([x3_0, F.interpolate(x4_0, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, F.interpolate(x3_1, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, F.interpolate(x2_2, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, F.interpolate(x1_3, scale_factor=2, mode='bilinear', align_corners=True)], 1))
        
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        else:
            return self.final(x0_4)
