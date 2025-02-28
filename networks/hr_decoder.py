from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
from .hr_layers import *

class DepthDecoder(nn.Module):
    
    def __init__(self, ch_enc = [64,128,216,288,288], scales=range(4),num_ch_enc = [ 64, 64, 128, 256, 512 ], num_output_channels=1):
        super(DepthDecoder, self).__init__()
        self.num_output_channels = num_output_channels
        self.num_ch_enc = num_ch_enc
        self.ch_enc = ch_enc
        self.scales = scales
        self.max_depth = 10
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.convs = nn.ModuleDict()

        # feature fusion
        self.convs["f4"] = Attention_Module(self.ch_enc[4]  , num_ch_enc[4])
        self.convs["f3"] = Attention_Module(self.ch_enc[3]  , num_ch_enc[3])
        self.convs["f2"] = Attention_Module(self.ch_enc[2]  , num_ch_enc[2])
        self.convs["f1"] = Attention_Module(self.ch_enc[1]  , num_ch_enc[1])

        self.all_position = ["01", "11", "21", "31", "02", "12", "22", "03", "13", "04"]
        self.attention_position = ["31", "22", "13", "04"]
        self.non_attention_position = ["01", "11", "21", "02", "12", "03"]
            
        for j in range(5):
            for i in range(5 - j):
                # upconv 0
                num_ch_in = num_ch_enc[i]
                if i == 0 and j != 0:
                    num_ch_in /= 2
                num_ch_out = num_ch_in / 2
                self.convs["X_{}{}_Conv_0".format(i, j)] = ConvBlock(num_ch_in, num_ch_out)

                # X_04 upconv 1, only add X_04 convolution
                if i == 0 and j == 4:
                    num_ch_in = num_ch_out
                    num_ch_out = self.num_ch_dec[i]
                    self.convs["X_{}{}_Conv_1".format(i, j)] = ConvBlock(num_ch_in, num_ch_out)

        # declare fSEModule and original module
        for index in self.attention_position:
            row = int(index[0])
            col = int(index[1])
            self.convs["X_" + index + "_attention"] = fSEModule(num_ch_enc[row + 1] // 2, self.num_ch_enc[row]
                                                                         + self.num_ch_dec[row + 1] * (col - 1))
        for index in self.non_attention_position:
            row = int(index[0])
            col = int(index[1])
            if col == 1:
                self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvBlock(num_ch_enc[row + 1] // 2 +
                                                                        self.num_ch_enc[row], self.num_ch_dec[row + 1])
            else:
                self.convs["X_"+index+"_downsample"] = Conv1x1(num_ch_enc[row+1] // 2 + self.num_ch_enc[row]
                                                                        + self.num_ch_dec[row+1]*(col-1), self.num_ch_dec[row + 1] * 2)
                self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)] = ConvBlock(self.num_ch_dec[row + 1] * 2, self.num_ch_dec[row + 1])

        for i in range(5):
            self.convs["dispconv{}".format(i)] = Conv3x3(self.num_ch_dec[i], self.num_output_channels)
                
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        
        self.normal_head0 = NormalHead(input_dim = self.num_ch_dec[4])
        self.distance_head0 = DistanceHead(input_dim = self.num_ch_dec[4])
        self.normal_head1 = NormalHead(input_dim = self.num_ch_dec[3])
        self.distance_head1 = DistanceHead(input_dim = self.num_ch_dec[3])
        self.normal_head2 = NormalHead(input_dim = self.num_ch_dec[2])
        self.distance_head2 = DistanceHead(input_dim = self.num_ch_dec[2])
        self.normal_head3 = NormalHead(input_dim = self.num_ch_dec[1])
        self.distance_head3 = DistanceHead(input_dim = self.num_ch_dec[1])
        
        self.depthoffset = DistanceHead(input_dim=self.num_ch_dec[0])

    def nestConv(self, conv, high_feature, low_features):
        conv_0 = conv[0]
        conv_1 = conv[1]
        assert isinstance(low_features, list)
        high_features = [upsample(conv_0(high_feature))]
        for feature in low_features:
            high_features.append(feature)
        high_features = torch.cat(high_features, 1)
        if len(conv) == 3:
            high_features = conv[2](high_features)
        return conv_1(high_features)

    def forward(self, input_features, inv_K):
        outputs = {}
        feat = {}
        
        feat[4] = self.convs["f4"](input_features[4])
        feat[3] = self.convs["f3"](input_features[3])
        feat[2] = self.convs["f2"](input_features[2])
        feat[1] = self.convs["f1"](input_features[1])
        feat[0] = input_features[0]
        
        features = {}
        for i in range(5):
            features["X_{}0".format(i)] = feat[i]
        # Network architecture
        for index in self.all_position:
            row = int(index[0])
            col = int(index[1])

            low_features = []
            for i in range(col):
                low_features.append(features["X_{}{}".format(row, i)])

            # add fSE block to decoder
            if index in self.attention_position:
                features["X_"+index] = self.convs["X_" + index + "_attention"](
                    self.convs["X_{}{}_Conv_0".format(row+1, col-1)](features["X_{}{}".format(row+1, col-1)]), low_features)
            elif index in self.non_attention_position:
                conv = [self.convs["X_{}{}_Conv_0".format(row + 1, col - 1)],
                        self.convs["X_{}{}_Conv_1".format(row + 1, col - 1)]]
                if col != 1:
                    conv.append(self.convs["X_" + index + "_downsample"])
                features["X_" + index] = self.nestConv(conv, features["X_{}{}".format(row+1, col-1)], low_features)

        x = features["X_04"]
        x = self.convs["X_04_Conv_0"](x)
        x = self.convs["X_04_Conv_1"](upsample(x))
        
        feat[0] = x
        feat[1] = features["X_04"]
        feat[2] = features["X_13"]
        feat[3] = features["X_22"]
        feat[4] = features["X_31"]
        
        n0 = F.normalize(self.normal_head0(feat[4], 1), dim=1, p=2)
        dist0 = self.distance_head0(feat[4], 1)
        
        # n1 = F.normalize(self.normal_head1(feat[3], 1), dim=1, p=2)
        # dist1 = self.distance_head1(feat[3], 1)
        
        n1_offset = F.normalize(self.normal_head1(feat[3], 1), dim=1, p=2)
        n1 = F.normalize(n1_offset + upsample(n0, scale_factor=2), dim=1, p=2)
        dist1_offset = (self.distance_head1(feat[3],1)-0.5)*2
        dist1 = dist1_offset + upsample(dist0, scale_factor=2)
        
        n2_offset = F.normalize(self.normal_head2(feat[2], 1), dim=1, p=2)
        n2 = F.normalize(n2_offset + upsample(n1, scale_factor=2), dim=1, p=2)
        dist2_offset = (self.distance_head2(feat[2],1)-0.5)*2
        dist2 = dist2_offset + upsample(dist1, scale_factor=2)
        
        # n3_offset = F.normalize(self.normal_head3(feat[1], 1), dim=1, p=2)
        # n3 = F.normalize(n3_offset + upsample(n2, scale_factor=2), dim=1, p=2)
        # dist3_offset = (self.distance_head3(feat[1],1)-0.5)*2
        # dist3 = dist3_offset + upsample(dist2, scale_factor=2)
        
        normals_and_distances = [(n0, dist0), (n1, dist1), (n2, dist2)]

        for i, (n, dist) in enumerate(normals_and_distances):
            b, c, h, w =  n.shape 
            device = n.device
            dn_to_depth = DN_to_depth(b, h, w).to(device)
            distance = dist * self.max_depth
            depth = dn_to_depth(n, distance, inv_K).clamp(0, self.max_depth)
            
            outputs[("depth", 0, 3-i)] = upsample(depth, scale_factor=2**(4-i))
        
        depth_offset = (self.depthoffset(feat[0],1)-0.5)*20
        
        outputs[("depth", 0, 0)] = (depth_offset + outputs[("depth", 0, 1)]).clamp(0, self.max_depth)

        outputs[("disp", 0)] = self.sigmoid(self.convs["dispconv0"](feat[0]))
        # outputs[("disp", 1)] = self.sigmoid(self.convs["dispconv1"](feat[1]))
        outputs[("disp", 1)] = self.sigmoid(self.convs["dispconv2"](feat[2]))
        outputs[("disp", 2)] = self.sigmoid(self.convs["dispconv3"](feat[3]))
        outputs[("disp", 3)] = self.sigmoid(self.convs["dispconv4"](feat[4]))
        
        return outputs

class NormalHead(nn.Module):
    def __init__(self, input_dim=100):
        super(NormalHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 3, 3, padding=1)
       
    def forward(self, x, scale):
        x = self.conv1(x)
        if scale > 1:
            x = upsample(x, scale_factor=scale)
        return x

class DistanceHead(nn.Module):
    def __init__(self, input_dim=100):
        super(DistanceHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, scale):
        x = self.sigmoid(self.conv1(x))
        if scale > 1:
            x = upsample(x, scale_factor=scale)
        return x
    
class DN_to_depth(nn.Module):
    """Layer to transform distance and normal into depth
    """
    def __init__(self, batch_size, height, width):
        super(DN_to_depth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32) # 2, H, W    
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords), requires_grad=False) # 2, H, W  

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False) # B, 1, H, W

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0) # 1, 2, L
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1) # B, 2, L
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1), requires_grad=False) # B, 3, L

    def forward(self, norm_normal, distance, inv_K):
        normalized_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        normalized_points = normalized_points.reshape(self.batch_size, 3, self.height, self.width)
        normal_points = (norm_normal * normalized_points).sum(1, keepdim=True)
        depth = distance / (normal_points + 1e-7)
        return depth.abs()
