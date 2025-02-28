from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
from .hr_layers import *

from torch import sigmoid

class PlaneDecoder(nn.Module):
    
    def __init__(self, ch_enc = [64,128,216,288,288], scales=range(4),num_ch_enc = [ 64, 64, 128, 256, 512 ], num_output_channels=1):
        super(PlaneDecoder, self).__init__()
        self.num_output_channels = num_output_channels
        self.num_ch_enc = num_ch_enc
        self.ch_enc = ch_enc
        self.scales = scales
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])
        self.convs = nn.ModuleDict()

        # feature fusion
        self.convs["f4"] = Attention_Module(self.ch_enc[4]  , num_ch_enc[4])
        self.convs["f3"] = Attention_Module(self.ch_enc[3]  , num_ch_enc[3])
        self.convs["f2"] = Attention_Module(self.ch_enc[2]  , num_ch_enc[2])
        self.convs["f1"] = Attention_Module(self.ch_enc[1]  , num_ch_enc[1])
        
        upconv_planes = np.array([256, 128, 64, 32, 16])
        self.predict_mask2 = nn.Conv2d(upconv_planes[2], 2, kernel_size=3, padding=1)
        self.predict_mask1 = nn.Conv2d(upconv_planes[1], 2, kernel_size=3, padding=1)
        self.predict_mask0 = nn.Conv2d(upconv_planes[0], 2, kernel_size=3, padding=1)
        
        self.pconv = nn.ModuleDict()
        for i in range(5):
            num_in = self.ch_enc[i]
            num_out = self.ch_enc[i]
            self.pconv[f"conv_{i}"] = ConvBlock(num_in, num_out)

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

        for i in range(4):
            self.convs["dispconv{}".format(i)] = Conv3x3(self.num_ch_dec[i], self.num_output_channels)
                
        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

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

    def forward(self, input_features):
        outputs = {}
        feat={}
        
        for i in range(4):
            input_features[i] = self.pconv[f"conv_{i}"](input_features[i])
        
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
        
        outputs[("plane_mask", 0, 3)] = sigmoid(self.predict_mask0(feat[4]))
        outputs[("plane_mask", 0, 2)] = sigmoid(self.predict_mask1(feat[3]))
        outputs[("plane_mask", 0, 1)] = sigmoid(self.predict_mask2(feat[2]))
        
        return outputs
