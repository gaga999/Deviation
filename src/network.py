import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnedOptimizerNet(nn.Module):
    def __init__(self, in_channels, n_layers, n_out=1, filters=32, kernel_size=3):
        """
        對應原版的 convnet 架構
        :param in_channels: 輸入的通道數 (在優化過程中可能會把影像、梯度等 concat 起來)
        :param n_layers: 隱藏卷積層的數量
        :param n_out: 輸出的通道數 (通常是 1，代表預測的更新步長或梯度)
        :param filters: 每層的通道數 (原版預設 32)
        :param kernel_size: 卷積核大小 (原版預設 3)
        """
        super(LearnedOptimizerNet, self).__init__()
        self.n_layers = n_layers
        
        # 1. 初始的 Instance Normalization
        # affine=True 代表會有可學習的縮放(gamma)與平移(beta)參數
        self.initial_inst_norm = nn.InstanceNorm2d(in_channels, affine=True)
        
        # 使用 ModuleList 來裝載動態數量的層
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # 建立隱藏層
        current_in_channels = in_channels
        for i in range(n_layers):
            # padding='same' 在 PyTorch 1.9+ 已經原生支援
            conv = nn.Conv2d(current_in_channels, filters, kernel_size, padding='same', bias=True)
            
            # 對應 TF 的 variance_scaling_initializer (類似 He initialization)
            nn.init.kaiming_normal_(conv.weight, mode='fan_in', nonlinearity='leaky_relu')
            self.convs.append(conv)
            
            # 加入 Instance Norm
            self.norms.append(nn.InstanceNorm2d(filters, affine=True))
            
            current_in_channels = filters
            
        # 2. 最終輸出層
        self.final_conv = nn.Conv2d(current_in_channels, n_out, kernel_size, padding='same', bias=True)
        # 輸出層不經過 LeakyReLU，所以初始化參數用 linear
        nn.init.kaiming_normal_(self.final_conv.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, x):
        """
        x 的維度必須是 (Batch, Channels, Height, Width)
        """
        # 第一步：對輸入做正規化
        x = self.initial_inst_norm(x)
        
        # 逐層 Forward
        for i in range(self.n_layers):
            x = self.convs[i](x)
            x = self.norms[i](x)
            # 原版 TF activation 為 leaky_relu (預設 negative_slope 通常是 0.2)
            x = F.leaky_relu(x, negative_slope=0.2)
            
        # 最後一層卷積輸出
        result = self.final_conv(x)
        return result