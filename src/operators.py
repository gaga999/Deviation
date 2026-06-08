import torch
import torch.nn as nn
import numpy as np

# =====================================================================
# 1. 物理算子定義 (CT 與 MRI)
# =====================================================================

class CTForwardOperator:
    def __init__(self, resolution=512, n_angles=1000, det_count=1000):
        """
        使用 torch-radon 實現的 CT 正向與反向投影算子
        """
        from torch_radon import Radon
        # 建立與原版相同的 0 到 2*pi 平行射束幾何
        self.angles = torch.linspace(0, 2 * np.pi, n_angles)
        self.radon = Radon(resolution, self.angles, det_count=det_count)
        self.scale = 1.0  # 後續由 Power Iteration 更新

    def forward(self, x):
        # torch-radon 支援 (B, C, H, W) 格式
        return self.radon.forward(x) * self.scale

    def adjoint(self, y):
        # Adjoint 運算即為反投影 (Backprojection)
        return self.radon.backprojection(y) * self.scale


class MRIForwardOperator:
    def __init__(self, mask):
        """
        使用 fastmri 實現的欠採樣傅立葉算子
        :param mask: 欠採樣遮罩 (Mask)，形狀與影像相同或可 broadcast
        """
        self.mask = mask  # 遮罩必須已經在 GPU 上
        self.scale = 1.0  # 後續由 Power Iteration 更新

    def forward(self, x):
        import fastmri
        # x 的形狀為 (B, 1, H, W)，是實數影像。
        # fastmri.fft2c 要求輸入最後一維是 2 (代表實部與虛部)
        x_complex = torch.stack([x, torch.zeros_like(x)], dim=-1)
        kspace = fastmri.fft2c(x_complex)
        
        # 套用欠採樣遮罩
        return kspace * self.mask

    def adjoint(self, y):
        import fastmri
        # y 的形狀為 (B, 1, H, W, 2)，是欠採樣的 k-space 資料
        masked_kspace = y * self.mask
        img_complex = fastmri.ifft2c(masked_kspace)
        
        # 為了嚴格符合「實數嵌入複數」的共軛轉置算子 (Adjoint)，
        # 我們必須取出其實部 (Real Part)，使其形狀回到 (B, 1, H, W)
        return img_complex[..., 0]

# =====================================================================
# 2. 空間與正則化算子定義 (Gradient 與 Wavelet)
# =====================================================================

class SpatialGradient:
    """
    對應原版 odl.Gradient，計算影像的水平與垂直梯度 (前向差分)
    """
    def forward(self, x):
        # x shape: (B, C, H, W)
        dx = torch.zeros_like(x)
        dy = torch.zeros_like(x)
        
        dx[..., :, :-1] = x[..., :, 1:] - x[..., :, :-1]
        dy[..., :-1, :] = x[..., 1:, :] - x[..., :-1, :]
        
        # 將兩個方向的梯度在 Channel 維度拼接，輸出形狀: (B, 2*C, H, W)
        return torch.cat([dx, dy], dim=1)

    def adjoint(self, g):
        # g shape: (B, 2*C, H, W) -> 共軛轉置運算為負散度 (Negative Divergence)
        C = g.shape[1] // 2
        dx = g[:, :C, :, :]
        dy = g[:, C:, :, :]
        
        div_x = torch.zeros_like(dx)
        div_y = torch.zeros_like(dy)
        
        div_x[..., :, 0] = -dx[..., :, 0]
        div_x[..., :, 1:-1] = dx[..., :, :-2] - dx[..., :, 1:-1]
        div_x[..., :, -1] = dx[..., :, -2]
        
        div_y[..., 0, :] = -dy[..., 0, :]
        div_y[..., 1:-1, :] = dy[..., :-2, :] - dy[..., 1:-1, :]
        div_y[..., -1, :] = dy[..., -2, :]
        
        return div_x + div_y


class WaveletOperator:
    def __init__(self, wavelet='sym5', level=5):
        """
        對應原版 ODL 小波轉換，使用 ptwt 套件實現 GPU 加速
        """
        self.wavelet = wavelet
        self.level = level

    def forward(self, x):
        import ptwt
        # ptwt.wavedec2 會回傳一個包含各層級係數的 list
        coeffs = ptwt.wavedec2(x, self.wavelet, level=self.level)
        
        # 完全重現原版 ODL 的權重縮放: W = np.power(1.8, scales) * W
        scaled_coeffs = []
        # coeffs[0] 是最粗糙的近似係數 (cAn)
        scaled_coeffs.append(coeffs[0] * (1.8 ** self.level))
        # 後續是各層的細節係數元組 (cH, cV, cD)
        for i, subbands in enumerate(coeffs[1:]):
            current_level = self.level - i
            scale_factor = 1.8 ** current_level
            scaled_coeffs.append(tuple(s * scale_factor for s in subbands))
        return scaled_coeffs

    def adjoint(self, scaled_coeffs):
        import ptwt
        # 進行反向的權重縮放
        coeffs = []
        coeffs.append(scaled_coeffs[0] / (1.8 ** self.level))
        for i, subbands in enumerate(scaled_coeffs[1:]):
            current_level = self.level - i
            scale_factor = 1.8 ** current_level
            coeffs.append(tuple(s / scale_factor for s in subbands))
        return ptwt.waverec2(coeffs, self.wavelet)

# =====================================================================
# 3. 工具函數：冪迭代法 (Power Iteration) 計算算子範數
# =====================================================================

def estimate_operator_norm(operator, input_shape, iterations=10, device='cuda'):
    """
    自動計算並設定物理算子的範數，確保最佳化演算法的收斂性
    """
    x = torch.randn(input_shape, device=device)
    x = x / torch.norm(x)
    for _ in range(iterations):
        y = operator.forward(x)
        x = operator.adjoint(y)
        norm = torch.norm(x)
        x = x / norm
    return torch.sqrt(norm).item()

# =====================================================================
# 4. 近端算子 (Proximal Operators) 數學操作保留
# =====================================================================

def prox_l1(x, alpha):
    """
    L1 軟閾值函數 (Soft-thresholding)。
    支援多層小波結構 (list/tuple) 或單一張量 (Tensor)。
    """
    if isinstance(x, list):
        return [prox_l1(item, alpha) for item in x]
    elif isinstance(x, tuple):
        return tuple(prox_l1(item, alpha) for item in x)
    return torch.sign(x) * torch.clamp(torch.abs(x) - alpha, min=0)

def structural_subtract(a, b):
    """
    輔助函數：對具有相同結構的 Tensor 或小波 list 進行減法
    """
    if isinstance(a, list):
        return [structural_subtract(ai, bi) for ai, bi in zip(a, b)]
    elif isinstance(a, tuple):
        return tuple(structural_subtract(ai, bi) for ai, bi in zip(a, b))
    return a - b

def prox(x, W, W_adj, gamma, lam, mu):
    """
    論文的核心近端優化操作，完全保留原始數學結構：
    res = x + 1/mu * W_adj(prox_l1(W(x), alpha) - W(x))
    """
    y = W.forward(x)
    alpha = lam * mu * gamma
    
    # 算子內部的減法
    diff = structural_subtract(prox_l1(y, alpha), y)
    
    # 共軛轉置後更新 x
    return x + (1.0 / mu) * W_adj.adjoint(diff)

# =====================================================================
# 5. 統一的管理接口
# =====================================================================

def get_operators(task_type="CT", regularizer_type="smooth", mask=None, device="cuda"):
    """
    用來取代原版的 operators_smooth() 與 operators_nonsmooth()
    """
    # 1. 初始化物理算子
    if task_type == "CT":
        T = CTForwardOperator(resolution=512, n_angles=1000, det_count=1000)
        input_shape = (1, 1, 512, 512)
    elif task_type == "MRI":
        assert mask is not None, "MRI task requires a sampling mask!"
        T = MRIForwardOperator(mask=mask)
        input_shape = (1, 1, mask.shape[-2], mask.shape[-1])
    else:
        raise ValueError("Unknown task_type")

    # 使用 Power Iteration 計算 T 的範數並縮放 (比照原版 T = (1 / T_norm) * T)
    T_norm = estimate_operator_norm(T, input_shape, device=device)
    T.scale = 1.0 / T_norm

    # 2. 初始化正則化算子
    if regularizer_type == "smooth":
        W = SpatialGradient()
    elif regularizer_type == "nonsmooth":
        W = WaveletOperator(wavelet='sym5', level=5)
    else:
        raise ValueError("Unknown regularizer_type")

    # W 的共軛轉置就是它自己內建的 adjoint 方法
    return T, W